// Copyright 2026 Weitang Ye
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * @FilePath: /Artea/include/artea/cpu/router/data_structures/lnbr_candidate_entry.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Lnbr-flavored candidate entry (layer_vid + explored bit +
 *               base_vid + distance, 12 bytes). Carries both identifiers an
 *               internal-graph beam search needs without a side-channel
 *               lookup table.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

#include <artea/cpu/router/data_structures/candidate_entry_concept.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Lnbr-flavored candidate entry with embedded explored/unexplored status.
 *
 * Unlike @c DnbrCandidateEntry, this entry stores BOTH the dedup key
 * (@c layer_vid — the vertex's id inside the current layer's CSR) and the
 * navigation key (@c base_vid — the vertex's id in the base dataset).
 * The former is used by the queue's visited table and the neighbor-block
 * lookup; the latter is used to fetch the actual coordinates from the base
 * vector array for distance computation and to carry through inter-layer
 * descent.
 *
 * Keeping both fields in the entry itself removes the need for a side
 * @c std::unordered_map<layer_vid, base_vid> that the original stacked_rgraph
 * beam-search loop maintained.
 *
 * Bit Layout of @c layer_vid_and_status (mirrors DnbrCandidateEntry exactly):
 * - Bit 31: Explored status (1 = explored, 0 = unexplored)
 * - Bits 0-30: layer_vid (supports up to 2^31 - 1 vertices per layer)
 *
 * @tparam RouterTraitsT Traits defining vertex types and distance types.
 */
template <typename RouterTraitsT>
struct alignas(8) LnbrCandidateEntry {   // 12 bytes useful, 16 aligned

    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;
    static constexpr distance_t max_distance = RouterTraitsT::max_distance;
    static constexpr distance_t min_distance = RouterTraitsT::min_distance;

    static_assert(sizeof(vertex_num_t) == 4, "vertex_num_t must be 4 bytes.");
    static_assert(std::is_unsigned<vertex_num_t>::value, "vertex_num_t must be unsigned (e.g., uint32_t).");
    static_assert(sizeof(vec_ele_t) == 4, "vec_ele_t must be 4 bytes.");

    // Bit masks (byte-identical to DnbrCandidateEntry):
    static constexpr vertex_id_t MASK_STATUS_EXPLORED = 0x80000000; // 1000...
    static constexpr vertex_id_t MASK_ID              = 0x7FFFFFFF; // 0111...

    /** @brief layer_vid (bits 0-30) + explored status (bit 31). */
    vertex_id_t layer_vid_and_status;   // 4 bytes

    /** @brief base dataset id for coordinate lookup. */
    vertex_id_t base_vid;               // 4 bytes

    /** @brief Distance to the candidate entry. */
    distance_t  distance;               // 4 bytes

private:
    static constexpr auto compute_lv_and_status(vertex_id_t layer_vid, bool is_explored) -> vertex_id_t {
        return is_explored ? ((layer_vid & MASK_ID) | MASK_STATUS_EXPLORED) : (layer_vid & MASK_ID);
    }

public:
    constexpr LnbrCandidateEntry()
        : layer_vid_and_status(invalid_vertex_id),
          base_vid(invalid_vertex_id),
          distance(0.0) {}

    constexpr LnbrCandidateEntry(const LnbrCandidateEntry&) = default;
    constexpr LnbrCandidateEntry& operator=(const LnbrCandidateEntry&) = default;
    LnbrCandidateEntry(LnbrCandidateEntry&&) = default;
    LnbrCandidateEntry& operator=(LnbrCandidateEntry&&) = default;
    ~LnbrCandidateEntry() = default;

    /** @brief Create a new LnbrCandidateEntry (unexplored by default). */
    constexpr LnbrCandidateEntry(
        const vertex_id_t base_vid_,
        const vertex_id_t layer_vid_,
        const distance_t  distance_
    ) : layer_vid_and_status(layer_vid_ & MASK_ID),
        base_vid(base_vid_),
        distance(distance_) {}

    /** @brief Create a new LnbrCandidateEntry with explored status. */
    constexpr LnbrCandidateEntry(
        const vertex_id_t base_vid_,
        const vertex_id_t layer_vid_,
        const distance_t  distance_,
        const bool is_explored
    ) : layer_vid_and_status(compute_lv_and_status(layer_vid_, is_explored)),
        base_vid(base_vid_),
        distance(distance_) {}

    static constexpr auto make_invalid_entry() -> LnbrCandidateEntry {
        return LnbrCandidateEntry(invalid_vertex_id, invalid_vertex_id, max_distance);
    }

    /** @brief Create a sentinel entry with minimum distance (for max-heap sentinel). */
    static constexpr auto make_min_entry() -> LnbrCandidateEntry {
        return LnbrCandidateEntry(invalid_vertex_id, invalid_vertex_id, min_distance);
    }

    // --- Concept-required API ---

    /** @brief Get the base dataset ID (= base_vid). */
    __attribute__((always_inline))
    auto get_base_id() const -> vertex_id_t {
        return base_vid;
    }

    /** @brief Get the layer-local ID (= layer_vid, with explored bit masked out). */
    __attribute__((always_inline))
    auto get_layer_id() const -> vertex_id_t {
        return static_cast<vertex_id_t>(layer_vid_and_status & MASK_ID);
    }

    /** @brief Check if this entry is an invalid/padding sentinel. */
    __attribute__((always_inline))
    auto is_invalid() const -> bool {
        return (layer_vid_and_status & MASK_ID) == MASK_ID;
    }

    __attribute__((always_inline))
    auto is_explored() const -> bool {
        return (layer_vid_and_status & MASK_STATUS_EXPLORED) != 0;
    }

    __attribute__((always_inline))
    auto is_unexplored() const -> bool {
        return (layer_vid_and_status & MASK_STATUS_EXPLORED) == 0;
    }

    __attribute__((always_inline))
    auto mark_as_explored() -> void {
        layer_vid_and_status |= MASK_STATUS_EXPLORED;
    }

    __attribute__((always_inline))
    auto mark_as_unexplored() -> void {
        layer_vid_and_status &= (~MASK_STATUS_EXPLORED);
    }

    __attribute__((always_inline))
    auto set_explored_status(const bool is_explored) -> void {
        if (is_explored) mark_as_explored();
        else mark_as_unexplored();
    }

    __attribute__((always_inline))
    auto get_distance() const -> distance_t {
        return distance;
    }

    __attribute__((always_inline))
    auto set_distance(const distance_t dist) -> void {
        distance = dist;
    }

    // --- Lnbr-specific convenience aliases ---

    /** @brief Alias for get_base_id(). */
    __attribute__((always_inline))
    auto get_base_vid() const -> vertex_id_t { return get_base_id(); }

    /** @brief Alias for get_layer_id(). */
    __attribute__((always_inline))
    auto get_layer_vid() const -> vertex_id_t { return get_layer_id(); }

    /**
     * @brief Comparison operators: primary key = distance, tie-break = base_vid
     *        (smaller base_vid sorts first when distances are equal).
     */
    __attribute__((always_inline))
    constexpr bool operator<(const LnbrCandidateEntry& other) const noexcept {
        if (distance != other.distance) return distance < other.distance;
        return base_vid < other.base_vid;
    }

    __attribute__((always_inline))
    constexpr bool operator>(const LnbrCandidateEntry& other) const noexcept {
        if (distance != other.distance) return distance > other.distance;
        return base_vid > other.base_vid;
    }

    __attribute__((always_inline))
    constexpr bool operator<=(const LnbrCandidateEntry& other) const noexcept {
        return !(*this > other);
    }

    __attribute__((always_inline))
    constexpr bool operator>=(const LnbrCandidateEntry& other) const noexcept {
        return !(*this < other);
    }

    __attribute__((always_inline))
    constexpr bool operator==(const LnbrCandidateEntry& other) const noexcept {
        return distance == other.distance && base_vid == other.base_vid;
    }

    __attribute__((always_inline))
    constexpr bool operator!=(const LnbrCandidateEntry& other) const noexcept {
        return !(*this == other);
    }

};  // struct LnbrCandidateEntry

}   // namespace cpu
}   // namespace artea
