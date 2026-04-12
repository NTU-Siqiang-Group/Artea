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
 * @FilePath: /Artea/include/artea/cpu/router/data_structures/candidate_entry.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Unified candidate entry (level_vid + explored bit + base_vid +
 *               distance, 12 bytes). Serves as both the beam-search candidate
 *               and the KNN result entry across all router families.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

namespace artea {
namespace cpu {

/**
 * @brief Unified candidate / result entry with embedded explored/unexplored status.
 *
 * This structure stores both identifiers a proximity-graph beam search needs:
 *   - @c level_vid — the vertex's id inside the current layer's CSR
 *     (dedup key for the visited table and neighbor-block lookup).
 *   - @c base_vid — the vertex's id in the base dataset
 *     (for coordinate lookup and inter-layer descent).
 *
 * For bottom-graph (single-layer) scenarios, @c level_vid == @c base_vid;
 * the 2-arg constructor sets both fields to the same value.
 *
 * Bit Layout of @c level_vid_and_status:
 * - Bit 31: Explored status (1 = explored, 0 = unexplored)
 * - Bits 0-30: level_vid (supports up to 2^31 - 1 vertices per layer)
 *
 * @tparam RouterTraitsT Traits defining vertex types and distance types.
 */
template <typename RouterTraitsT>
struct alignas(8) CandidateEntry {   // 12 bytes useful, 16 aligned

    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;
    static constexpr distance_t max_distance = RouterTraitsT::max_distance;
    static constexpr distance_t min_distance = RouterTraitsT::min_distance;

    static_assert(sizeof(vertex_num_t) == 4, "vertex_num_t must be 4 bytes.");
    static_assert(std::is_unsigned<vertex_num_t>::value, "vertex_num_t must be unsigned.");
    static_assert(sizeof(vec_ele_t) == 4, "vec_ele_t must be 4 bytes.");

    // Bit masks:
    static constexpr vertex_id_t MASK_STATUS_EXPLORED = 0x80000000;
    static constexpr vertex_id_t MASK_ID              = 0x7FFFFFFF;

    /** @brief level_vid (bits 0-30) + explored status (bit 31). */
    vertex_id_t level_vid_and_status;   // 4 bytes

    /** @brief base dataset id for coordinate lookup. */
    vertex_id_t base_vid;               // 4 bytes

    /** @brief Distance to the query vector. */
    distance_t  distance;               // 4 bytes

private:
    static constexpr auto _compute_lv_and_status(vertex_id_t level_vid, bool is_explored) -> vertex_id_t {
        return is_explored ? ((level_vid & MASK_ID) | MASK_STATUS_EXPLORED) : (level_vid & MASK_ID);
    }

public:
    constexpr CandidateEntry()
        : level_vid_and_status(invalid_vertex_id),
          base_vid(invalid_vertex_id),
          distance(0.0) {}

    constexpr CandidateEntry(const CandidateEntry&) = default;
    constexpr CandidateEntry& operator=(const CandidateEntry&) = default;
    CandidateEntry(CandidateEntry&&) = default;
    CandidateEntry& operator=(CandidateEntry&&) = default;
    ~CandidateEntry() = default;

    // ---- 2-arg constructors (bottom-graph: level_vid == base_vid) ----

    /** @brief Create entry with vid used as both level_vid and base_vid. */
    constexpr CandidateEntry(
        const vertex_id_t vid,
        const distance_t distance_
    ) : level_vid_and_status(vid & MASK_ID),
        base_vid(vid & MASK_ID),
        distance(distance_) {}

    /** @brief Create entry with vid + explored status. */
    constexpr CandidateEntry(
        const vertex_id_t vid,
        const distance_t distance_,
        const bool is_explored
    ) : level_vid_and_status(_compute_lv_and_status(vid, is_explored)),
        base_vid(vid & MASK_ID),
        distance(distance_) {}

    // ---- 3-arg constructors (internal-graph: separate identifiers) ----

    /** @brief Create entry with separate base_vid and level_vid. */
    constexpr CandidateEntry(
        const vertex_id_t base_vid_,
        const vertex_id_t level_vid_,
        const distance_t  distance_
    ) : level_vid_and_status(level_vid_ & MASK_ID),
        base_vid(base_vid_),
        distance(distance_) {}

    /** @brief Create entry with separate base_vid, level_vid, and explored status. */
    constexpr CandidateEntry(
        const vertex_id_t base_vid_,
        const vertex_id_t level_vid_,
        const distance_t  distance_,
        const bool is_explored
    ) : level_vid_and_status(_compute_lv_and_status(level_vid_, is_explored)),
        base_vid(base_vid_),
        distance(distance_) {}

    // ---- Sentinel factories ----

    static constexpr auto make_invalid_entry() -> CandidateEntry {
        return CandidateEntry(invalid_vertex_id, invalid_vertex_id, max_distance);
    }

    static constexpr auto make_min_entry() -> CandidateEntry {
        return CandidateEntry(invalid_vertex_id, invalid_vertex_id, min_distance);
    }

    /** @brief Convenience factory for bottom-graph entries. */
    __attribute__((always_inline))
    static auto make_entry(
        const vertex_id_t vid,
        const distance_t distance_
    ) -> CandidateEntry {
        return CandidateEntry{vid, distance_};
    }

    // ---- Identity accessors ----

    /** @brief Get the base dataset ID. */
    __attribute__((always_inline))
    auto get_base_id() const -> vertex_id_t {
        return base_vid;
    }

    /** @brief Get the layer-local ID (level_vid, with explored bit masked out). */
    __attribute__((always_inline))
    auto get_layer_id() const -> vertex_id_t {
        return static_cast<vertex_id_t>(level_vid_and_status & MASK_ID);
    }

    /** @brief Alias for get_base_id(). */
    __attribute__((always_inline))
    auto get_base_vid() const -> vertex_id_t { return get_base_id(); }

    /** @brief Alias for get_layer_id(). */
    __attribute__((always_inline))
    auto get_layer_vid() const -> vertex_id_t { return get_layer_id(); }

    // ---- In-place mutation for inter-layer remapping ----

    /**
     * @brief Replace the level_vid with @p new_level_vid, preserving
     *        explored status. Used by inter-layer conversion to remap
     *        entries to the next layer's CSR identity.
     */
    __attribute__((always_inline))
    auto set_level_vid(const vertex_id_t new_level_vid) -> void {
        const bool explored = is_explored();
        level_vid_and_status = _compute_lv_and_status(new_level_vid, explored);
    }

    // ---- Validity ----

    __attribute__((always_inline))
    auto is_invalid() const -> bool {
        return (level_vid_and_status & MASK_ID) == MASK_ID;
    }

    // ---- Explored / unexplored state ----

    __attribute__((always_inline))
    auto is_explored() const -> bool {
        return (level_vid_and_status & MASK_STATUS_EXPLORED) != 0;
    }

    __attribute__((always_inline))
    auto is_unexplored() const -> bool {
        return (level_vid_and_status & MASK_STATUS_EXPLORED) == 0;
    }

    __attribute__((always_inline))
    auto mark_as_explored() -> void {
        level_vid_and_status |= MASK_STATUS_EXPLORED;
    }

    __attribute__((always_inline))
    auto mark_as_unexplored() -> void {
        level_vid_and_status &= (~MASK_STATUS_EXPLORED);
    }

    __attribute__((always_inline))
    auto set_explored_status(const bool is_explored) -> void {
        if (is_explored) mark_as_explored();
        else mark_as_unexplored();
    }

    // ---- Distance accessors ----

    __attribute__((always_inline))
    auto get_distance() const -> distance_t {
        return distance;
    }

    __attribute__((always_inline))
    auto set_distance(const distance_t dist) -> void {
        distance = dist;
    }

    // ---- Comparison operators (distance primary, base_vid tie-break) ----

    __attribute__((always_inline))
    constexpr bool operator<(const CandidateEntry& other) const noexcept {
        if (distance != other.distance) return distance < other.distance;
        return base_vid < other.base_vid;
    }

    __attribute__((always_inline))
    constexpr bool operator>(const CandidateEntry& other) const noexcept {
        if (distance != other.distance) return distance > other.distance;
        return base_vid > other.base_vid;
    }

    __attribute__((always_inline))
    constexpr bool operator<=(const CandidateEntry& other) const noexcept {
        return !(*this > other);
    }

    __attribute__((always_inline))
    constexpr bool operator>=(const CandidateEntry& other) const noexcept {
        return !(*this < other);
    }

    __attribute__((always_inline))
    constexpr bool operator==(const CandidateEntry& other) const noexcept {
        return distance == other.distance && base_vid == other.base_vid;
    }

    __attribute__((always_inline))
    constexpr bool operator!=(const CandidateEntry& other) const noexcept {
        return !(*this == other);
    }

};  // struct CandidateEntry

}   // namespace cpu
}   // namespace artea
