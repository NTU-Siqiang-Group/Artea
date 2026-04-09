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
 * @FilePath: /Artea/include/artea/cpu/router/data_structures/dnbr_candidate_entry.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Dnbr-flavored candidate entry (vertex_id + explored bit +
 *               distance, 8 bytes). Carries exactly the information a
 *               descent-graph router needs during beam search.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

#include <artea/cpu/router/data_structures/candidate_entry_concept.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Dnbr-flavored candidate entry with embedded explored/unexplored status.
 *
 * Design Philosophy:
 * This structure embeds the explored/unexplored state directly into the vertex ID
 * using the highest bit (bit 31), similar to the Neighbor structure. This eliminates
 * the need for a separate bitmap and avoids the bitmap shift overhead when inserting
 * elements into a sorted array.
 *
 * Bit Layout:
 * - Bit 31: Explored status (1 = explored, 0 = unexplored)
 * - Bits 0-30: Vertex ID (supports up to 2^31 - 1 vertices)
 *
 * @tparam RouterTraitsT Traits defining vertex types and distance types.
 */
template <typename RouterTraitsT>
struct alignas(8) DnbrCandidateEntry {   // 8 bytes

    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;
    static constexpr distance_t max_distance = RouterTraitsT::max_distance;
    static constexpr distance_t min_distance = RouterTraitsT::min_distance;

    // currently we assert that vertex_id_t is uint32_t (4 bytes)
    static_assert(sizeof(vertex_num_t) == 4, "vertex_num_t must be 4 bytes.");
    static_assert(std::is_unsigned<vertex_num_t>::value, "vertex_num_t must be unsigned (e.g., uint32_t).");
    static_assert(sizeof(vec_ele_t) == 4, "vec_ele_t must be 4 bytes to keep struct size 8 bytes.");

    // Bit masks configuration:
    // Bit 31: Explored/Unexplored status (1 = explored, 0 = unexplored)
    // Bit 0-30: Vertex ID
    static constexpr vertex_id_t MASK_STATUS_EXPLORED = 0x80000000; // 1000...
    static constexpr vertex_id_t MASK_ID              = 0x7FFFFFFF; // 0111...

    /** @brief Vertex ID and explored status packed into a single 32-bit type.
      * @note Bit 31: 1 = "explored", 0 = "unexplored".
      * @note Bits 0-30: Vertex ID. Available range: [0, 2^31 - 1].
      */
    vertex_id_t entry_id_and_status;     // 4 bytes

    /** @brief Distance to the candidate entry. */
    distance_t distance;      // 4 bytes

private:
    /** @brief Helper function to compute entry_id_and_status value with status flag. */
    static constexpr auto compute_id_and_status(vertex_id_t entry_id, bool is_explored) -> vertex_id_t {
        return is_explored ? ((entry_id & MASK_ID) | MASK_STATUS_EXPLORED) : (entry_id & MASK_ID);
    }

public:
    constexpr DnbrCandidateEntry() : entry_id_and_status(invalid_vertex_id), distance(0.0) {}

    constexpr DnbrCandidateEntry(const DnbrCandidateEntry&) = default;
    constexpr DnbrCandidateEntry& operator=(const DnbrCandidateEntry&) = default;
    DnbrCandidateEntry(DnbrCandidateEntry&&) = default;
    DnbrCandidateEntry& operator=(DnbrCandidateEntry&&) = default;
    ~DnbrCandidateEntry() = default;

    /** @brief Create a new DnbrCandidateEntry with given ID and distance (unexplored by default). */
    constexpr DnbrCandidateEntry(
        const vertex_id_t entry_id,
        const distance_t distance
    ) : entry_id_and_status(entry_id & MASK_ID), distance(distance) {}

    /** @brief Create a new DnbrCandidateEntry with given ID, distance, and explored status. */
    constexpr DnbrCandidateEntry(
        const vertex_id_t entry_id,
        const distance_t distance,
        const bool is_explored
    ) : entry_id_and_status(compute_id_and_status(entry_id, is_explored)), distance(distance) {}

    static constexpr auto make_invalid_entry() -> DnbrCandidateEntry {
        return DnbrCandidateEntry(invalid_vertex_id, max_distance);
    }

    /** @brief Create a sentinel entry with minimum distance (for max-heap sentinel). */
    static constexpr auto make_min_entry() -> DnbrCandidateEntry {
        return DnbrCandidateEntry(invalid_vertex_id, min_distance);
    }

    /** @brief Create a new DnbrCandidateEntry with given ID and distance (unexplored by default). */
    __attribute__((always_inline))
    static auto make_entry(
        const vertex_id_t entry_id,
        const distance_t distance
    ) -> DnbrCandidateEntry {
        return DnbrCandidateEntry{entry_id, distance};
    }

    /** @brief Get the entry ID masking out status bit (Bit 31). */
    __attribute__((always_inline))
    auto get_id() const -> vertex_id_t {
        return static_cast<vertex_id_t>(entry_id_and_status & MASK_ID);
    }

    /** @brief Check if this entry is an invalid/padding sentinel (stored ID == MASK_ID). */
    __attribute__((always_inline))
    auto is_invalid() const -> bool {
        return (entry_id_and_status & MASK_ID) == MASK_ID;
    }

    /** @brief Check if the entry is marked as "explored" (Bit 31 is set). */
    __attribute__((always_inline))
    auto is_explored() const -> bool {
        return (entry_id_and_status & MASK_STATUS_EXPLORED) != 0;
    }

    /** @brief Check if the entry is marked as "unexplored" (Bit 31 is unset). */
    __attribute__((always_inline))
    auto is_unexplored() const -> bool {
        return (entry_id_and_status & MASK_STATUS_EXPLORED) == 0;
    }

    /** @brief Mark the entry as "explored" by setting Bit 31. */
    __attribute__((always_inline))
    auto mark_as_explored() -> void {
        entry_id_and_status |= MASK_STATUS_EXPLORED;
    }

    /** @brief Mark the entry as "unexplored" by clearing Bit 31. */
    __attribute__((always_inline))
    auto mark_as_unexplored() -> void {
        entry_id_and_status &= (~MASK_STATUS_EXPLORED);
    }

    /** @brief Set the explored/unexplored status. */
    __attribute__((always_inline))
    auto set_explored_status(const bool is_explored) -> void {
        if (is_explored) mark_as_explored();
        else mark_as_unexplored();
    }

    /** @brief Get the distance to the candidate entry. */
    __attribute__((always_inline))
    auto get_distance() const -> distance_t {
        return distance;
    }

    /** @brief Set the distance to the candidate entry. */
    __attribute__((always_inline))
    auto set_distance(const distance_t dist) -> void {
        distance = dist;
    }

    /** @brief Comparison operators based on distance. */
    __attribute__((always_inline))
    constexpr bool operator<(const DnbrCandidateEntry& other) const noexcept {
        return distance < other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator>(const DnbrCandidateEntry& other) const noexcept {
        return distance > other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator<=(const DnbrCandidateEntry& other) const noexcept {
        return distance <= other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator>=(const DnbrCandidateEntry& other) const noexcept {
        return distance >= other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator==(const DnbrCandidateEntry& other) const noexcept {
        return distance == other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator!=(const DnbrCandidateEntry& other) const noexcept {
        return distance != other.distance;
    }

};  // struct DnbrCandidateEntry

/** @brief Comparator for DnbrCandidateEntry by distance. */
template <typename RouterTraitsT>
struct DnbrCandidateEntryComparator {
    using entry_t = DnbrCandidateEntry<RouterTraitsT>;

    __attribute__((always_inline))
    constexpr bool operator()(const entry_t& a, const entry_t& b) const noexcept {
        return a.get_distance() < b.get_distance();
    }
};  // struct DnbrCandidateEntryComparator

}   // namespace cpu
}   // namespace artea
