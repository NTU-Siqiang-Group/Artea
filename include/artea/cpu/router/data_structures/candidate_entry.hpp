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
 * @FilePath: /Artea/include/artea/cpu/router/candidate_entry.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Candidate entry structure for routing operations.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

namespace artea {
namespace cpu {

/** @brief Candidate entry structure for storing vertex ID and distance during routing. */
template <typename RouterTraitsT>
struct alignas(8) CandidateEntry {   // 8 bytes

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

    /** @brief Vertex ID (full 32 bits available). */
    vertex_id_t entry_id;     // 4 bytes

    /** @brief Distance to the candidate entry. */
    distance_t distance;      // 4 bytes

    constexpr CandidateEntry() : entry_id(invalid_vertex_id), distance(0.0) {}

    constexpr CandidateEntry(const CandidateEntry&) = default;
    constexpr CandidateEntry& operator=(const CandidateEntry&) = default;
    CandidateEntry(CandidateEntry&&) = default;
    CandidateEntry& operator=(CandidateEntry&&) = default;
    ~CandidateEntry() = default;

    /** @brief Create a new CandidateEntry with given ID and distance. */
    constexpr CandidateEntry(
        const vertex_id_t entry_id,
        const distance_t distance
    ) : entry_id(entry_id), distance(distance) {}

    static constexpr auto make_invalid_entry() -> CandidateEntry {
        return CandidateEntry(invalid_vertex_id, max_distance);
    }

    /** @brief Create a sentinel entry with minimum distance (for max-heap sentinel). */
    static constexpr auto make_min_entry() -> CandidateEntry {
        return CandidateEntry(invalid_vertex_id, min_distance);
    }

    /** @brief Create a new CandidateEntry with given ID and distance. */
    __attribute__((always_inline))
    static auto make_entry(
        const vertex_id_t entry_id,
        const distance_t distance
    ) -> CandidateEntry {
        return CandidateEntry{entry_id, distance};
    }

    /** @brief Get the entry ID. */
    __attribute__((always_inline))
    auto get_id() const -> vertex_id_t {
        return entry_id;
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
    constexpr bool operator<(const CandidateEntry& other) const noexcept {
        return distance < other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator>(const CandidateEntry& other) const noexcept {
        return distance > other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator<=(const CandidateEntry& other) const noexcept {
        return distance <= other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator>=(const CandidateEntry& other) const noexcept {
        return distance >= other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator==(const CandidateEntry& other) const noexcept {
        return distance == other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator!=(const CandidateEntry& other) const noexcept {
        return distance != other.distance;
    }

};  // struct CandidateEntry

// TO enable optimizations for POD types
template <typename RouterTraitsT>
inline constexpr bool __candidate_entry_is_trivially_copyable =
    std::is_trivially_copyable<CandidateEntry<RouterTraitsT>>::value;

template <typename RouterTraitsT>
inline constexpr bool __candidate_entry_is_trivially_destructible =
    std::is_trivially_destructible<CandidateEntry<RouterTraitsT>>::value;

/** @brief Comparator for CandidateEntry by distance. */
template <typename RouterTraitsT>
struct CandidateEntryComparator {
    using entry_t = CandidateEntry<RouterTraitsT>;

    __attribute__((always_inline))
    constexpr bool operator()(const entry_t& a, const entry_t& b) const noexcept {
        return a.get_distance() < b.get_distance();
    }
};  // struct CandidateEntryComparator

}   // namespace cpu
}   // namespace artea
