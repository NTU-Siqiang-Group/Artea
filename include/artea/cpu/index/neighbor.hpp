// Copyright 2025 Weitang Ye
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
 * @FilePath: /Artea/include/artea/cpu/index/neighbor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Neighbor structure definition.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

#include <artea/common/definitions.hpp>

namespace artea {
namespace cpu {

/** @brief Neighbor structure for storing vertex ID and distance. */
template <typename BaseTraitsT>
struct alignas(8) Neighbor {   // 8 bytes

    using vertex_num_t = typename BaseTraitsT::vertex_num_t;
    using vertex_id_t = typename BaseTraitsT::vertex_id_t;
    using vec_ele_t = typename BaseTraitsT::vec_ele_t;
    using distance_t = typename BaseTraitsT::distance_t;

    // currently we assert that vertex_id_t is uint32_t (4 bytes)
    static_assert(sizeof(vertex_num_t) == 4, "vertex_num_t must be 4 bytes.");
    static_assert(std::is_unsigned<vertex_num_t>::value, "vertex_num_t must be unsigned (e.g., uint32_t).");
    static_assert(sizeof(vec_ele_t) == 4, "vec_ele_t must be 4 bytes to keep struct size 8 bytes.");

    // Bit masks configuration:
    // Bit 31: New/Old status
    // Bit 30: Removed status
    // Bit 0-29: Vertex ID
    static constexpr vertex_id_t MASK_STATUS_NEW     = 0x80000000; // 1000...
    static constexpr vertex_id_t MASK_STATUS_REMOVED = 0x40000000; // 0100...
    static constexpr vertex_id_t MASK_ID             = 0x3FFFFFFF; // 0011...

    /** @brief Neighbor ID and status packed into a single 32-bit type.
      * @note Bit 31: 1 = "new", 0 = "old".
      * @note Bit 30: 1 = "removed", 0 = "active".
      * @note Bits 0-29: Vertex ID. Available range: [0, 2^30 - 1].
      */
    vertex_id_t nbr_id_and_status;     // 4 bytes

    /** @brief Distance to the neighbor. */
    distance_t distance;               // 4 bytes

    Neighbor() : nbr_id_and_status(invalid_vertex_id<vertex_id_t>()), distance(0.0) {}

    /** @brief Create a new Neighbor with given ID, distance, and flags. */
    Neighbor(
        const vertex_id_t nbr_id,
        const distance_t distance,
        const bool is_new,
        const bool is_removed = false
    ) : distance(distance) {
        // Mask the ID to ensure it fits in 30 bits, then apply flags
        nbr_id_and_status = nbr_id & MASK_ID;
        if (is_new) {
            nbr_id_and_status |= MASK_STATUS_NEW;
        }
        if (is_removed) {
            nbr_id_and_status |= MASK_STATUS_REMOVED;
        }
    }

    /** @brief Create a new Neighbor with given ID and distance, marked as "new" and "active". */
    __attribute__((always_inline))
    static auto create_new(
        const vertex_id_t nbr_id,
        const distance_t distance
    ) -> Neighbor {
        return Neighbor{
            nbr_id,
            distance,
            true,  // is_new
            false  // is_removed
        };
    }

    /** @brief Get the neighbor ID masking out status bits (Bits 30 and 31). */
    __attribute__((always_inline))
    auto get_id() const -> vertex_id_t {
        return static_cast<vertex_id_t>(nbr_id_and_status & MASK_ID);
    }

    /** @brief Check if the neighbor is marked as "new" (Bit 31 is set). */
    __attribute__((always_inline))
    auto is_new() const -> bool {
        return (nbr_id_and_status & MASK_STATUS_NEW) != 0;
    }

    /** @brief Check if the neighbor is marked as "old" (Bit 31 is unset). */
    __attribute__((always_inline))
    auto is_old() const -> bool {
        return (nbr_id_and_status & MASK_STATUS_NEW) == 0;
    }

    /** @brief Check if the neighbor is marked as "removed" (Bit 30 is set). */
    __attribute__((always_inline))
    auto is_removed() const -> bool {
        return (nbr_id_and_status & MASK_STATUS_REMOVED) != 0;
    }

    /** @brief Mark the neighbor as "removed" by setting Bit 30. */
    __attribute__((always_inline))
    auto mark_as_removed() -> void {
        nbr_id_and_status |= MASK_STATUS_REMOVED;
    }

    /** @brief Mark the neighbor as "new" by setting Bit 31. */
    __attribute__((always_inline))
    auto mark_as_new() -> void {
        nbr_id_and_status |= MASK_STATUS_NEW;
    }

    /** @brief Mark the neighbor as "old" by clearing Bit 31. Preserves removed status. */
    __attribute__((always_inline))
    auto mark_as_old() -> void {
        nbr_id_and_status &= (~MASK_STATUS_NEW);
    }

    /** @brief Set the new/old status. */
    __attribute__((always_inline))
    auto set_status(const bool is_new) -> void {
        if (is_new) mark_as_new();
        else mark_as_old();
    }

    /** @brief Get the distance to the neighbor. */
    __attribute__((always_inline))
    auto get_distance() const -> distance_t {
        return distance;
    }

    /** @brief Set the distance to the neighbor. */
    __attribute__((always_inline))
    auto set_distance(const distance_t dist) -> void {
        distance = dist;
    }

};  // struct Neighbor

/** @brief Comparator for Neighbor, lambda function. */
template <typename vertex_num_t, typename vec_ele_t>
constexpr auto NeighborComparator = [](
    const Neighbor<vertex_num_t, vec_ele_t>& a,
    const Neighbor<vertex_num_t, vec_ele_t>& b
) __attribute__((always_inline)) -> bool {
    return (a.get_distance() < b.get_distance()) or
        (a.get_distance() == b.get_distance() && a.get_id() < b.get_id());
};  // constexpr auto NeighborComparator

/** @brief Strict comparator for Neighbor, lambda function. */
template <typename vertex_num_t, typename vec_ele_t>
constexpr auto StrictNeighborComparator = [](
    const Neighbor<vertex_num_t, vec_ele_t>& a,
    const Neighbor<vertex_num_t, vec_ele_t>& b
) __attribute__((always_inline)) -> bool {
    if (a.get_distance() != b.get_distance()) {
        return a.get_distance() < b.get_distance();
    }
    if (a.get_id() != b.get_id()) {
        return a.get_id() < b.get_id();
    }
    /** @note StrictNeighborComparator ensures that alive neighbors come before removed ones. */
    return a.is_removed() < b.is_removed();
};  //  constexpr auto StrictNeighborComparator

/** @brief Comparator for Neighbor by ID, lambda function. */
template <typename vertex_num_t, typename vec_ele_t>
constexpr auto NeighborIdComparator = [](
    const Neighbor<vertex_num_t, vec_ele_t>& a,
    const Neighbor<vertex_num_t, vec_ele_t>& b
) __attribute__((always_inline)) -> bool {
    return a.get_id() < b.get_id();
};  // constexpr auto NeighborIdComparator

/** @brief Comparator for Neighbor by Distance, lambda function. */
template <typename vertex_num_t, typename vec_ele_t>
constexpr auto NeighborDistanceComparator = [](
    const Neighbor<vertex_num_t, vec_ele_t>& a,
    const Neighbor<vertex_num_t, vec_ele_t>& b
) __attribute__((always_inline)) -> bool {
    return a.get_distance() < b.get_distance();
};  // constexpr auto NeighborDistanceComparator

}   // namespace cpu
}   // namespace artea