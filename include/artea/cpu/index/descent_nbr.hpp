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
 * @FilePath: /Artea/include/artea/cpu/index/descent_nbr.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Descent Graph Neighbor structure definition.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

namespace artea {
namespace cpu {

/** @brief Descent Graph Neighbor structure for storing vertex ID and distance. */
template <typename BaseTraitsT>
struct alignas(8) DescentNeighbor {   // 8 bytes

    using vertex_num_t = typename BaseTraitsT::vertex_num_t;
    using vertex_id_t = typename BaseTraitsT::vertex_id_t;
    using vec_ele_t = typename BaseTraitsT::vec_ele_t;
    using distance_t = typename BaseTraitsT::distance_t;

    static constexpr vertex_id_t invalid_vertex_id = BaseTraitsT::invalid_vertex_id;
    static constexpr distance_t max_distance = BaseTraitsT::max_distance;

    // currently we assert that vertex_id_t is uint32_t (4 bytes)
    static_assert(sizeof(vertex_num_t) == 4, "vertex_num_t must be 4 bytes.");
    static_assert(std::is_unsigned<vertex_num_t>::value, "vertex_num_t must be unsigned (e.g., uint32_t).");
    static_assert(sizeof(vec_ele_t) == 4, "vec_ele_t must be 4 bytes to keep struct size 8 bytes.");

    // Bit masks configuration:
    // Bit 31: New/Old status
    // Bit 0-30: Vertex ID
    static constexpr vertex_id_t MASK_STATUS_NEW     = 0x80000000; // 1000...
    static constexpr vertex_id_t MASK_ID             = 0x7FFFFFFF; // 0111...

    /** @brief Neighbor ID and status packed into a single 32-bit type.
      * @note Bit 31: 1 = "new", 0 = "old".
      * @note Bits 0-30: Vertex ID. Available range: [0, 2^31 - 1].
      */
    vertex_id_t nbr_id_and_status;     // 4 bytes

    /** @brief Distance to the neighbor. */
    distance_t distance;               // 4 bytes

private:
    /** @brief Helper function to compute nbr_id_and_status value with status flag. */
    static constexpr auto compute_id_and_status(vertex_id_t nbr_id, bool is_new) -> vertex_id_t {
        return is_new ? ((nbr_id & MASK_ID) | MASK_STATUS_NEW) : (nbr_id & MASK_ID);
    }

public:
    constexpr DescentNeighbor() : nbr_id_and_status(invalid_vertex_id), distance(0.0) {}

    static constexpr auto make_invalid_nbr() -> DescentNeighbor {
        return DescentNeighbor{ invalid_vertex_id, max_distance };
    }

    DescentNeighbor(const DescentNeighbor&) = default;
    DescentNeighbor& operator=(const DescentNeighbor&) = default;
    DescentNeighbor(DescentNeighbor&&) = default;
    DescentNeighbor& operator=(DescentNeighbor&&) = default;
    ~DescentNeighbor() = default;

    /** @brief Create a Neighbor with given ID and distance (2-parameter constructor for constexpr). */
    constexpr DescentNeighbor(
        const vertex_id_t nbr_id,
        const distance_t distance
    ) : nbr_id_and_status(nbr_id & MASK_ID), distance(distance) {}

    /** @brief Create a Neighbor with given ID, distance, and flags. */
    constexpr DescentNeighbor(
        const vertex_id_t nbr_id,
        const distance_t distance,
        const bool is_new
    ) : nbr_id_and_status(compute_id_and_status(nbr_id, is_new)), distance(distance) {
        // Mask the ID to ensure it fits in 31 bits, then apply flags using helper function
    }

    /** @brief Create a new Neighbor with given ID and distance, marked as "new". */
    __attribute__((always_inline))
    static auto make_new_nbr(
        const vertex_id_t nbr_id,
        const distance_t distance
    ) -> DescentNeighbor {
        return DescentNeighbor{
            nbr_id,
            distance,
            true  // is_new
        };
    }

    /** @brief Create a old Neighbor with given ID and distance, marked as "new". */
    __attribute__((always_inline))
    static auto make_old_nbr(
        const vertex_id_t nbr_id,
        const distance_t distance
    ) -> DescentNeighbor {
        return DescentNeighbor{
            nbr_id,
            distance,
            false // is_new
        };
    }

    /** @brief Get the neighbor ID masking out status bit (Bit 31). */
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

    /** @brief Mark the neighbor as "new" by setting Bit 31. */
    __attribute__((always_inline))
    auto mark_as_new() -> void {
        nbr_id_and_status |= MASK_STATUS_NEW;
    }

    /** @brief Mark the neighbor as "old" by clearing Bit 31. */
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

    /** @brief Equality operator for testing. */
    __attribute__((always_inline))
    constexpr bool operator==(const DescentNeighbor& other) const noexcept {
        return get_id() == other.get_id() && distance == other.distance;
    }

    /** @brief Inequality operator for testing. */
    __attribute__((always_inline))
    constexpr bool operator!=(const DescentNeighbor& other) const noexcept {
        return !(*this == other);
    }

};  // struct DescentNeighbor

// TO enable optimizations for POD types
template <typename BaseTraitsT>
inline constexpr bool __descent_neighbor_is_trivially_copyable =
    std::is_trivially_copyable<DescentNeighbor<BaseTraitsT>>::value;

template <typename BaseTraitsT>
inline constexpr bool __descent_neighbor_is_trivially_destructible =
    std::is_trivially_destructible<DescentNeighbor<BaseTraitsT>>::value;

static_assert(std::is_trivially_copyable<DescentNeighbor<BaseTraits<uint32_t, float>>>::value,
            "Neighbor must be trivially copyable to enable vector memcpy optimizations!");
static_assert(std::is_trivially_destructible<DescentNeighbor<BaseTraits<uint32_t, float>>>::value,
            "Neighbor must be trivially destructible!");

/** @brief Comparator for Neighbor (Distance primary, ID secondary). */
template <typename BaseTraitsT>
struct DNbrComparator {
    using dnbr_t = DescentNeighbor<BaseTraitsT>;

    __attribute__((always_inline))
    constexpr bool operator()(const dnbr_t& a, const dnbr_t& b) const noexcept {
        return (a.get_distance() < b.get_distance()) ||
               (a.get_distance() == b.get_distance() && a.get_id() < b.get_id());
    }
};  // struct DNbrComparator

/** @brief Strict Comparator for Neighbor. */
template <typename BaseTraitsT>
struct StrictDNbrComparator {
    using dnbr_t = DescentNeighbor<BaseTraitsT>;

    __attribute__((always_inline))
    constexpr bool operator()(const dnbr_t& a, const dnbr_t& b) const noexcept {
        if (a.get_distance() != b.get_distance()) {
            return a.get_distance() < b.get_distance();
        }
        return a.get_id() < b.get_id();
    }
};  // struct StrictDNbrComparator

/** @brief Comparator for Neighbor by ID only. */
template <typename BaseTraitsT>
struct DNbrIdComparator {
    using dnbr_t = DescentNeighbor<BaseTraitsT>;

    __attribute__((always_inline))
    constexpr bool operator()(const dnbr_t& a, const dnbr_t& b) const noexcept {
        return a.get_id() < b.get_id();
    }
};  // struct DNbrIdComparator

/** @brief Comparator for Neighbor by Distance only. */
template <typename BaseTraitsT>
struct DNbrDistanceComparator {
    using dnbr_t = DescentNeighbor<BaseTraitsT>;

    __attribute__((always_inline))
    constexpr bool operator()(const dnbr_t& a, const dnbr_t& b) const noexcept {
        return a.get_distance() < b.get_distance();
    }
};  // struct DNbrDistanceComparator

}   // namespace cpu
}   // namespace artea