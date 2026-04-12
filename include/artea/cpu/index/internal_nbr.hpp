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
 * @FilePath: /Artea/include/artea/cpu/index/internal_nbr.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Internal (multi-layer) neighbor structure with dual
 *               identifiers, distance, and new/old status.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include <limits>

namespace artea {
namespace cpu {

/**
 * @brief Internal neighbor structure storing dual vertex identifiers,
 *        distance, and new/old status (12 bytes useful, 16 aligned).
 *
 * Each InternalNeighbor records:
 *   - @c level_vid : the neighbor's position in the layer-local CSR storage
 *                    (dedup key for the visited table).
 *                    The high bit (31) carries the "new/old" status flag.
 *   - @c base_vid  : the neighbor's position in the global vector table
 *                    (used for coordinate lookup and inter-layer descent).
 *   - @c distance  : distance to the source vertex of the containing
 *                    neighbor list.
 *
 * Bit Layout of @c level_vid_and_status:
 * - Bit 31: New/Old status (1 = new, 0 = old)
 * - Bits 0-30: level_vid (supports up to 2^31 - 1 vertices per layer)
 */
template <typename BaseTraitsT>
struct alignas(8) InternalNeighbor {   // 12 bytes useful, 16 aligned

    using vertex_num_t = typename BaseTraitsT::vertex_num_t;
    using vertex_id_t  = typename BaseTraitsT::vertex_id_t;
    using vec_ele_t    = typename BaseTraitsT::vec_ele_t;
    using distance_t   = typename BaseTraitsT::distance_t;

    static constexpr vertex_id_t invalid_vertex_id = BaseTraitsT::invalid_vertex_id;
    static constexpr distance_t  max_distance      = BaseTraitsT::max_distance;

    static_assert(sizeof(vertex_num_t) == 4, "vertex_num_t must be 4 bytes.");
    static_assert(std::is_unsigned<vertex_num_t>::value, "vertex_num_t must be unsigned.");
    static_assert(sizeof(vec_ele_t) == 4, "vec_ele_t must be 4 bytes.");

    // Bit masks:
    static constexpr vertex_id_t MASK_STATUS_NEW = 0x80000000;
    static constexpr vertex_id_t MASK_ID         = 0x7FFFFFFF;

    /** @brief level_vid (bits 0-30) + new/old status (bit 31). */
    vertex_id_t level_vid_and_status;   // 4 bytes

    /** @brief base dataset id for coordinate lookup. */
    vertex_id_t base_vid;               // 4 bytes

    /** @brief Distance to the source vertex of the containing list. */
    distance_t  distance;               // 4 bytes

private:
    static constexpr auto _compute_lv_and_status(vertex_id_t level_vid, bool is_new) -> vertex_id_t {
        return is_new ? ((level_vid & MASK_ID) | MASK_STATUS_NEW) : (level_vid & MASK_ID);
    }

public:
    constexpr InternalNeighbor()
        : level_vid_and_status(invalid_vertex_id),
          base_vid(invalid_vertex_id),
          distance(0.0) {}

    /** @brief Create an InternalNeighbor with given ids (distance = 0, old). */
    constexpr InternalNeighbor(
        const vertex_id_t base_vid_,
        const vertex_id_t level_vid_
    ) : level_vid_and_status(level_vid_ & MASK_ID),
        base_vid(base_vid_),
        distance(0.0) {}

    /** @brief Create an InternalNeighbor with ids and distance (old by default). */
    constexpr InternalNeighbor(
        const vertex_id_t base_vid_,
        const vertex_id_t level_vid_,
        const distance_t  distance_
    ) : level_vid_and_status(level_vid_ & MASK_ID),
        base_vid(base_vid_),
        distance(distance_) {}

    /** @brief Create an InternalNeighbor with ids, distance, and status. */
    constexpr InternalNeighbor(
        const vertex_id_t base_vid_,
        const vertex_id_t level_vid_,
        const distance_t  distance_,
        const bool is_new
    ) : level_vid_and_status(_compute_lv_and_status(level_vid_, is_new)),
        base_vid(base_vid_),
        distance(distance_) {}

    InternalNeighbor(const InternalNeighbor&) = default;
    InternalNeighbor& operator=(const InternalNeighbor&) = default;
    InternalNeighbor(InternalNeighbor&&) = default;
    InternalNeighbor& operator=(InternalNeighbor&&) = default;
    ~InternalNeighbor() = default;

    // ---- Factories ----

    static constexpr auto make_invalid_nbr() -> InternalNeighbor {
        return InternalNeighbor(invalid_vertex_id, invalid_vertex_id, max_distance);
    }

    __attribute__((always_inline))
    static auto make_new_nbr(
        const vertex_id_t base_vid_,
        const vertex_id_t level_vid_,
        const distance_t  distance_
    ) -> InternalNeighbor {
        return InternalNeighbor{base_vid_, level_vid_, distance_, /*is_new=*/true};
    }

    __attribute__((always_inline))
    static auto make_old_nbr(
        const vertex_id_t base_vid_,
        const vertex_id_t level_vid_,
        const distance_t  distance_
    ) -> InternalNeighbor {
        return InternalNeighbor{base_vid_, level_vid_, distance_, /*is_new=*/false};
    }

    // ---- Identity accessors ----

    /** @brief Get the level-local vertex ID (masking out status bit). */
    __attribute__((always_inline))
    auto get_level_vid() const -> vertex_id_t {
        return static_cast<vertex_id_t>(level_vid_and_status & MASK_ID);
    }

    /** @brief Get the base dataset ID. */
    __attribute__((always_inline))
    auto get_base_vid() const -> vertex_id_t {
        return base_vid;
    }

    /**
     * @brief Set the level_vid, preserving status bit. Used during
     *        inter-layer remapping.
     */
    __attribute__((always_inline))
    auto set_level_vid(const vertex_id_t new_level_vid) -> void {
        const bool new_status = is_new();
        level_vid_and_status = _compute_lv_and_status(new_level_vid, new_status);
    }

    // ---- New/Old status ----

    __attribute__((always_inline))
    auto is_new() const -> bool {
        return (level_vid_and_status & MASK_STATUS_NEW) != 0;
    }

    __attribute__((always_inline))
    auto is_old() const -> bool {
        return (level_vid_and_status & MASK_STATUS_NEW) == 0;
    }

    __attribute__((always_inline))
    auto mark_as_new() -> void {
        level_vid_and_status |= MASK_STATUS_NEW;
    }

    __attribute__((always_inline))
    auto mark_as_old() -> void {
        level_vid_and_status &= (~MASK_STATUS_NEW);
    }

    __attribute__((always_inline))
    auto set_status(const bool is_new) -> void {
        if (is_new) mark_as_new();
        else mark_as_old();
    }

    // ---- Validity ----

    __attribute__((always_inline))
    auto is_invalid() const -> bool {
        return (level_vid_and_status & MASK_ID) == MASK_ID;
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

    // ---- Equality (ignores distance and status — identity-only) ----

    __attribute__((always_inline))
    constexpr bool operator==(const InternalNeighbor& other) const noexcept {
        return get_level_vid() == other.get_level_vid() &&
               base_vid == other.base_vid;
    }

    __attribute__((always_inline))
    constexpr bool operator!=(const InternalNeighbor& other) const noexcept {
        return !(*this == other);
    }

};  // struct InternalNeighbor

}   // namespace cpu
}   // namespace artea
