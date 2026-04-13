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
 * @FilePath: /Artea/include/artea/cpu/index/neighbor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Unified neighbor record: vertex id + distance + new/old
 *               status, packed to 8 bytes. Used by the new hierarchical-graph
 *               storage — every layer's neighbor array is a contiguous
 *               sub-span of Neighbor entries.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace artea {
namespace cpu {

/**
 * @brief Neighbor record holding a vertex id (with a new/old status bit
 *        packed into the high bit) and a distance. 8 bytes, 8-byte aligned.
 *
 * Bit layout of @c _nbr_vid_and_status:
 *   - Bit 31    : 1 = "new", 0 = "old"
 *   - Bits 0-30 : vertex id (max representable id = 2^31 - 1)
 *
 * @tparam BaseTraitsT The base traits type (supplies id/distance types).
 */
template <typename BaseTraitsT>
struct alignas(8) Neighbor {   // 8 bytes

    using vertex_num_t = typename BaseTraitsT::vertex_num_t;
    using vertex_id_t  = typename BaseTraitsT::vertex_id_t;
    using vec_ele_t    = typename BaseTraitsT::vec_ele_t;
    using distance_t   = typename BaseTraitsT::distance_t;

    static constexpr vertex_id_t invalid_vertex_id = BaseTraitsT::invalid_vertex_id;
    static constexpr distance_t  max_distance      = BaseTraitsT::max_distance;

    static_assert(sizeof(vertex_num_t) == 4, "vertex_num_t must be 4 bytes.");
    static_assert(std::is_unsigned<vertex_num_t>::value,
                  "vertex_num_t must be unsigned (e.g., uint32_t).");
    static_assert(sizeof(vec_ele_t) == 4, "vec_ele_t must be 4 bytes to keep struct size 8 bytes.");

    // Bit masks for the packed id + status word.
    static constexpr vertex_id_t MASK_STATUS_NEW = 0x80000000;
    static constexpr vertex_id_t MASK_ID         = 0x7FFFFFFF;

    /** @brief Packed vertex id (bits 0-30) + new/old status (bit 31). */
    vertex_id_t nbr_vid_and_status;   // 4 bytes

    /** @brief Distance from the owning vertex to this neighbor. */
    distance_t  distance;             // 4 bytes

private:
    static constexpr auto _compute_vid_and_status(vertex_id_t nbr_vid, bool is_new) -> vertex_id_t {
        return is_new ? ((nbr_vid & MASK_ID) | MASK_STATUS_NEW) : (nbr_vid & MASK_ID);
    }

public:
    constexpr Neighbor()
        : nbr_vid_and_status(invalid_vertex_id),
          distance(0.0) {}

    /** @brief 2-arg constructor: vertex id + distance (status = old). */
    constexpr Neighbor(
        const vertex_id_t nbr_vid,
        const distance_t  dist
    ) : nbr_vid_and_status(nbr_vid & MASK_ID),
        distance(dist) {}

    /** @brief 3-arg constructor: vertex id + distance + new/old status. */
    constexpr Neighbor(
        const vertex_id_t nbr_vid,
        const distance_t  dist,
        const bool        is_new
    ) : nbr_vid_and_status(_compute_vid_and_status(nbr_vid, is_new)),
        distance(dist) {}

    Neighbor(const Neighbor&)            = default;
    Neighbor& operator=(const Neighbor&) = default;
    Neighbor(Neighbor&&)                 = default;
    Neighbor& operator=(Neighbor&&)      = default;
    ~Neighbor()                          = default;

    // ---- Factories ----

    static constexpr auto make_invalid_nbr() -> Neighbor {
        return Neighbor{invalid_vertex_id, max_distance};
    }

    __attribute__((always_inline))
    static auto make_new_nbr(
        const vertex_id_t nbr_vid,
        const distance_t  dist
    ) -> Neighbor {
        return Neighbor{nbr_vid, dist, /*is_new=*/true};
    }

    __attribute__((always_inline))
    static auto make_old_nbr(
        const vertex_id_t nbr_vid,
        const distance_t  dist
    ) -> Neighbor {
        return Neighbor{nbr_vid, dist, /*is_new=*/false};
    }

    // ---- Identity accessors ----

    /** @brief Get the vertex id (status bit masked out). */
    __attribute__((always_inline))
    auto get_vid() const -> vertex_id_t {
        return static_cast<vertex_id_t>(nbr_vid_and_status & MASK_ID);
    }

    // ---- New/Old status ----

    __attribute__((always_inline))
    auto is_new() const -> bool {
        return (nbr_vid_and_status & MASK_STATUS_NEW) != 0;
    }

    __attribute__((always_inline))
    auto is_old() const -> bool {
        return (nbr_vid_and_status & MASK_STATUS_NEW) == 0;
    }

    __attribute__((always_inline))
    auto mark_as_new() -> void {
        nbr_vid_and_status |= MASK_STATUS_NEW;
    }

    __attribute__((always_inline))
    auto mark_as_old() -> void {
        nbr_vid_and_status &= (~MASK_STATUS_NEW);
    }

    __attribute__((always_inline))
    auto set_status(const bool is_new) -> void {
        if (is_new) mark_as_new();
        else mark_as_old();
    }

    // ---- Validity ----

    __attribute__((always_inline))
    auto is_invalid() const -> bool {
        return (nbr_vid_and_status & MASK_ID) == MASK_ID;
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

    // ---- Equality (id + distance; ignores status) ----

    __attribute__((always_inline))
    constexpr bool operator==(const Neighbor& other) const noexcept {
        return get_vid() == other.get_vid() && distance == other.distance;
    }

    __attribute__((always_inline))
    constexpr bool operator!=(const Neighbor& other) const noexcept {
        return !(*this == other);
    }

};  // struct Neighbor

template <typename BaseTraitsT>
inline constexpr bool __neighbor_is_trivially_copyable =
    std::is_trivially_copyable<Neighbor<BaseTraitsT>>::value;

template <typename BaseTraitsT>
inline constexpr bool __neighbor_is_trivially_destructible =
    std::is_trivially_destructible<Neighbor<BaseTraitsT>>::value;

namespace detail {
/** @brief Minimal traits stub used only for Neighbor triviality checks
 *         below. Avoids a circular include of @c base_traits.hpp. */
struct __neighbor_triviality_check_traits {
    using vertex_num_t = uint32_t;
    using vertex_id_t  = uint32_t;
    using vec_ele_t    = float;
    using distance_t   = float;
    static constexpr vertex_id_t invalid_vertex_id =
        std::numeric_limits<vertex_id_t>::max();
    static constexpr distance_t max_distance =
        std::numeric_limits<distance_t>::max();
};
}  // namespace detail

static_assert(
    std::is_trivially_copyable<Neighbor<detail::__neighbor_triviality_check_traits>>::value,
    "Neighbor must be trivially copyable to enable vector memcpy optimizations!");
static_assert(
    std::is_trivially_destructible<Neighbor<detail::__neighbor_triviality_check_traits>>::value,
    "Neighbor must be trivially destructible!");

/** @brief Comparator (distance primary, vid secondary). */
template <typename BaseTraitsT>
struct NbrComparator {
    using nbr_t = Neighbor<BaseTraitsT>;

    __attribute__((always_inline))
    constexpr bool operator()(const nbr_t& a, const nbr_t& b) const noexcept {
        return (a.get_distance() < b.get_distance()) ||
               (a.get_distance() == b.get_distance() && a.get_vid() < b.get_vid());
    }
};

/** @brief Strict comparator (tie-break by vid). */
template <typename BaseTraitsT>
struct StrictNbrComparator {
    using nbr_t = Neighbor<BaseTraitsT>;

    __attribute__((always_inline))
    constexpr bool operator()(const nbr_t& a, const nbr_t& b) const noexcept {
        if (a.get_distance() != b.get_distance()) {
            return a.get_distance() < b.get_distance();
        }
        return a.get_vid() < b.get_vid();
    }
};

/** @brief Comparator by id only. */
template <typename BaseTraitsT>
struct NbrIdComparator {
    using nbr_t = Neighbor<BaseTraitsT>;

    __attribute__((always_inline))
    constexpr bool operator()(const nbr_t& a, const nbr_t& b) const noexcept {
        return a.get_vid() < b.get_vid();
    }
};

/** @brief Comparator by distance only. */
template <typename BaseTraitsT>
struct NbrDistanceComparator {
    using nbr_t = Neighbor<BaseTraitsT>;

    __attribute__((always_inline))
    constexpr bool operator()(const nbr_t& a, const nbr_t& b) const noexcept {
        return a.get_distance() < b.get_distance();
    }
};

}   // namespace cpu
}   // namespace artea
