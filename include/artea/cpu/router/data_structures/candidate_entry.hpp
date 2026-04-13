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
 * @Description: Unified candidate entry (vid + explored bit + distance,
 *               8 bytes). Serves as both the beam-search candidate and the
 *               KNN result entry across all router families.
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
 * Bit Layout of the packed identity/status word:
 * - Bit 31: Explored status (1 = explored, 0 = unexplored)
 * - Bits 0-30: vertex id (supports up to 2^31 - 1 vertices)
 *
 * @tparam RouterTraitsT Traits defining vertex types and distance types.
 */
template <typename RouterTraitsT>
struct alignas(8) CandidateEntry {   // 8 bytes

    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t  = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t    = typename RouterTraitsT::vec_ele_t;
    using distance_t   = typename RouterTraitsT::distance_t;

    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;
    static constexpr distance_t  max_distance      = RouterTraitsT::max_distance;
    static constexpr distance_t  min_distance      = RouterTraitsT::min_distance;

    static_assert(sizeof(vertex_num_t) == 4, "vertex_num_t must be 4 bytes.");
    static_assert(std::is_unsigned<vertex_num_t>::value, "vertex_num_t must be unsigned.");
    static_assert(sizeof(vec_ele_t) == 4, "vec_ele_t must be 4 bytes.");

    static constexpr vertex_id_t MASK_STATUS_EXPLORED = 0x80000000;
    static constexpr vertex_id_t MASK_ID              = 0x7FFFFFFF;

private:
    /** @brief Packed vertex id (bits 0-30) + explored status (bit 31). */
    vertex_id_t _vid_and_status;   // 4 bytes

    /** @brief Distance to the query vector. */
    distance_t  _distance;         // 4 bytes

    static constexpr auto _compute_vid_and_status(vertex_id_t vid, bool is_explored) -> vertex_id_t {
        return is_explored ? ((vid & MASK_ID) | MASK_STATUS_EXPLORED) : (vid & MASK_ID);
    }

public:
    constexpr CandidateEntry()
        : _vid_and_status(invalid_vertex_id),
          _distance(0.0) {}

    constexpr CandidateEntry(
        const vertex_id_t vid,
        const distance_t  dist
    ) : _vid_and_status(vid & MASK_ID),
        _distance(dist) {}

    constexpr CandidateEntry(
        const vertex_id_t vid,
        const distance_t  dist,
        const bool        is_explored
    ) : _vid_and_status(_compute_vid_and_status(vid, is_explored)),
        _distance(dist) {}

    constexpr CandidateEntry(const CandidateEntry&)            = default;
    constexpr CandidateEntry& operator=(const CandidateEntry&) = default;
    CandidateEntry(CandidateEntry&&)                           = default;
    CandidateEntry& operator=(CandidateEntry&&)                = default;
    ~CandidateEntry()                                          = default;

    // ---- Sentinel factories ----

    static constexpr auto make_invalid_entry() -> CandidateEntry {
        return CandidateEntry(invalid_vertex_id, max_distance);
    }

    static constexpr auto make_min_entry() -> CandidateEntry {
        return CandidateEntry(invalid_vertex_id, min_distance);
    }

    __attribute__((always_inline))
    static auto make_entry(
        const vertex_id_t vid,
        const distance_t  dist
    ) -> CandidateEntry {
        return CandidateEntry{vid, dist};
    }

    // ---- Identity accessor ----

    __attribute__((always_inline))
    auto get_vid() const -> vertex_id_t {
        return static_cast<vertex_id_t>(_vid_and_status & MASK_ID);
    }

    // ---- Validity ----

    __attribute__((always_inline))
    auto is_invalid() const -> bool {
        return (_vid_and_status & MASK_ID) == MASK_ID;
    }

    // ---- Explored / unexplored state ----

    __attribute__((always_inline))
    auto is_explored() const -> bool {
        return (_vid_and_status & MASK_STATUS_EXPLORED) != 0;
    }

    __attribute__((always_inline))
    auto is_unexplored() const -> bool {
        return (_vid_and_status & MASK_STATUS_EXPLORED) == 0;
    }

    __attribute__((always_inline))
    auto mark_as_explored() -> void {
        _vid_and_status |= MASK_STATUS_EXPLORED;
    }

    __attribute__((always_inline))
    auto mark_as_unexplored() -> void {
        _vid_and_status &= (~MASK_STATUS_EXPLORED);
    }

    __attribute__((always_inline))
    auto set_explored_status(const bool is_explored) -> void {
        if (is_explored) mark_as_explored();
        else mark_as_unexplored();
    }

    // ---- Distance accessors ----

    __attribute__((always_inline))
    auto get_distance() const -> distance_t {
        return _distance;
    }

    __attribute__((always_inline))
    auto set_distance(const distance_t dist) -> void {
        _distance = dist;
    }

    // ---- Comparison operators (distance primary, vid tie-break) ----

    __attribute__((always_inline))
    constexpr bool operator<(const CandidateEntry& other) const noexcept {
        if (_distance != other._distance) return _distance < other._distance;
        return get_vid() < other.get_vid();
    }

    __attribute__((always_inline))
    constexpr bool operator>(const CandidateEntry& other) const noexcept {
        if (_distance != other._distance) return _distance > other._distance;
        return get_vid() > other.get_vid();
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
        return _distance == other._distance && get_vid() == other.get_vid();
    }

    __attribute__((always_inline))
    constexpr bool operator!=(const CandidateEntry& other) const noexcept {
        return !(*this == other);
    }

};  // struct CandidateEntry

}   // namespace cpu
}   // namespace artea
