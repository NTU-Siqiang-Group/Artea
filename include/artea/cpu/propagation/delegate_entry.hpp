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
 * @FilePath: /Artea/include/artea/cpu/propagation/delegate_entry.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

namespace artea {
namespace cpu {

enum class EndpointStatus : uint8_t {
    UNVALIDATED = 0,
    VALIDATED_VALID = 1,
    VALIDATED_INVALID = 2
};

template <typename vertex_num_t, typename vec_ele_t>
struct alignas(8) DelegateEntry {   // 16 bytes

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;

    /** @brief The source vertex id. */
    vertex_id_t src;

    /** @brief The destination vertex id. */
    vertex_id_t dest;

    /** @brief The distance between source and destination vertices. */
    distance_t distance;

    /** @brief Whether the source vertex has been validated by RNG strategy. */
    EndpointStatus src_status;

    /** @brief Whether the destination vertex has been validated by RNG strategy. */
    EndpointStatus dest_status;

    // padding for alignment
    uint8_t _padding[2];

    /* --- Constructor --- */

    DelegateEntry(
        const vertex_id_t src,
        const EndpointStatus src_status,
        const vertex_id_t dest,
        const EndpointStatus dest_status,
        const distance_t distance
    ) :
        src(src),
        dest(dest),
        distance(distance),
        src_status(src_status),
        dest_status(dest_status)
    {}

    DelegateEntry(
        const vertex_id_t src,
        const vertex_id_t dest,
        const distance_t distance
    ) :
        src(src),
        dest(dest),
        distance(distance),
        src_status(EndpointStatus::UNVALIDATED),
        dest_status(EndpointStatus::UNVALIDATED)
    {}

    /* --- Accessors --- */

    __attribute__((always_inline))
    auto get_src_id() const -> vertex_id_t {
        return src;
    }

    __attribute__((always_inline))
    auto get_dest_id() const -> vertex_id_t {
        return dest;
    }

    __attribute__((always_inline))
    auto get_src_status() const -> EndpointStatus {
        return src_status;
    }

    __attribute__((always_inline))
    auto set_src_status(const EndpointStatus status) -> void {
        src_status = status;
    }

    __attribute__((always_inline))
    auto get_dest_status() const -> EndpointStatus {
        return dest_status;
    }

    __attribute__((always_inline))
    auto set_dest_status(const EndpointStatus status) -> void {
        dest_status = status;
    }

    __attribute__((always_inline))
    auto get_distance() const -> distance_t {
        return distance;
    }

    __attribute__((always_inline))
    auto set_distance(const distance_t dist) -> void {
        distance = dist;
    }

};  // struct DelegateEntry

template <typename vertex_num_t, typename vec_ele_t>
struct alignas(4) CompactDelegateEntry {   // 12 bytes

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;

    // When using this struct, ensure vertex_id_t is 4 bytes
    static_assert(sizeof(vertex_id_t) == 4, "vertex_id_t must be 4 bytes");

    constexpr static vertex_id_t STATUS_MASK = 0xC0000000;  // upper 2 bits for status
    constexpr static vertex_id_t ID_MASK = 0x3FFFFFFF;      // lower 30 bits for id

    /** @brief The source vertex id. */
    vertex_id_t src_id_and_status;  // use upper 2 bits for status

    /** @brief The destination vertex id. */
    vertex_id_t dest_id_and_status;  // use upper 2 bits for status

    /** @brief The distance between source and destination vertices. */
    distance_t distance;

    /* --- Constructor --- */

    CompactDelegateEntry(
        const vertex_id_t src_id,
        const EndpointStatus src_status,
        const vertex_id_t dest_id,
        const EndpointStatus dest_status,
        const distance_t distance
    ) :
        src_id_and_status(
            (src_id & ID_MASK) |
            (static_cast<vertex_id_t>(src_status) << 30)
        ),
        dest_id_and_status(
            (dest_id & ID_MASK) |
            (static_cast<vertex_id_t>(dest_status) << 30)
        ),
        distance(distance)
    {}

    CompactDelegateEntry(
        const vertex_id_t src_id,
        const vertex_id_t dest_id,
        const distance_t distance
    ) :
        src_id_and_status((src_id & ID_MASK) | 0),
        dest_id_and_status((dest_id & ID_MASK) | 0),
        distance(distance)
    {}

    /* --- Accessors --- */

    /** @brief Get the status of the source vertex. */
    __attribute__((always_inline))
    auto src_id() const -> vertex_id_t {
        return src_id_and_status & ID_MASK;
    }

    /** @brief Get the status of the source vertex. */
    __attribute__((always_inline))
    auto src_status() const -> EndpointStatus {
        return static_cast<EndpointStatus>((src_id_and_status & STATUS_MASK) >> 30);
    }

    /** @brief Set the status of the source vertex. */
    __attribute__((always_inline))
    auto set_src_status(const EndpointStatus status) -> void {
        src_id_and_status =
            (src_id_and_status & ID_MASK) |
            (static_cast<vertex_id_t>(status) << 30);
    }

    /** @brief Get the status of the destination vertex. */
    __attribute__((always_inline))
    auto dest_id() const -> vertex_id_t {
        return dest_id_and_status & ID_MASK;
    }

    /** @brief Get the status of the destination vertex. */
    __attribute__((always_inline))
    auto dest_status() const -> EndpointStatus {
        return static_cast<EndpointStatus>((dest_id_and_status & STATUS_MASK) >> 30);
    }

    /** @brief Set the status of the destination vertex. */
    __attribute__((always_inline))
    auto set_dest_status(const EndpointStatus status) -> void {
        dest_id_and_status =
            (dest_id_and_status & ID_MASK) |
            (static_cast<vertex_id_t>(status) << 30);
    }

    /** @brief Get the distance between source and destination vertices. */
    __attribute__((always_inline))
    auto get_distance() const -> distance_t {
        return distance;
    }

    /** @brief Set the distance between source and destination vertices. */
    __attribute__((always_inline))
    auto set_distance(const distance_t dist) -> void {
        distance = dist;
    }

};  // struct CompactDelegateEntry

}   // namespace cpu
}   // namespace artea