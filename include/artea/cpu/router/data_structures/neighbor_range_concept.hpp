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
 * @FilePath: /Artea/include/artea/cpu/router/data_structures/neighbor_range_concept.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: NeighborRange concept — a thin adapter that lets one
 *               beam-search loop body iterate either compact (raw vid
 *               span) or dynamic (nbr_t span) graph storage uniformly,
 *               yielding vertex_id_t and stopping at the first invalid
 *               sentinel.
 */

#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

namespace artea {
namespace cpu {

/**
 * @brief Adapter contract used by the unified beam-search / greedy-search
 *        loop bodies. An adapter binds a graph (and optionally a layer id
 *        and a per-call neighbor cap) to expose a single method
 *        @c of(vid) that returns a range yielding @c vertex_id_t values
 *        and stops at the first sentinel.
 *
 * Implementations live in @c include/artea/cpu/router/detail/ — one each
 * for compact-layer, dynamic-layer, compact-flat, dynamic-flat. The router
 * loop body is identical regardless of which adapter is plugged in.
 *
 * Iterator dereference must yield a type convertible to the adapter's
 * @c vertex_id_t. Sentinel handling is the adapter's responsibility:
 * the iterator reports end-of-range as soon as the underlying storage
 * hits an invalid neighbor (whether that is @c invalid_vertex_id for
 * compact storage or @c nbr.is_invalid() for dynamic storage).
 *
 * @tparam AdapterT The adapter type to check.
 */
template <typename AdapterT>
concept NeighborRange = requires(const AdapterT& a, typename AdapterT::vertex_id_t v) {
    typename AdapterT::vertex_id_t;
    { a.of(v) } -> std::ranges::input_range;
};

}   // namespace cpu
}   // namespace artea
