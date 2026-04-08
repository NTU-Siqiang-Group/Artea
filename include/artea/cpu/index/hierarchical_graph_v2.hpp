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
 * @FilePath: /Artea/include/artea/cpu/index/hierarchical_graph_v2.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Hierarchical container of InternalGraph layers with atomic
 *               lnbr_t entry point for concurrent construction.
 */

#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Hierarchical graph that holds @c InternalGraph layers and tracks
 *        an atomic @c lnbr_t entry point.
 *
 * Each layer is stored as a @c std::unique_ptr<internal_graph_t>; this is
 * required because @c InternalGraph deletes both copy and move (its CSR
 * stores reinterpret-cast atomic headers).
 *
 * The entry point is a @c std::atomic<lnbr_t>, allowing concurrent
 * construction threads to update it via @c update_entry_point().
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class HierarchicalGraphV2 {

    using vertex_num_t     = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t      = typename IndexTraitsT::vertex_id_t;
    using layer_num_t      = typename IndexTraitsT::layer_num_t;
    using layer_id_t       = typename IndexTraitsT::layer_id_t;
    using lnbr_t           = typename IndexTraitsT::lnbr_t;
    using internal_graph_t = typename IndexTraitsT::internal_graph_t;

    static_assert(std::atomic<lnbr_t>::is_always_lock_free,
        "std::atomic<lnbr_t> must be lock-free for HierarchicalGraphV2.");

public:
    /**
     * @brief Construct a HierarchicalGraphV2 with @p num_layers empty layer slots.
     *        Layers are populated later via @c set_layer_graph().
     *        The entry point is initialized to @c invalid_lnbr.
     */
    explicit HierarchicalGraphV2(const layer_num_t num_layers) :
        _layer_graphs(num_layers),
        _entry_point(IndexTraitsT::invalid_lnbr) {}

    // Copy & move both deleted (std::atomic is non-movable).
    HierarchicalGraphV2(const HierarchicalGraphV2&) = delete;
    HierarchicalGraphV2& operator=(const HierarchicalGraphV2&) = delete;
    HierarchicalGraphV2(HierarchicalGraphV2&&) = delete;
    HierarchicalGraphV2& operator=(HierarchicalGraphV2&&) = delete;

    ~HierarchicalGraphV2() = default;

    // --- Layer Access ---

    __attribute__((always_inline))
    auto get_num_layers() const -> layer_num_t {
        return static_cast<layer_num_t>(_layer_graphs.size());
    }

    /**
     * @brief Access a layer's @c InternalGraph by id (mutable).
     * @param layer_id Index of the layer (0 = bottom).
     */
    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) -> internal_graph_t& {
        return *_layer_graphs[layer_id];
    }

    __attribute__((always_inline))
    auto get_layer_graph(const layer_id_t layer_id) const -> const internal_graph_t& {
        return *_layer_graphs[layer_id];
    }

    /**
     * @brief Install a layer graph at the given layer id (transfers ownership).
     */
    __attribute__((always_inline))
    auto set_layer_graph(const layer_id_t layer_id,
                         std::unique_ptr<internal_graph_t> graph) -> void {
        _layer_graphs[layer_id] = std::move(graph);
    }

    __attribute__((always_inline))
    auto get_layer_graphs() -> std::vector<std::unique_ptr<internal_graph_t>>& {
        return _layer_graphs;
    }

    __attribute__((always_inline))
    auto get_layer_graphs() const -> const std::vector<std::unique_ptr<internal_graph_t>>& {
        return _layer_graphs;
    }

    // --- Entry Point (atomic) ---

    /**
     * @brief Atomically read the current entry point (acquire ordering).
     */
    __attribute__((always_inline))
    auto get_entry_point() const -> lnbr_t {
        return _entry_point.load(std::memory_order_acquire);
    }

    /**
     * @brief Atomically update the entry point (release ordering). Thread-safe.
     */
    __attribute__((always_inline))
    auto update_entry_point(const lnbr_t new_entry_point) -> void {
        _entry_point.store(new_entry_point, std::memory_order_release);
    }

private:
    /** @brief Per-layer InternalGraph instances (unique_ptr because InternalGraph is non-movable). */
    std::vector<std::unique_ptr<internal_graph_t>> _layer_graphs;

    /** @brief Atomic entry point (lnbr_t). Initialized to invalid_lnbr. */
    std::atomic<lnbr_t> _entry_point;

};  // class HierarchicalGraphV2

}   // namespace cpu
}   // namespace artea
