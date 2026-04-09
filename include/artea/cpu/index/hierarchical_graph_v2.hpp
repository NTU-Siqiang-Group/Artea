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
#include <mutex>
#include <utility>
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
 * The underlying @c _layer_graphs vector is sized once at construction to
 * @c max_layers slots (all initially null). The visible layer count is an
 * atomic @c layer_num_t that tracks how many front slots have been
 * committed; it is bumped via @c grow_layers() or @c commit_layer() with
 * release ordering, and read via @c get_num_layers() with acquire ordering.
 * This enables concurrent readers to safely access recently added layers
 * without locks (reserving the backing vector at construction avoids
 * reallocation, so pointer-writes at slot @p i are visible to concurrent
 * readers once the layer count crosses @p i+1).
 *
 * No entry point is maintained — algorithms that need a starting vertex
 * should seed their beam search by sampling the top layer directly.
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

public:
    /**
     * @brief Construct a HierarchicalGraphV2 with capacity for @p max_layers
     *        total layers. The visible layer count starts at 0; layers are
     *        committed as they are populated via @c grow_layers() or
     *        @c commit_layer().
     *
     * The backing @c _layer_graphs vector is sized to @p max_layers (not
     * reserved) so that concurrent writes to slot @p i never reallocate.
     *
     * @param max_layers Hard upper bound on the number of layers this
     *                   container can ever hold.
     */
    explicit HierarchicalGraphV2(const layer_num_t max_layers) :
        _layer_graphs(max_layers),
        _num_layers(0) {}

    // Copy & move both deleted (contains std::mutex and std::atomic members).
    HierarchicalGraphV2(const HierarchicalGraphV2&) = delete;
    HierarchicalGraphV2& operator=(const HierarchicalGraphV2&) = delete;
    HierarchicalGraphV2(HierarchicalGraphV2&&) = delete;
    HierarchicalGraphV2& operator=(HierarchicalGraphV2&&) = delete;

    ~HierarchicalGraphV2() = default;

    // --- Layer Access ---

    /**
     * @brief Atomically read the number of visible (committed) layers.
     */
    __attribute__((always_inline))
    auto get_num_layers() const -> layer_num_t {
        return _num_layers.load(std::memory_order_acquire);
    }

    /**
     * @brief Maximum number of layers this container can ever hold
     *        (the reserved capacity of the backing vector).
     */
    __attribute__((always_inline))
    auto max_layers() const -> layer_num_t {
        return static_cast<layer_num_t>(_layer_graphs.size());
    }

    /**
     * @brief Access a layer's @c InternalGraph by id (mutable).
     * @param layer_id Index of the layer (0 = bottom). Must be less than
     *                 the value returned by @c get_num_layers() at the
     *                 moment of the call.
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
     * @brief Install a layer graph at the given layer id (transfers
     *        ownership). NOT thread-safe; for single-threaded / offline
     *        construction pipelines. New code should prefer
     *        @c grow_layers().
     *
     * This does NOT bump @c _num_layers — callers must pair it with a
     * subsequent @c commit_layer(layer_id) call to publish the new slot.
     */
    __attribute__((always_inline))
    auto set_layer_graph(const layer_id_t layer_id,
                         std::unique_ptr<internal_graph_t> graph) -> void {
        _layer_graphs[layer_id] = std::move(graph);
    }

    /**
     * @brief Publish a previously-set layer by advancing the visible layer
     *        count. Requires @p layer_id == get_num_layers() (i.e. commit
     *        the next slot in order).
     *
     * Not thread-safe; callers must serialize their own calls or use
     * @c grow_layers() instead.
     */
    __attribute__((always_inline))
    auto commit_layer(const layer_id_t layer_id) -> void {
        #ifndef NDEBUG
        const auto cur = _num_layers.load(std::memory_order_relaxed);
        if (layer_id != cur) {
            ARTEA_ERROR(fmt::format(
                "commit_layer: layer_id ({}) must equal current num_layers ({})",
                layer_id, cur));
        }
        #endif
        _num_layers.store(layer_id + 1, std::memory_order_release);
    }

    /**
     * @brief Ensure at least @p target_num_layers layers are committed,
     *        constructing any missing layers via @p layer_factory.
     *
     * Holds an internal mutex for the duration of the growth. Multiple
     * callers requesting growth concurrently are serialized; only the first
     * one to reach a given slot performs the allocation. Idempotent.
     *
     * @tparam LayerFactoryFnT Callable with signature
     *         @code
     *         std::unique_ptr<internal_graph_t>(layer_id_t new_layer_id)
     *         @endcode
     *         Invoked under the mutex, once per newly committed slot.
     * @param target_num_layers Desired minimum visible layer count.
     * @param layer_factory     Factory producing a fresh InternalGraph.
     * @return The new visible layer count after the call.
     */
    template <typename LayerFactoryFnT>
    auto grow_layers(const layer_num_t target_num_layers,
                     LayerFactoryFnT&& layer_factory) -> layer_num_t {
        // Fast path: already satisfied.
        if (_num_layers.load(std::memory_order_acquire) >= target_num_layers) {
            return _num_layers.load(std::memory_order_acquire);
        }

        std::lock_guard<std::mutex> guard(_extend_mutex);

        // Re-check under the lock.
        layer_num_t cur = _num_layers.load(std::memory_order_relaxed);
        if (cur >= target_num_layers) return cur;

        if (target_num_layers > max_layers()) {
            ARTEA_ERROR(fmt::format(
                "grow_layers: target ({}) exceeds max_layers ({})",
                target_num_layers, max_layers()));
        }

        // Commit new slots one by one. Publish _num_layers after each slot
        // so concurrent readers always see a contiguous prefix of non-null
        // layers.
        for (layer_num_t l = cur; l < target_num_layers; ++l) {
            _layer_graphs[l] = layer_factory(l);
            _num_layers.store(l + 1, std::memory_order_release);
        }
        return target_num_layers;
    }

    /**
     * @brief Atomically extend the hierarchy by constructing every missing
     *        layer up to @p target_num_layers and running @p seed_fn on the
     *        FINAL new layer before publishing it.
     *
     * The final new layer's visible slot (i.e. the bump of @c _num_layers)
     * is written AFTER @p seed_fn completes, so other threads observing the
     * new layer count via @c get_num_layers() are guaranteed to also see at
     * least one vertex populated in the top layer. This prevents other
     * threads from racing in on an empty top layer.
     *
     * Returns @c true if growth occurred (and @p seed_fn ran), or @c false
     * if another thread had already extended past @p target_num_layers
     * before we acquired the mutex — in that case @p seed_fn is NOT invoked.
     *
     * @tparam LayerFactoryFnT Same signature as @c grow_layers.
     * @tparam SeedFnT Callable with signature
     *         @code
     *         void(layer_id_t top_layer_id, internal_graph_t& top_layer)
     *         @endcode
     *         Invoked under the mutex exactly once, on the final new layer.
     * @param target_num_layers Desired visible layer count after growth.
     * @param layer_factory     Factory producing a fresh InternalGraph.
     * @param seed_fn           Callable that inserts the first vertex into
     *                          the new top layer.
     * @return @c true if this call grew the hierarchy (and seeded the new
     *         top); @c false if another thread had already extended.
     */
    template <typename LayerFactoryFnT, typename SeedFnT>
    auto extend_and_seed(const layer_num_t target_num_layers,
                         LayerFactoryFnT&& layer_factory,
                         SeedFnT&& seed_fn) -> bool {
        std::lock_guard<std::mutex> guard(_extend_mutex);

        const layer_num_t cur = _num_layers.load(std::memory_order_relaxed);
        if (cur >= target_num_layers) return false;  // lost the race

        if (target_num_layers > max_layers()) {
            ARTEA_ERROR(fmt::format(
                "extend_and_seed: target ({}) exceeds max_layers ({})",
                target_num_layers, max_layers()));
        }

        // Grow all but the last new layer normally (publish each slot as
        // soon as it's populated).
        for (layer_num_t l = cur; l + 1 < target_num_layers; ++l) {
            _layer_graphs[l] = layer_factory(l);
            _num_layers.store(l + 1, std::memory_order_release);
        }

        // Final new layer: construct it, run `seed_fn` to insert at least
        // one vertex, and THEN publish it. Other threads that only read
        // `get_num_layers()` never observe this layer as empty.
        const layer_num_t last = target_num_layers - 1;
        _layer_graphs[last] = layer_factory(last);
        seed_fn(static_cast<layer_id_t>(last), *_layer_graphs[last]);
        _num_layers.store(last + 1, std::memory_order_release);

        return true;
    }

    __attribute__((always_inline))
    auto get_layer_graphs() -> std::vector<std::unique_ptr<internal_graph_t>>& {
        return _layer_graphs;
    }

    __attribute__((always_inline))
    auto get_layer_graphs() const -> const std::vector<std::unique_ptr<internal_graph_t>>& {
        return _layer_graphs;
    }

private:
    /** @brief Per-layer InternalGraph instances (unique_ptr because InternalGraph is non-movable). */
    std::vector<std::unique_ptr<internal_graph_t>> _layer_graphs;

    /** @brief Atomic visible layer count (release/acquire pair with grow_layers). */
    std::atomic<layer_num_t> _num_layers;

    /** @brief Serializes concurrent grow_layers callers. */
    mutable std::mutex _extend_mutex;

};  // class HierarchicalGraphV2

}   // namespace cpu
}   // namespace artea
