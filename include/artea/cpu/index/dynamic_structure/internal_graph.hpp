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
 * @FilePath: /Artea/include/artea/cpu/index/internal_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Internal graph with concurrent vertex insertion.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <immintrin.h>
#include <tbb/concurrent_vector.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {
namespace dynamic {

/**
 * @brief Internal graph supporting concurrent vertex insertion.
 *
 * Per-vertex CSR layout (stride = 1 + max_nbr_size lnbr_t slots):
 * @code
 *   [ header | nbr_0 | nbr_1 | ... | nbr_{max_nbr_size-1} ]
 * @endcode
 * The header slot is reinterpreted as a @c std::atomic<uint64_t> whose
 * lower 63 bits hold the valid neighbor count (@c num_valid_nbrs) and
 * whose MSB (bit 63) serves as a per-vertex spinlock for @c add_nbr().
 *
 * Per-vertex metadata (parallel to the CSR, indexed by layer_vid) lives in
 * the concurrent vector @c _vertex_info, which stores one @c lnbr_t slot
 * per vertex carrying both:
 *   - @c .base_vid — the vertex's id in the base dataset (write-once at
 *     @c add_vertex time).
 *   - @c .layer_vid — the vertex's layer_vid in the layer directly below,
 *     initially a placeholder equal to @c base_vid and overwritten by
 *     @c set_inter_layer_link when the caller knows the lower-layer id.
 * Both fields are accessible in O(1) via @c get_base_vid / @c get_inter_layer_link.
 *
 * Vertices are added concurrently via @c add_vertex(): each call atomically
 * claims a layer_vid by pushing to @c _vertex_info, then deterministically
 * locates its pre-allocated neighbor block in @c _csr_nbrs
 * (position = layer_vid * stride). Since each thread writes to a disjoint
 * region, no synchronization on @c _csr_nbrs is needed.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class InternalGraph {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using lnbr_t = typename IndexTraitsT::lnbr_t;

    using csr_lnbrs_t = cache_aligned_container_t<lnbr_t>;

    static constexpr vertex_id_t invalid_vertex_id = IndexTraitsT::invalid_vertex_id;
    static constexpr vertex_num_t default_max_nbr_size = 63;

    /** @brief Bit 63 of the atomic header: 1 = locked for add_nbr, 0 = unlocked. */
    static constexpr uint64_t LOCK_BIT   = uint64_t(1) << 63;
    static constexpr uint64_t COUNT_MASK = ~LOCK_BIT;

    __attribute__((always_inline))
    static auto _is_locked(const uint64_t header) -> bool {
        return (header & LOCK_BIT) != 0;
    }

    __attribute__((always_inline))
    static auto _get_count(const uint64_t header) -> uint64_t {
        return header & COUNT_MASK;
    }

    __attribute__((always_inline))
    static auto _make_header(const uint64_t count, const bool locked) -> uint64_t {
        return count | (locked ? LOCK_BIT : 0);
    }

    static_assert(sizeof(lnbr_t) == sizeof(uint64_t),
        "lnbr_t must be 8 bytes to alias with atomic<uint64_t> header.");

    /** @brief Per-vertex stride: 1 header slot + max_nbr_size neighbor slots. */
    __attribute__((always_inline))
    auto _stride() const -> size_t {
        return static_cast<size_t>(_max_nbr_size) + 1;
    }

    /** @brief Get atomic reference to the header of a vertex's block. */
    __attribute__((always_inline))
    auto _header_ref(const vertex_id_t src) -> std::atomic<uint64_t>& {
        return *reinterpret_cast<std::atomic<uint64_t>*>(
            &_csr_nbrs[static_cast<size_t>(src) * _stride()]
        );
    }

    __attribute__((always_inline))
    auto _header_ref(const vertex_id_t src) const -> const std::atomic<uint64_t>& {
        return *reinterpret_cast<const std::atomic<uint64_t>*>(
            &_csr_nbrs[static_cast<size_t>(src) * _stride()]
        );
    }

public:
    /**
     * @brief Construct a InternalGraph with pre-allocated CSR storage.
     * @param max_num_vertices Maximum number of vertices (determines CSR capacity).
     * @param max_nbr_size Fixed number of neighbor slots per vertex (default: 63).
     */
    InternalGraph(
        const vertex_num_t max_num_vertices,
        const vertex_num_t max_nbr_size = default_max_nbr_size
    ) :
        _max_num_vertices(max_num_vertices),
        _max_nbr_size(max_nbr_size)
    {
        // Pre-allocate the full CSR array; each vertex takes (1 + max_nbr_size) lnbr_t slots.
        _csr_nbrs.resize(static_cast<size_t>(_max_num_vertices) * _stride());
    }

    // Copying is deleted
    InternalGraph(const InternalGraph&) = delete;
    InternalGraph& operator=(const InternalGraph&) = delete;

    // Move is deleted (contains std::atomic via reinterpret)
    InternalGraph(InternalGraph&&) = delete;
    InternalGraph& operator=(InternalGraph&&) = delete;

    ~InternalGraph() = default;

    // --- Vertex Insertion ---

    /**
     * @brief Concurrently add a new vertex to this layer.
     *
     * 1. Atomically claims a layer_vid by pushing @c lnbr_t(base_vid, base_vid)
     *    into the concurrent @c _vertex_info vector. The second field of
     *    the slot is a placeholder for the lower-layer link; callers that
     *    know the actual lower_layer_vid should overwrite it via
     *    @c set_inter_layer_link afterwards. The @c base_vid field is
     *    write-once and can be read race-free via @c get_base_vid.
     * 2. Uses the claimed layer_vid to deterministically locate the
     *    pre-allocated block in @c _csr_nbrs (offset = layer_vid * stride).
     * 3. Initializes the header (num_valid_nbrs = 0) and fills all
     *    neighbor slots with @c IndexTraitsT::invalid_lnbr.
     *
     * @param base_vid This vertex's id in the base dataset. Stored in
     *                 @c _vertex_info[layer_vid].base_vid (write-once);
     *                 also used as the initial placeholder for the
     *                 lower-layer link until @c set_inter_layer_link
     *                 updates it.
     * @return The newly assigned layer_vid in this layer.
     */
    auto add_vertex(const vertex_id_t base_vid) -> vertex_id_t {
        auto it = _vertex_info.push_back(lnbr_t(base_vid, base_vid));
        const vertex_id_t layer_vid = static_cast<vertex_id_t>(
            it - _vertex_info.begin()
        );

        // Ensure the CSR array has room for this layer_vid. If the
        // pre-allocation is exhausted, grow by 2x under a mutex.
        if (layer_vid >= _max_num_vertices) {
            _ensure_capacity(layer_vid + 1);
        }

        const auto base = static_cast<size_t>(layer_vid) * _stride();
        // Initialize header: num_valid_nbrs = 0
        new (&_csr_nbrs[base]) std::atomic<uint64_t>(0);
        // Initialize neighbor slots to invalid
        std::fill(
            &_csr_nbrs[base + 1],
            &_csr_nbrs[base + 1] + _max_nbr_size,
            IndexTraitsT::invalid_lnbr
        );

        return layer_vid;
    }

    /**
     * @brief Concurrently add a new vertex with lock-free neighbor prefilling.
     *
     * Same as @c add_vertex(lower_layer_vid), but additionally invokes
     * @p nbr_prefilling_fn to populate the neighbor array before returning.
     *
     * Since the newly added vertex has no incoming edges yet (no other vertex
     * links to it), its neighbor sub-array is exclusively owned by the calling
     * thread at this point. The prefilling functor can therefore write to the
     * neighbor slots **without acquiring the spinlock**, which is safe as long
     * as it completes before any "add reverse edge" step makes this vertex
     * reachable by other threads' @c add_nbr() calls.
     *
     * @warning The caller must ensure that no other thread can modify this
     *          vertex's neighbor list between @c add_vertex() and the completion
     *          of @p nbr_prefilling_fn. In practice this means the "add reverse
     *          edge" step (which exposes this vertex to other threads' @c add_nbr()
     *          calls) must happen **after** this function returns.
     *
     * @tparam NbrPrefillingFnT Callable with signature:
     *         @code uint64_t(lnbr_t* slots, vertex_num_t max_nbr_size) @endcode
     *         Writes initial neighbors into @p slots and returns the number
     *         of valid neighbors written.
     * @param base_vid          This vertex's id in the base dataset
     *                          (see single-argument overload).
     * @param nbr_prefilling_fn Functor to populate neighbors lock-free.
     * @return The newly assigned layer_vid in this layer.
     */
    template <typename NbrPrefillingFnT>
    auto add_vertex(const vertex_id_t base_vid, NbrPrefillingFnT&& nbr_prefilling_fn) -> vertex_id_t {
        const vertex_id_t layer_vid = add_vertex(base_vid);

        lnbr_t* nbr_slots = &_csr_nbrs[static_cast<size_t>(layer_vid) * _stride() + 1];
        const uint64_t prefilled_count = nbr_prefilling_fn(nbr_slots, _max_nbr_size);
        _header_ref(layer_vid).store(
            _make_header(prefilled_count, false), std::memory_order_release
        );

        return layer_vid;
    }

    // --- Neighbor Insertion ---

    /** @brief Result of an add_nbr operation. */
    enum class AddNbrEvent : uint8_t {
        APPENDED,   ///< Neighbor was appended to a non-full array.
        PRUNED      ///< Array was full; pruning was performed.
    };

    using add_nbr_result_t = AddNbrEvent;

    /**
     * @brief Thread-safe: execute a functor under exclusive per-vertex spinlock.
     *
     * Acquires the spinlock (encoded in the header MSB) for the vertex's
     * neighbor sub-array, invokes @p fn with the neighbor slots, current
     * valid count, and max_nbr_size, then releases the lock with the new
     * count returned by @p fn.
     *
     * @tparam FnT Callable with signature:
     *         @code uint64_t(lnbr_t* slots, uint64_t count, vertex_num_t max_nbr_size) @endcode
     *         Must return the new valid neighbor count (written slots must be in-place).
     * @param src The layer_vid of the source vertex.
     * @param fn  The functor to execute under lock.
     * @return The value returned by @p fn (new valid neighbor count).
     */
    template <typename FnT>
    auto with_locked_nbrs(const vertex_id_t src, FnT&& fn) -> uint64_t {
        auto& header = _header_ref(src);
        lnbr_t* nbr_slots = &_csr_nbrs[static_cast<size_t>(src) * _stride() + 1];

        // Acquire: CAS to set the lock bit
        uint64_t cur = header.load(std::memory_order_relaxed);
        int backoff = 0;
        while (true) {
            if (_is_locked(cur)) {
                for (int i = 0; i < (1 << std::min(backoff, 6)); ++i) {
                    _mm_pause();
                }
                ++backoff;
                cur = header.load(std::memory_order_relaxed);
                continue;
            }
            const uint64_t locked = _make_header(_get_count(cur), true);
            if (header.compare_exchange_weak(cur, locked,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                break;
            }
        }

        // With locks, operate neighbor array
        const uint64_t count = _get_count(cur);
        const uint64_t new_count = fn(nbr_slots, count, _max_nbr_size);

        // Release
        header.store(_make_header(new_count, false), std::memory_order_release);
        return new_count;
    }

    /**
     * @brief Thread-safe: add a neighbor to an existing vertex.
     *
     * Under the per-vertex spinlock, either appends or prunes:
     *   - If count < max_nbr_size: writes @p new_nbr at slot[count], count++.
     *   - If count == max_nbr_size: invokes @p prune_fn to select neighbors,
     *     invalidates trailing slots, updates count.
     *
     * @tparam OverflowFnT Callable with signature:
     *         @code uint64_t(lnbr_t* slots, lnbr_t new_nbr) @endcode
     *         The array has exactly max_nbr_size valid entries when called.
     *         Must return the new count after pruning (neighbors written in-place).
     * @param src       The layer_vid of the source vertex.
     * @param new_nbr   The neighbor to add.
     * @param on_nbrs_overflow  Overflow functor invoked when the array is full.
     * @return AddNbrEvent indicating whether the neighbor was appended or pruning occurred.
     */
    template <typename OverflowFnT>
    auto add_nbr(const vertex_id_t src, const lnbr_t new_nbr, OverflowFnT&& on_nbrs_overflow) -> AddNbrEvent {
        AddNbrEvent result;

        with_locked_nbrs(src, [&](lnbr_t* slots, uint64_t count, vertex_num_t max_nbr_size) -> uint64_t {
            if (count < max_nbr_size) {
                slots[count] = new_nbr;
                result = AddNbrEvent::APPENDED;
                return count + 1;
            }
            ARTEA_ASSERT(count, static_cast<uint64_t>(max_nbr_size));
            const uint64_t new_count = on_nbrs_overflow(slots, new_nbr);
            for (uint64_t i = new_count; i < max_nbr_size; ++i) {
                slots[i] = IndexTraitsT::invalid_lnbr;
            }
            result = AddNbrEvent::PRUNED;
            return new_count;
        });

        return result;
    }

    // --- Neighbor Access ---

    /**
     * @brief Get the number of valid neighbors for a vertex (atomic load).
     * @param src The layer_vid of the source vertex.
     * @note Masks off the lock bit, so this is safe to call even during concurrent add_nbr.
     */
    __attribute__((always_inline))
    auto num_valid_nbrs(const vertex_id_t src) const -> uint64_t {
        return _get_count(_header_ref(src).load(std::memory_order_relaxed));
    }

    /**
     * @brief Set the number of valid neighbors for a vertex (atomic store).
     * @param src The layer_vid of the source vertex.
     * @param count The new valid neighbor count.
     * @note Must only be called when the vertex is not locked (e.g., during single-threaded init).
     */
    __attribute__((always_inline))
    auto num_valid_nbrs(const vertex_id_t src, const uint64_t count) -> void {
        _header_ref(src).store(_make_header(count, false), std::memory_order_relaxed);
    }

    /**
     * @brief Fetch the full per-vertex block including header and neighbor slots (const).
     * @param src The layer_vid of the source vertex.
     * @return A const span of (1 + max_nbr_size) lnbr_t entries.
     *
     * The returned span has the following layout:
     *   - span[0] (header): reinterpreted as @c std::atomic<uint64_t> holding
     *     the valid neighbor count. Do NOT access it as a regular lnbr_t.
     *   - span[1 .. max_nbr_size]: the neighbor slots.
     */
    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) const -> std::span<const lnbr_t> {
        return std::span<const lnbr_t>(
            &_csr_nbrs[static_cast<size_t>(src) * _stride()],
            _stride()
        );
    }

    /**
     * @brief Fetch the full per-vertex block including header and neighbor slots (mutable).
     * @param src The layer_vid of the source vertex.
     * @return A mutable span of (1 + max_nbr_size) lnbr_t entries.
     *
     * @see fetch_nbrs(vertex_id_t) const for the span layout description.
     */
    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) -> std::span<lnbr_t> {
        return std::span<lnbr_t>(
            &_csr_nbrs[static_cast<size_t>(src) * _stride()],
            _stride()
        );
    }

    // --- Per-Vertex Metadata ---

    /**
     * @brief Get this vertex's id in the base dataset.
     * @param layer_vid The vertex's layer_vid in the current layer.
     * @return The corresponding base_vid. Write-once at @c add_vertex time;
     *         safe to read concurrently with any mutator.
     */
    __attribute__((always_inline))
    auto get_base_vid(const vertex_id_t layer_vid) const -> vertex_id_t {
        return _vertex_info[layer_vid].base_vid;
    }

    /**
     * @brief Get the layer_vid of a vertex in the next (lower) layer.
     * @param layer_vid The vertex's layer_vid in the current layer.
     * @return The corresponding layer_vid in the next layer. Initially a
     *         placeholder equal to @c base_vid; callers overwrite via
     *         @c set_inter_layer_link once the lower-layer id is known.
     */
    __attribute__((always_inline))
    auto get_inter_layer_link(const vertex_id_t layer_vid) const -> vertex_id_t {
        return _vertex_info[layer_vid].layer_vid;
    }

    /**
     * @brief Set the inter-layer link for a vertex.
     * @param layer_vid The vertex's layer_vid in the current layer.
     * @param next_layer_vid The vertex's layer_vid in the next (lower) layer.
     */
    __attribute__((always_inline))
    auto set_inter_layer_link(const vertex_id_t layer_vid, const vertex_id_t next_layer_vid) -> void {
        _vertex_info[layer_vid].layer_vid = next_layer_vid;
    }

    /**
     * @brief Get the per-vertex metadata concurrent vector (const).
     */
    __attribute__((always_inline))
    auto get_vertex_info() const -> const tbb::concurrent_vector<lnbr_t>& {
        return _vertex_info;
    }

    /**
     * @brief Get the per-vertex metadata concurrent vector (mutable).
     */
    __attribute__((always_inline))
    auto get_vertex_info() -> tbb::concurrent_vector<lnbr_t>& {
        return _vertex_info;
    }

    // --- Properties ---

    __attribute__((always_inline))
    auto max_nbr_size() const -> vertex_num_t {
        return _max_nbr_size;
    }

    __attribute__((always_inline))
    auto max_nbr_size(const vertex_num_t new_size) -> void {
        _max_nbr_size = new_size;
    }

    /** @brief Number of vertices currently inserted. */
    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return static_cast<vertex_num_t>(_vertex_info.size());
    }

    __attribute__((always_inline))
    auto get_max_num_vertices() const -> vertex_num_t {
        return _max_num_vertices;
    }

    __attribute__((always_inline))
    auto get_csr_nbrs() const -> const csr_lnbrs_t& {
        return _csr_nbrs;
    }

    __attribute__((always_inline))
    auto get_csr_nbrs() -> csr_lnbrs_t& {
        return _csr_nbrs;
    }

private:
    /**
     * @brief Grow the CSR array so that at least @p required_capacity
     *        vertices can be stored. Doubles the capacity each time
     *        (amortized O(1) growth). Thread-safe: holds @c _resize_mutex
     *        for the duration of the resize; concurrent add_nbr / fetch_nbrs
     *        on already-initialized vertices are NOT blocked (they access
     *        indices < old capacity whose memory is stable after resize).
     */
    auto _ensure_capacity(const vertex_num_t required_capacity) -> void {
        std::lock_guard<std::mutex> guard(_resize_mutex);

        // Re-check under lock (another thread may have already resized).
        if (required_capacity <= _max_num_vertices) return;

        const vertex_num_t new_cap =
            std::max<vertex_num_t>(required_capacity, _max_num_vertices * 2);
        _csr_nbrs.resize(static_cast<size_t>(new_cap) * _stride());
        _max_num_vertices = new_cap;
    }

    /** @brief Current CSR capacity (number of vertex slots). Grows via _ensure_capacity. */
    vertex_num_t _max_num_vertices;

    /** @brief Fixed number of neighbor slots per vertex. */
    vertex_num_t _max_nbr_size;

    /**
     * @brief CSR-format neighbor storage.
     *
     * Each vertex occupies (1 + _max_nbr_size) contiguous lnbr_t slots.
     * The first slot of each block is reinterpreted as std::atomic<uint64_t>
     * to hold num_valid_nbrs. Grown via _ensure_capacity when exhausted.
     */
    csr_lnbrs_t _csr_nbrs;

    /**
     * @brief Concurrent per-vertex metadata: _vertex_info[layer_vid] stores
     *        an @c lnbr_t whose @c .base_vid is this vertex's id in the
     *        base dataset (write-once) and whose @c .layer_vid is the
     *        layer_vid in the next (lower) layer (initially a placeholder
     *        equal to base_vid, overwritten via @c set_inter_layer_link).
     *
     * push_back() is used in add_vertex() to atomically claim a new layer_vid.
     */
    tbb::concurrent_vector<lnbr_t> _vertex_info;

    /** @brief Serializes CSR resize operations. */
    mutable std::mutex _resize_mutex;

};  // class InternalGraph

}   // namespace dynamic
}   // namespace cpu
}   // namespace artea
