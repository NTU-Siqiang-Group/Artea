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

#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <immintrin.h>
#include <artea/cpu/containers/allocator.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief High-Performance Linear Scan Priority Queue for Small Candidate Sets (N <= 64).
 *
 * Design Philosophy:
 * For small candidate sets (typically N <= 64), maintaining a heap structure introduces
 * unnecessary overhead. This implementation uses a sorted linear array with lazy deletion,
 * which provides better cache locality and simpler code paths.
 *
 * Key Features:
 * 1. Sorted Array: Candidates are kept in sorted order by distance (ascending).
 * 2. Lazy Deletion: Instead of removing explored candidates, we mark them and skip during traversal.
 * 3. Fast Rejection: O(1) distance threshold check before insertion.
 * 4. Cache-Friendly: Linear memory layout optimizes CPU cache utilization.
 *
 * Performance Characteristics:
 * - try_push: O(L) due to sorted insertion (acceptable for small N)
 * - pop_best_unexplored: O(L) worst case, but amortized O(1) with cursor optimization
 * - Memory: O(L) with cache-aligned allocation for SIMD operations
 *
 * @tparam RouterTraitsT Traits defining vertex types, distance types, and candidate entry types.
 */
template <typename RouterTraitsT>
class LinearCandidateQueue {

public:
    // Type aliases for clarity and convenience
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using candidate_entry_t = typename RouterTraitsT::candidate_entry_t;
    using entry_comp_t = typename RouterTraitsT::entry_comp_t;
    using knn_results_t = typename RouterTraitsT::knn_results_t;
    using random_seq_t = typename RouterTraitsT::random_seq_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using visited_table_t = typename RouterTraitsT::visited_table_t;

    /** @brief Container for candidate entries. */
    using container_t = std::vector<candidate_entry_t>;

    /** @brief Maximum possible distance value (used for threshold initialization). */
    static constexpr distance_t max_distance = RouterTraitsT::max_distance;

    /** @brief Invalid vertex ID constant. */
    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

    /** @brief Comparator instance for candidate entry comparison. */
    static constexpr entry_comp_t entry_comparator = entry_comp_t();

    /** @brief Sentinel value representing an invalid/non-existent candidate. */
    static constexpr candidate_entry_t invalid_candidate_entry = candidate_entry_t::make_invalid_entry();

    /**
     * @brief Construct a LinearCandidateQueue with a fixed capacity.
     * @param capacity Maximum number of candidates to maintain in the queue.
     * @note Reserves capacity + 1 to avoid reallocation during insertion before trimming.
     */
    explicit LinearCandidateQueue(std::size_t capacity) : _capacity(capacity) {
        _data.reserve(capacity + 1);
    }

    // Prevent accidental copy
    LinearCandidateQueue(const LinearCandidateQueue&) = delete;
    LinearCandidateQueue& operator=(const LinearCandidateQueue&) = delete;

    // Allow move semantics for efficient transfer of ownership
    LinearCandidateQueue(LinearCandidateQueue&&) = default;
    LinearCandidateQueue& operator=(LinearCandidateQueue&&) = default;

    /**
     * @brief Initialize the queue with a set of candidate entries.
     * @param init_candidates Initial set of candidates (will be moved, not copied).
     * @note Sorts candidates by distance, trims to capacity, and updates threshold.
     * @complexity O(N log N) due to sorting.
     */
    void initialize(container_t& init_candidates) {
        #ifndef NDEBUG
        assert(_capacity == init_candidates.size() && "Init candidates size must equal capacity!");
        #endif
        _check_cursor = 0;
        _data = std::move(init_candidates);
        std::sort(_data.begin(), _data.end(), entry_comparator);
        _update_thresh_distance();
    }

    /**
     * @brief Initialize the queue with random candidate entries.
     * @param random_seq Random sequence generator for generating vertex IDs (thread-safe).
     * @param dist_func Distance function for computing distances.
     * @param query_vec Pointer to the query vector.
     * @param base_vecs Reference to the base vector array.
     * @param visited_table Reference to visited table to mark initial candidates.
     * @note Generates random IDs, computes distances using dist_func, and calls seeded_initialize.
     * @complexity O(N) for generation + O(N log N) for sorting.
     */
    void random_initialize(
        random_seq_t& random_seq,
        const dist_func_t& dist_func,
        const vec_ele_t* query_vec,
        const vector_array_t& base_vecs,
        visited_table_t& visited_table
    ) {
        // Generate random vertex IDs directly into std::vector
        std::vector<vertex_id_t> init_vids(_capacity);
        random_seq.generate(init_vids, _capacity);

        // Call seeded_initialize with the generated IDs
        seeded_initialize(init_vids, dist_func, query_vec, base_vecs, visited_table);
    }

    /**
     * @brief Initialize the queue with a vector of vertex IDs.
     * @param init_vids Vector of vertex IDs to initialize the queue with.
     * @param dist_func Distance function for computing distances.
     * @param query_vec Pointer to the query vector.
     * @param base_vecs Reference to the base vector array.
     * @param visited_table Reference to visited table to mark initial candidates.
     * @note If init_vids.size() > capacity, only the best capacity candidates are kept.
     * @complexity O(N) for distance computation + O(N log N) for sorting.
     */
    void seeded_initialize(
        const std::vector<vertex_id_t>& init_vids,
        const dist_func_t& dist_func,
        const vec_ele_t* query_vec,
        const vector_array_t& base_vecs,
        visited_table_t& visited_table
    ) {
        // Create candidate entries with computed distances
        container_t init_candidates;
        init_candidates.reserve(init_vids.size());
        for (vertex_id_t vid : init_vids) {
            const vec_ele_t* base_vec = base_vecs.get(vid);
            distance_t dist = dist_func(query_vec, base_vec);
            init_candidates.emplace_back(vid, dist);
        }

        // If we have more candidates than capacity, keep only the best capacity candidates
        if (init_candidates.size() > _capacity) {
            std::partial_sort(
                init_candidates.begin(),
                init_candidates.begin() + _capacity,
                init_candidates.end(),
                entry_comparator
            );
            init_candidates.resize(_capacity);
        }

        _check_cursor = 0;
        _data = std::move(init_candidates);
        std::sort(_data.begin(), _data.end(), entry_comparator);

        // Mark all initial candidates as visited
        for (const auto& entry : _data) {
            visited_table.set(entry.get_id());
        }

        _update_thresh_distance();
    }

    /** @brief Check if the queue is empty. */
    __attribute__((always_inline))
    auto empty() const -> bool { return _data.empty(); }

    /** @brief Get the approximate memory footprint of this queue in bytes. */
    __attribute__((always_inline))
    auto size() const -> std::size_t {
        return _data.capacity() * sizeof(candidate_entry_t) + sizeof(*this);
    }

    /** @brief Get the number of result candidates currently maintained. */
    __attribute__((always_inline))
    auto get_result_size() const -> std::size_t { return _data.size(); }

    /** @brief Get the number of unexplored candidates remaining. */
    __attribute__((always_inline))
    auto get_unexplored_size() const -> std::size_t {
        std::size_t count = 0;
        for (std::size_t i = _check_cursor; i < _data.size(); ++i) {
            if (_data[i].is_unexplored()) { ++count; }
        }
        return count;
    }

    /** @brief Get the maximum capacity of the queue. */
    __attribute__((always_inline))
    auto capacity() const -> std::size_t { return _capacity; }

    /** @brief Clear all candidates and reset internal state. */
    __attribute__((always_inline))
    auto clear() -> void {
        _data.clear();
        _check_cursor = 0;
        _thresh_distance = max_distance;
    }

    /**
     * @brief Attempt to insert a new candidate entry into the queue.
     *
     * Uses fast rejection (O(1)) for entries that are too far, then performs
     * binary search insertion to maintain sorted order. Updates cursor and
     * threshold as needed.
     *
     * @param vertex_id The vertex ID to insert.
     * @param distance The distance to the vertex.
     * @return true if the entry was inserted, false if rejected.
     * @complexity O(N) worst case due to insertion shift, but O(1) for rejected entries.
     */
    __attribute__((always_inline))
    auto try_push(vertex_id_t vertex_id, distance_t distance) -> bool {
        // Fast rejection: O(1) check before expensive operations
        if (_data.size() >= _capacity && distance >= _thresh_distance) {
            return false;
        }

        // Create a new unexplored entry
        candidate_entry_t entry(vertex_id, distance, false);

        auto it = std::upper_bound(_data.begin(), _data.end(), entry, entry_comparator);
        std::size_t insert_place = it - _data.begin();
        _data.insert(it, entry);    // O(L) memmove

        if (_data.size() > _capacity) {
            _data.pop_back();
        }
        if (insert_place < _check_cursor) {
            _check_cursor = insert_place;
        }
        if (_data.size() >= _capacity) {
            _thresh_distance = _data.back().distance;
        }
        return true;
    }

    /**
     * @brief Retrieve and mark the best unexplored candidate.
     *
     * Uses embedded status bits to track explored state. A cursor tracks the position
     * of the last explored candidate to avoid redundant scans.
     *
     * @return Pair of (vertex_id, distance) for the best unexplored candidate,
     *         or (invalid_vertex_id, max_distance) if none found.
     * @complexity O(N) worst case, but amortized O(1) with cursor optimization.
     */
    __attribute__((always_inline))
    auto pop_best_unexplored() -> std::pair<vertex_id_t, distance_t> {
        _move_to_unexplored();

        if (_check_cursor >= _data.size()) {
            return {invalid_vertex_id, max_distance};
        }

        // Mark as explored and advance cursor
        _data[_check_cursor].mark_as_explored();
        auto& entry = _data[_check_cursor];
        _check_cursor++;

        return {entry.get_id(), entry.get_distance()};
    }

    /**
     * @brief Check if the search should terminate early.
     *
     * Early Termination Condition:
     * If the closest unexplored candidate is farther than the threshold distance,
     * and the queue is full, then no better candidates can be found.
     *
     * @return true if search should terminate, false otherwise.
     */
    __attribute__((always_inline))
    auto should_terminate() -> bool {
        _move_to_unexplored();

        // No unexplored candidates left
        if (_check_cursor >= _data.size()) {
            return true;
        }

        // Check if queue is full and the closest unexplored candidate is too far
        if (_data.size() >= _capacity && _data[_check_cursor].distance > _thresh_distance) {
            return true;
        }

        return false;
    }

    /**
     * @brief Extract the top-k results from the candidate queue.
     * @param k Number of top results to extract.
     * @return knn_results_t of result entries sorted by distance (ascending order).
     * @note The data is already sorted, so we just extract the first k entries.
     * @note After calling this method, the candidate queue may be in an invalid state.
     */
    auto extract_results(std::size_t k) -> knn_results_t {
        #ifndef NDEBUG
        assert(_data.size() >= k && "Not enough candidates in queue");
        #endif

        knn_results_t results = std::move(_data);
        results.resize(k);
        return results;
    }

private:
    /** @brief Sorted array of candidate entries (ascending by distance). */
    container_t _data;

    /** @brief Maximum number of candidates to maintain in the queue. */
    std::size_t _capacity;

    /**
     * @brief Cursor tracking the position of the last explored candidate.
     * Used to skip already-explored entries during pop_best_unexplored().
     */
    std::size_t _check_cursor = 0;

    /**
     * @brief Distance threshold for fast rejection in try_push().
     * Set to the distance of the worst candidate when queue is full.
     */
    distance_t _thresh_distance = max_distance;

    /**
     * @brief Update the distance threshold based on current queue state.
     * @note Called after initialization or when queue reaches capacity.
     */
    __attribute__((always_inline))
    auto _update_thresh_distance() -> void {
        _thresh_distance = (_data.size() >= _capacity && !_data.empty()) ?
            _data.back().distance : max_distance;
    }

    /**
     * @brief Move the cursor to the next unexplored candidate.
     * @note This method advances _check_cursor to point to the next unexplored entry.
     */
    __attribute__((always_inline))
    auto _move_to_unexplored() -> void {
        while (_check_cursor < _data.size() && _data[_check_cursor].is_explored()) {
            _check_cursor++;
        }
    }

};  // class LinearCandidateQueue

}   // namespace cpu
}   // namespace artea