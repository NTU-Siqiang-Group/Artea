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
 * @FilePath: /Artea/include/artea/cpu/router/std_candidate_queue.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Dual-heap candidate queue implementation inspired by hnswlib.
 */

#pragma once

#include <queue>
#include <vector>
#include <limits>
#include <algorithm>
#include <cassert>

namespace artea {
namespace cpu {

/**
 * @brief Dual-Heap Candidate Queue Implementation (inspired by hnswlib).
 *
 * Design Philosophy:
 * This implementation uses two priority queues to efficiently manage candidates during
 * graph search, following the approach used in hnswlib:
 *
 * 1. Unexplored Set (Min-Heap): Stores candidates to be explored, ordered by distance (ascending).
 *    The top element is the closest unexplored candidate.
 *
 * 2. Top Candidates (Max-Heap): Maintains the best L candidates found so far, ordered by
 *    distance (descending). The top element is the worst among the best L candidates.
 *
 * Key Advantages over Linear Scan:
 * - O(log L) insertion instead of O(L) for sorted array
 * - O(log L) extraction instead of O(L) linear scan
 * - Better for larger candidate sets (L > 64)
 * - No need for manual sorting or cursor tracking
 *
 * Trade-offs:
 * - Higher constant factors due to heap operations
 * - Less cache-friendly than linear array
 * - More memory overhead for heap structure
 *
 * Performance Characteristics:
 * - try_push: O(log L) for heap insertion
 * - pop_best_unexplored: O(log L) for heap extraction
 * - Memory: O(L) for both heaps
 *
 * @tparam RouterTraitsT Traits defining vertex types, distance types, and candidate entry types.
 */
template <typename RouterTraitsT>
class StdCandidateQueue {

public:
    // Type aliases for clarity and convenience
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using candidate_entry_t = typename RouterTraitsT::candidate_entry_t;
    using random_seq_t = typename RouterTraitsT::random_seq_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;

    /** @brief Maximum possible distance value (used for threshold initialization). */
    static constexpr distance_t max_distance = RouterTraitsT::max_distance;

    /** @brief Invalid vertex ID constant. */
    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

    /** @brief Sentinel value representing an invalid/non-existent candidate. */
    static constexpr candidate_entry_t invalid_candidate_entry = RouterTraitsT::invalid_candidate_entry;

    /**
     * @brief Comparator for Min-Heap (smallest distance at top).
     * Used for the unexplored set to prioritize closest unexplored candidates.
     */
    struct MinHeapComparator {
        __attribute__((always_inline))
        constexpr bool operator()(const candidate_entry_t& a, const candidate_entry_t& b) const noexcept {
            return a.distance > b.distance;  // Reverse comparison for min-heap
        }
    };

    /**
     * @brief Comparator for Max-Heap (largest distance at top).
     * Used for top candidates to maintain the best L candidates and quickly access the worst.
     */
    struct MaxHeapComparator {
        __attribute__((always_inline))
        constexpr bool operator()(const candidate_entry_t& a, const candidate_entry_t& b) const noexcept {
            return a.distance < b.distance;  // Standard comparison for max-heap
        }
    };

    /** @brief Min-heap type for unexplored set (closest candidates at top). */
    using min_heap_t = std::priority_queue<candidate_entry_t,
                                           std::vector<candidate_entry_t>,
                                           MinHeapComparator>;

    /** @brief Max-heap type for top candidates (worst of best at top). */
    using max_heap_t = std::priority_queue<candidate_entry_t,
                                           std::vector<candidate_entry_t>,
                                           MaxHeapComparator>;

    /**
     * @brief Construct a StdCandidateQueue with a fixed capacity.
     * @param capacity Maximum number of top candidates to maintain (similar to ef_search in hnswlib).
     */
    explicit StdCandidateQueue(std::size_t capacity)
        : _capacity(capacity), _lower_bound(max_distance) {
    }

    // Prevent accidental copy
    StdCandidateQueue(const StdCandidateQueue&) = delete;
    StdCandidateQueue& operator=(const StdCandidateQueue&) = delete;

    // Allow move semantics for efficient transfer of ownership
    StdCandidateQueue(StdCandidateQueue&&) = default;
    StdCandidateQueue& operator=(StdCandidateQueue&&) = default;

    /**
     * @brief Initialize the queue with a set of candidate entries.
     * @param init_candidates Initial set of candidates to populate both heaps.
     * @note All candidates are added to unexplored_set for exploration and top_candidates for results.
     * @complexity O(N log N) where N is the number of initial candidates.
     */
    void initialize(const std::vector<candidate_entry_t>& init_candidates) {
        #ifndef NDEBUG
        assert(_capacity == init_candidates.size() && "Init candidates size must equal capacity!");
        #endif
        // Clear existing state
        _unexplored_set = min_heap_t();
        _top_candidates = max_heap_t();
        _lower_bound = max_distance;

        // Add all candidates to both heaps
        for (auto& entry : init_candidates) {
            _unexplored_set.push(entry);
            _top_candidates.push(entry);
        }

        // Update lower bound
        _update_lower_bound();
    }

    /**
     * @brief Initialize the queue with random candidate entries.
     * @param random_seq Random sequence generator for generating vertex IDs (thread-safe).
     * @param dist_func Distance function for computing distances.
     * @param query_vec Pointer to the query vector.
     * @param base_vecs Reference to the base vector array.
     * @note Generates random IDs, computes distances using dist_func, and calls seeded_initialize.
     * @complexity O(N) for generation + O(N log N) for heap operations.
     */
    void random_initialize(
        random_seq_t& random_seq,
        const dist_func_t& dist_func,
        const vec_ele_t* query_vec,
        const vector_array_t& base_vecs
    ) {
        // Generate random vertex IDs directly into std::vector
        std::vector<vertex_id_t> init_vids(_capacity);
        random_seq.generate(init_vids, _capacity);

        // Call seeded_initialize with the generated IDs
        seeded_initialize(init_vids, dist_func, query_vec, base_vecs);
    }

    /**
     * @brief Initialize the queue with a vector of vertex IDs.
     * @param init_vids Vector of vertex IDs to initialize the queue with.
     * @param dist_func Distance function for computing distances.
     * @param query_vec Pointer to the query vector.
     * @param base_vecs Reference to the base vector array.
     * @note If init_vids.size() > capacity, only the best capacity candidates are kept.
     * @complexity O(N) for distance computation + O(N log N) for sorting + O(L log L) for heap operations.
     */
    void seeded_initialize(
        const std::vector<vertex_id_t>& init_vids,
        const dist_func_t& dist_func,
        const vec_ele_t* query_vec,
        const vector_array_t& base_vecs
    ) {
        // Create candidate entries with computed distances
        std::vector<candidate_entry_t> init_candidates;
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
                [](const candidate_entry_t& a, const candidate_entry_t& b) {
                    return a.distance < b.distance;
                }
            );
            init_candidates.resize(_capacity);
        }

        _unexplored_set = min_heap_t();
        _top_candidates = max_heap_t();
        _lower_bound = max_distance;

        for (auto& entry : init_candidates) {
            _unexplored_set.push(entry);
            _top_candidates.push(entry);
        }

        _update_lower_bound();
    }

    /** @brief Check if the unexplored set is empty (no more candidates to explore). */
    __attribute__((always_inline))
    auto empty() const -> bool {
        return _unexplored_set.empty();
    }

    /** @brief Get the approximate memory footprint of this queue in bytes. */
    __attribute__((always_inline))
    auto size() const -> std::size_t {
        return (_unexplored_set.size() + _top_candidates.size()) * sizeof(candidate_entry_t)
               + sizeof(*this);
    }

    /** @brief Get the current number of candidates in the exploration set. */
    __attribute__((always_inline))
    auto get_unexplored_size() const -> std::size_t {
        return _unexplored_set.size();
    }

    /** @brief Get the number of top candidates currently maintained. */
    __attribute__((always_inline))
    auto get_result_size() const -> std::size_t {
        return _top_candidates.size();
    }

    /** @brief Get the maximum capacity of top candidates. */
    __attribute__((always_inline))
    auto capacity() const -> std::size_t {
        return _capacity;
    }

    /** @brief Get the current lower bound (worst distance in top candidates). */
    __attribute__((always_inline))
    auto lower_bound() const -> distance_t {
        return _lower_bound;
    }

    /**
     * @brief Clear all candidates and reset internal state.
     * Clears both the unexplored set and top candidates heaps.
     */
    __attribute__((always_inline))
    auto clear() -> void {
        _unexplored_set = min_heap_t();
        _top_candidates = max_heap_t();
        _lower_bound = max_distance;
    }

    /**
     * @brief Attempt to insert a new candidate entry into the queue.
     *
     * Insertion Strategy (following hnswlib approach):
     * 1. Fast Rejection: If top_candidates is full and new distance >= lower_bound, reject (O(1)).
     * 2. Add to Unexplored Set: Insert into min-heap for future exploration (O(log L)).
     * 3. Add to Top Candidates: Insert into max-heap to maintain best L candidates (O(log L)).
     * 4. Trim: If top_candidates exceeds capacity, remove worst candidate (O(log L)).
     * 5. Update Lower Bound: Set to the worst distance in top_candidates (O(1)).
     *
     * @param vertex_id The vertex ID to insert.
     * @param distance The distance to the vertex.
     * @return true if the entry was inserted, false if rejected.
     * @complexity O(log L) for heap operations, but O(1) for rejected entries.
     */
    __attribute__((always_inline))
    auto try_push(vertex_id_t vertex_id, distance_t distance) -> bool {
        // Fast rejection: O(1) check before expensive heap operations
        if (_top_candidates.size() >= _capacity && distance >= _lower_bound) {
            return false;
        }

        candidate_entry_t entry(vertex_id, distance);

        // Add to unexplored set for exploration (min-heap)
        _unexplored_set.push(entry);

        // Add to top candidates for results (max-heap)
        _top_candidates.push(entry);

        // Maintain capacity constraint
        if (_top_candidates.size() > _capacity) {
            _top_candidates.pop();
        }

        // Update lower bound
        _update_lower_bound();

        return true;
    }

    /**
     * @brief Retrieve and mark the best unexplored candidate.
     *
     * Extraction Strategy:
     * - Pop the top element from unexplored_set (min-heap), which is the closest unexplored candidate.
     * - Mark it as explored before returning.
     * - If candidate_set is empty, return (invalid_vertex_id, max_distance).
     *
     * @return Pair of (vertex_id, distance) for the best unexplored candidate,
     *         or (invalid_vertex_id, max_distance) if none found.
     * @complexity O(log L) for heap extraction.
     */
    __attribute__((always_inline))
    auto pop_best_unexplored() -> std::pair<vertex_id_t, distance_t> {
        if (_unexplored_set.empty()) {
            return {invalid_vertex_id, max_distance};
        }
        // Extract the closest candidate
        candidate_entry_t entry = _unexplored_set.top();
        _unexplored_set.pop();
        return {entry.get_id(), entry.get_distance()};
    }

    /**
     * @brief Check if the search should terminate early.
     *
     * Early Termination Condition (from hnswlib):
     * If the closest unexplored candidate is farther than the worst candidate in top_candidates,
     * and top_candidates is full, then we can stop searching because no better candidates exist.
     *
     * @return true if search should terminate, false otherwise.
     */
    __attribute__((always_inline))
    auto should_terminate() const -> bool {
        if (_unexplored_set.empty()) {
            return true;
        }
        if (_top_candidates.size() < _capacity) {
            return false;
        }
        // Check if closest unexplored > worst in top candidates
        return _unexplored_set.top().distance > _lower_bound;
    }

    /**
     * @brief Extract the top-k results from the candidate queue.
     * @param k Number of top results to extract.
     * @return Vector of (vertex_id, distance) pairs sorted by distance (ascending order).
     * @note This method empties the top_candidates heap.
     */
    auto extract_results(std::size_t k) -> std::vector<std::pair<vertex_id_t, distance_t>> {
        #ifndef NDEBUG
        assert(_top_candidates.size() >= k && "Not enough candidates in queue");
        #endif

        std::vector<candidate_entry_t> result_candidates;
        result_candidates.reserve(k);

        // Extract all candidates from max-heap (they come out in descending order)
        while (!_top_candidates.empty()) {
            result_candidates.push_back(_top_candidates.top());
            _top_candidates.pop();
        }

        // Reverse to get ascending order (O(n) instead of O(n log n) sort)
        std::reverse(result_candidates.begin(), result_candidates.end());

        // Extract (vertex_id, distance) pairs
        std::vector<std::pair<vertex_id_t, distance_t>> results;
        results.reserve(k);
        for (std::size_t i = 0; i < k && i < result_candidates.size(); ++i) {
            results.emplace_back(result_candidates[i].get_id(), result_candidates[i].get_distance());
        }

        return results;
    }

    /**
     * @brief Extract the top-k result IDs from the candidate queue.
     * @param k Number of top results to extract.
     * @return Vector of vertex IDs sorted by distance (ascending order).
     * @note This method empties the top_candidates heap.
     */
    auto extract_result_ids(std::size_t k) -> std::vector<vertex_id_t> {
        #ifndef NDEBUG
        assert(_top_candidates.size() >= k && "Not enough candidates in queue");
        #endif

        std::vector<candidate_entry_t> result_candidates;
        result_candidates.reserve(k);

        // Extract all candidates from max-heap (they come out in descending order)
        while (!_top_candidates.empty()) {
            result_candidates.push_back(_top_candidates.top());
            _top_candidates.pop();
        }

        // Reverse to get ascending order (O(n) instead of O(n log n) sort)
        std::reverse(result_candidates.begin(), result_candidates.end());

        // Extract vertex IDs
        std::vector<vertex_id_t> result_ids;
        result_ids.reserve(k);
        for (std::size_t i = 0; i < k && i < result_candidates.size(); ++i) {
            result_ids.push_back(result_candidates[i].get_id());
        }

        return result_ids;
    }

private:
    /**
     * @brief Unexplored set (min-heap): stores candidates to be explored.
     * Top element is the closest unexplored candidate.
     */
    min_heap_t _unexplored_set;

    /**
     * @brief Top candidates (max-heap): maintains the best L candidates found so far.
     * Top element is the worst among the best L candidates.
     */
    max_heap_t _top_candidates;

    /** @brief Maximum number of top candidates to maintain. */
    std::size_t _capacity;

    /**
     * @brief Distance threshold for fast rejection in try_push().
     * Set to the distance of the worst candidate in top_candidates.
     */
    distance_t _lower_bound;

    /**
     * @brief Update the lower bound based on current top_candidates state.
     * @note Called after insertion or when top_candidates changes.
     */
    __attribute__((always_inline))
    auto _update_lower_bound() -> void {
        _lower_bound = (!_top_candidates.empty()) ?
            _top_candidates.top().distance : max_distance;
    }

};  // class StdCandidateQueue

}   // namespace cpu
}   // namespace artea
