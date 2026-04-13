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
#include <functional>
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>
#include <artea/cpu/containers/four_ary_heap.hpp>
#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Dual FourAryHeap Candidate Queue Implementation (inspired by hnswlib).
 *
 * Design Philosophy:
 * This implementation uses two FourAryHeap instances to efficiently manage candidates during
 * graph search, following the approach used in hnswlib:
 *
 * 1. Unexplored Set (Min-Heap): Stores candidates to be explored, ordered by distance (ascending).
 *    The top element is the closest unexplored candidate.
 *
 * 2. Top Candidates (Max-Heap): Maintains the best L candidates found so far, ordered by
 *    distance (descending). The top element is the worst among the best L candidates.
 *
 * Performance Characteristics:
 * - try_push: O(log_4 L) for heap insertion
 * - pop_best_unexplored: O(log_4 L) for heap extraction
 * - Memory: O(L) for both heaps
 *
 * @tparam RouterTraitsT Traits defining vertex types, distance types, and candidate entry types.
 * @tparam EntryT Candidate-entry type carried by the queue. Defaults to the
 *         router traits' candidate entry, preserving existing behavior.
 */
template <typename RouterTraitsT,
          typename EntryT = typename RouterTraitsT::candidate_entry_t>
class FHCandidateQueue {

public:
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using candidate_entry_t = EntryT;
    /** @brief knn_results_t is a queue-local type that tracks @c EntryT. */
    using knn_results_t = std::vector<candidate_entry_t>;
    using container_t = cache_aligned_container_t<candidate_entry_t>;
    using random_seq_t = typename RouterTraitsT::random_seq_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using visited_table_t = typename RouterTraitsT::visited_table_t;

    /** @brief Maximum possible distance value (used for threshold initialization). */
    static constexpr distance_t max_distance = RouterTraitsT::max_distance;

    /** @brief Invalid vertex ID constant. */
    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

    /** @brief Sentinel value with max distance (for min-heap sentinel). */
    static constexpr candidate_entry_t invalid_candidate_entry =
        candidate_entry_t::make_invalid_entry();

    /** @brief Sentinel value with min distance (for max-heap sentinel). */
    static constexpr candidate_entry_t min_candidate_entry =
        candidate_entry_t::make_min_entry();

    /**
     * @brief Min-heap type (smallest distance at top).
     * Compare = std::greater: comparator(a, b) = (a > b), so b wins when b < a.
     * Sentinel = invalid_candidate_entry (max distance), never wins.
     */
    using min_heap_t = FourAryHeap<candidate_entry_t, container_t, std::greater<candidate_entry_t>>;

    /**
     * @brief Max-heap type (largest distance at top).
     * Compare = std::less (default): comparator(a, b) = (a < b), so b wins when b > a.
     * Sentinel = min_candidate_entry (min distance), never wins.
     */
    using max_heap_t = FourAryHeap<candidate_entry_t, container_t, std::less<candidate_entry_t>>;

    /**
     * @brief Construct a FHCandidateQueue with a fixed capacity.
     * @param capacity Maximum number of top candidates to maintain.
     */
    explicit FHCandidateQueue(std::size_t capacity)
        : _unexplored_set(invalid_candidate_entry),
          _top_candidates(min_candidate_entry),
          _capacity(capacity),
          _lower_bound(max_distance) {
    }

    // Prevent accidental copy
    FHCandidateQueue(const FHCandidateQueue&) = delete;
    FHCandidateQueue& operator=(const FHCandidateQueue&) = delete;

    // Allow move semantics
    FHCandidateQueue(FHCandidateQueue&&) = default;
    FHCandidateQueue& operator=(FHCandidateQueue&&) = default;

    /**
     * @brief Initialize the queue with a set of candidate entries.
     * @param init_candidates Initial set of candidates to populate both heaps.
     * @complexity O(N log N) where N is the number of initial candidates.
     */
    void initialize(const std::vector<candidate_entry_t>& init_candidates) {
        #ifndef NDEBUG
        assert(_capacity == init_candidates.size() && "Init candidates size must equal capacity!");
        #endif
        _unexplored_set.clear();
        _top_candidates.clear();
        _lower_bound = max_distance;

        for (auto& entry : init_candidates) {
            _unexplored_set.push(entry);
            _top_candidates.push(entry);
        }

        _update_lower_bound();
    }

    /**
     * @brief Initialize the queue with random candidate entries.
     * @param random_seq Random sequence generator for generating vertex IDs (thread-safe).
     * @param dist_func Distance function for computing distances.
     * @param query_vec Pointer to the query vector.
     * @param base_vecs Reference to the base vector array.
     * @param visited_table Reference to visited table to mark initial candidates.
     * @note Generates random IDs, computes distances using dist_func, and calls seeded_initialize.
     * @complexity O(N) for generation + O(N log N) for heap operations.
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
        random_seq.generate(init_vids, base_vecs.get_num_vecs(), _capacity);

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
     * @complexity O(N) for distance computation + O(N log N) for sorting + O(L log L) for heap operations.
     */
    void seeded_initialize(
        const std::vector<vertex_id_t>& init_vids,
        const dist_func_t& dist_func,
        const vec_ele_t* query_vec,
        const vector_array_t& base_vecs,
        visited_table_t& visited_table
    ) {
        // This overload assumes a 2-arg entry constructor
        // (vertex_id, distance). Multi-layer callers should seed manually via
        // try_push(...) instead.
        static_assert(
            std::is_constructible_v<candidate_entry_t, vertex_id_t, distance_t>,
            "seeded_initialize requires an entry with (vid, dist) constructor");

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
                    return a.get_distance() < b.get_distance();
                }
            );
            init_candidates.resize(_capacity);
        }

        _unexplored_set.clear();
        _top_candidates.clear();
        _lower_bound = max_distance;

        for (auto& entry : init_candidates) {
            _unexplored_set.push(entry);
            _top_candidates.push(entry);
            visited_table.set(entry.get_vid());
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

    /**
     * @brief Adjust the result-set capacity.
     *
     *   - Upward resize: contents preserved, threshold relaxes.
     *   - Downward resize: pop worst-distance entries from
     *     @c _top_candidates until size <= @p new_capacity. The
     *     unexplored-set is not touched.
     */
    __attribute__((always_inline))
    auto set_capacity(std::size_t new_capacity) -> void {
        _capacity = new_capacity;
        while (_top_candidates.size() > _capacity) {
            _top_candidates.pop();
        }
        _update_lower_bound();
    }

    /** @brief Get the current lower bound (worst distance in top candidates). */
    __attribute__((always_inline))
    auto lower_bound() const -> distance_t {
        return _lower_bound;
    }

    /**
     * @brief Clear all candidates and reset internal state.
     */
    __attribute__((always_inline))
    auto clear() -> void {
        _unexplored_set.clear();
        _top_candidates.clear();
        _lower_bound = max_distance;
    }

    /**
     * @brief Attempt to insert a new candidate entry into the queue.
     *
     * Variadic-perfect-forwarding API: the arguments are forwarded directly
     * into @c candidate_entry_t's constructor (with a trailing @c false for
     * the explored flag). Works for both entries.
     *
     * @return true if the entry was inserted, false if rejected.
     * @complexity O(log_4 L) for heap operations, but O(1) for rejected entries.
     */
    template <typename... Args>
    __attribute__((always_inline))
    auto try_push(Args&&... args) -> bool {
        static_assert(
            std::is_constructible_v<candidate_entry_t, Args..., bool>,
            "EntryT must accept these args plus a trailing bool is_explored flag");

        candidate_entry_t entry(std::forward<Args>(args)..., /*is_explored=*/false);
        const distance_t entry_dist = entry.get_distance();

        // Fast rejection: O(1) check before expensive heap operations
        if (_top_candidates.size() >= _capacity && entry_dist >= _lower_bound) {
            return false;
        }

        _unexplored_set.push(entry);
        _top_candidates.push(entry);

        if (_top_candidates.size() > _capacity) {
            _top_candidates.pop();
        }

        _update_lower_bound();
        return true;
    }

    /**
     * @brief Retrieve and return the FULL best-unexplored entry.
     */
    __attribute__((always_inline))
    auto pop_best_unexplored_entry() -> candidate_entry_t {
        if (_unexplored_set.empty()) {
            return invalid_candidate_entry;
        }
        candidate_entry_t entry = _unexplored_set.top();
        _unexplored_set.pop();
        return entry;
    }

    /**
     * @brief Retrieve the best unexplored candidate.
     *
     * Legacy 2-element pair overload kept for backward compatibility.
     *
     * @return Pair of (vertex_id, distance) for the best unexplored candidate,
     *         or (invalid_vertex_id, max_distance) if none found.
     * @complexity O(log_4 L) for heap extraction.
     */
    __attribute__((always_inline))
    auto pop_best_unexplored() -> std::pair<vertex_id_t, distance_t> {
        const candidate_entry_t entry = pop_best_unexplored_entry();
        if (entry.is_invalid()) {
            return {invalid_vertex_id, max_distance};
        }
        return {entry.get_vid(), entry.get_distance()};
    }

    /**
     * @brief Check if the search should terminate early.
     *
     * Early Termination Condition (from hnswlib):
     * If the closest unexplored candidate is farther than the worst candidate in top_candidates,
     * and top_candidates is full, then no better candidates can be found.
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
        return _unexplored_set.top().get_distance() > _lower_bound;
    }

    /**
     * @brief Mutable iterator to the beginning of the result-set (top_candidates).
     * @note Iteration order is heap-order (NOT sorted by distance).
     */
    auto begin()       { return _top_candidates.begin(); }
    auto end()         { return _top_candidates.end(); }
    auto begin() const { return _top_candidates.begin(); }
    auto end()   const { return _top_candidates.end(); }

    /**
     * @brief Create an independent deep copy of this queue.
     * @return A new FHCandidateQueue with identical logical state.
     */
    auto clone() const -> FHCandidateQueue {
        FHCandidateQueue copy(_capacity);
        copy._unexplored_set = _unexplored_set;
        copy._top_candidates = _top_candidates;
        copy._lower_bound    = _lower_bound;
        return copy;
    }

    /**
     * @brief Extract the top-k results from the candidate queue.
     * @param k Number of top results to extract.
     * @return knn_results_t of result entries sorted by distance (ascending order).
     * @note This method empties the top_candidates heap.
     */
    auto extract_results(std::size_t k) -> knn_results_t {
        #ifndef NDEBUG
        assert(_top_candidates.size() >= k && "Not enough candidates in queue");
        #endif

        knn_results_t results;
        results.reserve(k);

        // Extract all candidates from max-heap (they come out in descending order)
        while (!_top_candidates.empty()) {
            results.push_back(_top_candidates.top());
            _top_candidates.pop();
        }

        // Reverse to get ascending order (O(n) instead of O(n log n) sort)
        std::reverse(results.begin(), results.end());
        results.resize(k);

        return results;
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
     */
    __attribute__((always_inline))
    auto _update_lower_bound() -> void {
        _lower_bound = (!_top_candidates.empty()) ?
            _top_candidates.top().get_distance() : max_distance;
    }

};  // class FHCandidateQueue

} // namespace cpu
} // namespace artea
