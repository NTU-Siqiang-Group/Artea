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

#include <functional>
#include <queue>
#include <vector>
#include <limits>
#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>

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
 * @tparam EntryT Candidate-entry type carried by the queue. Defaults to the
 *         router traits' candidate entry, preserving existing behavior.
 */
template <typename RouterTraitsT,
          typename EntryT = typename RouterTraitsT::candidate_entry_t>
class StdCandidateQueue {

public:
    // Type aliases for clarity and convenience
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using candidate_entry_t = EntryT;
    /** @brief knn_results_t is now a queue-local type that tracks @c EntryT. */
    using knn_results_t = std::vector<candidate_entry_t>;
    using random_seq_t = typename RouterTraitsT::random_seq_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using visited_table_t = typename RouterTraitsT::visited_table_t;

    /** @brief Maximum possible distance value (used for threshold initialization). */
    static constexpr distance_t max_distance = RouterTraitsT::max_distance;

    /** @brief Invalid vertex ID constant. */
    static constexpr vertex_id_t invalid_vertex_id = RouterTraitsT::invalid_vertex_id;

    /** @brief Sentinel value representing an invalid/non-existent candidate. */
    static constexpr candidate_entry_t invalid_candidate_entry =
        candidate_entry_t::make_invalid_entry();

    /**
     * @brief Thin wrapper around std::priority_queue that exposes the
     *        underlying std::vector via begin()/end() iterators.
     *
     * std::priority_queue keeps its container as a protected member @c c.
     * This subclass simply surfaces it so callers can iterate over entries
     * (in heap-order, NOT sorted) without popping them.
     */
    template <typename Compare>
    struct InternalQueue : std::priority_queue<candidate_entry_t,
                                              std::vector<candidate_entry_t>,
                                              Compare> {
        using base_t = std::priority_queue<candidate_entry_t,
                                           std::vector<candidate_entry_t>,
                                           Compare>;
        using base_t::base_t;   // inherit constructors

        auto begin()       { return this->c.begin(); }
        auto end()         { return this->c.end(); }
        auto begin() const { return this->c.begin(); }
        auto end()   const { return this->c.end(); }
    };

    /** @brief Min-heap type for unexplored set (closest candidates at top).
     *  std::greater makes the smallest entry bubble to the top. */
    using min_heap_t = InternalQueue<std::greater<candidate_entry_t>>;

    /** @brief Max-heap type for top candidates (worst of best at top).
     *  std::less (default) makes the largest entry bubble to the top. */
    using max_heap_t = InternalQueue<std::less<candidate_entry_t>>;

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
        // (vertex_id, distance).
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

        _unexplored_set = min_heap_t();
        _top_candidates = max_heap_t();
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
     *   - Upward resize (@p new_capacity >= current capacity): existing
     *     contents are preserved; the fast-rejection threshold relaxes
     *     to allow more entries before trimming.
     *   - Downward resize (@p new_capacity < current capacity): the
     *     worst-distance entries are popped from the top-candidates
     *     max-heap until its size is <= @p new_capacity. The
     *     unexplored-set is not touched (still-to-explore seeds stay in
     *     place even if their distance now exceeds the result-set
     *     threshold).
     *
     * Used by the stacked_rgraph::IndexFactory to reuse descent-phase
     * queue snapshots with a larger capacity during select-neighbors
     * search.
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
     * @brief Smallest distance currently in the result set.
     *        Returns @c max_distance when the result set is empty.
     *
     * @c _top_candidates is a max-heap (top = worst), so the best entry
     * lives somewhere in the underlying vector. The scan is O(L) with
     * L == capacity; for beam-search capacities this is tens of entries.
     */
    auto best_result_distance() const -> distance_t {
        distance_t best = max_distance;
        for (const auto& cand : _top_candidates) {
            if (cand.get_distance() < best) best = cand.get_distance();
        }
        return best;
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
     * Variadic-perfect-forwarding API: the arguments are forwarded directly
     * into the entry type's constructor (plus a trailing @c false for the
     * explored flag).
     *
     * Insertion Strategy (following hnswlib approach):
     * 1. Build the entry via perfect-forwarding.
     * 2. Fast Rejection: If top_candidates is full and new distance >= lower_bound, reject (O(1)).
     * 3. Add to Unexplored Set: Insert into min-heap for future exploration (O(log L)).
     * 4. Add to Top Candidates: Insert into max-heap to maintain best L candidates (O(log L)).
     * 5. Trim: If top_candidates exceeds capacity, remove worst candidate (O(log L)).
     * 6. Update Lower Bound: Set to the worst distance in top_candidates (O(1)).
     *
     * @return true if the entry was inserted, false if rejected.
     * @complexity O(log L) for heap operations, but O(1) for rejected entries.
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
     * @brief Retrieve and return the FULL best-unexplored entry.
     *
     * Returns the complete @c candidate_entry_t.
     *
     * @return The entry with smallest distance in the unexplored set, or an
     *         invalid sentinel entry if the unexplored set is empty.
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
     * @brief Re-mark every top-candidate as "unexplored" by rebuilding
     *        @c _unexplored_set from @c _top_candidates.
     *
     * After a beam_search call, every candidate it popped is gone from
     * @c _unexplored_set forever. In a single-layer search that's fine,
     * but the shared-queue hierarchical descent calls beam_search
     * multiple times on the same queue (once per level). Without this
     * reset, candidates that were popped at level H never get their
     * level-(H-1) neighbors expanded in the subsequent beam_search —
     * the queue silently collapses and @c min_dist_per_level[1] reflects
     * only the residual unexplored entries rather than the true L_1 NN.
     */
    auto reset_exploration() -> void {
        _unexplored_set = min_heap_t();
        for (const auto& cand : _top_candidates) {
            _unexplored_set.push(cand);
        }
    }

    /**
     * @brief Relax @c _lower_bound back to @c max_distance.
     *
     * Needed when @c set_capacity() is used to grow a previously-saturated
     * queue (e.g. reusing a descent-phase queue for Step D select with a
     * larger beam). @c set_capacity on upsize does not touch
     * @c _lower_bound — it still equals the OLD worst-of-top. The beam
     * extension would then immediately @c should_terminate because the
     * residual unexplored entries are, by construction, farther than that
     * old bound. Calling this relaxes the bound so the extended beam
     * accepts the newly-within-capacity candidates.
     */
    __attribute__((always_inline))
    auto reset_lower_bound() -> void {
        _lower_bound = max_distance;
    }

    /**
     * @brief Seed this queue from another queue's top_candidates.
     *
     * Clears current state, then pushes each entry in @p src's top
     * candidates into both heaps of @c *this, honoring @c _capacity.
     * Avoids the extract_results → try_push round-trip (two O(N log N)
     * passes) when forking a per-level descent queue into the next
     * level's working queue. Also resets @c _lower_bound to
     * @c max_distance so the forked queue starts with a clean rejection
     * threshold.
     */
    auto seed_from_queue(const StdCandidateQueue& src) -> void {
        _unexplored_set = min_heap_t();
        _top_candidates = max_heap_t();
        _lower_bound = max_distance;
        for (const auto& cand : src._top_candidates) {
            if (_top_candidates.size() >= _capacity &&
                cand.get_distance() >= _lower_bound) {
                continue;
            }
            _unexplored_set.push(cand);
            _top_candidates.push(cand);
            if (_top_candidates.size() > _capacity) {
                _top_candidates.pop();
            }
            _update_lower_bound();
        }
    }

    /**
     * @brief Retrieve and mark the best unexplored candidate.
     *
     * Legacy 2-element pair overload kept for backward compatibility with
     * existing descent-graph routers. Internally implemented in terms of
     * @c pop_best_unexplored_entry().
     *
     * @return Pair of (vertex_id, distance) for the best unexplored candidate,
     *         or (invalid_vertex_id, max_distance) if none found.
     * @complexity O(log L) for heap extraction.
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
     * @return A new StdCandidateQueue with identical logical state.
     */
    auto clone() const -> StdCandidateQueue {
        StdCandidateQueue copy(_capacity);
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
     * @note Called after insertion or when top_candidates changes.
     */
    __attribute__((always_inline))
    auto _update_lower_bound() -> void {
        _lower_bound = (!_top_candidates.empty()) ?
            _top_candidates.top().get_distance() : max_distance;
    }

};  // class StdCandidateQueue

}   // namespace cpu
}   // namespace artea
