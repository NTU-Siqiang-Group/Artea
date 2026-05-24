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

#include <concepts>
#include <cstddef>
#include <utility>

namespace artea {
namespace cpu {

/**
 * @brief Concept defining the required interface for candidate queues
 *        used in proximity graph routing.
 *
 * All candidate queue implementations (LinearCandidateQueue, StdCandidateQueue,
 * FHCandidateQueue) must satisfy this concept to be used as a template parameter
 * in SingleLayerRouter / HierarchicalGraphRouter.
 *
 * Design Philosophy:
 * The queue is parameterized on an entry type (@c candidate_entry_t) that
 * satisfies the @c CandidateEntry concept. @c try_push forwards its arguments
 * into the entry constructor — it does NOT have a fixed signature — so this
 * concept does not check it. Callers that want the legacy 2-arg interface
 * (vertex_id, distance) should verify separately that their entry type
 * accepts that constructor shape.
 *
 * Semantic Requirements:
 *
 * 1. Construction:
 *    - Must be constructible with a capacity (std::size_t) parameter.
 *
 * 2. Query Operations (const):
 *    - empty(), size(), get_result_size(), get_unexplored_size(), capacity()
 *
 * 3. Mutating Operations:
 *    - clear(): Removes all candidates and resets internal state
 *    - pop_best_unexplored(): legacy (vid, dist) pair overload
 *    - pop_best_unexplored_entry(): returns the full entry (for inbr callers)
 *    - should_terminate(): early-termination check
 *
 * 4. Result Extraction:
 *    - extract_results(k): Returns top-k result entries (knn_results_t)
 *
 * 5. Iteration:
 *    - begin(), end(): Provide mutable iterator access to the result-set
 *      entries. Traversal order is implementation-defined and NOT guaranteed
 *      to be sorted. Useful for in-place modification of entries (e.g.,
 *      updating distances or flags) without extracting and re-inserting.
 *
 * 6. Deep Copy:
 *    - clone(): Returns a fully independent copy of the queue with
 *      identical logical state. The original queue is not modified.
 *
 * 7. Capacity Adjustment:
 *    - set_capacity(new_capacity): Adjust the queue's result-set capacity.
 *      Upward resize leaves existing contents untouched; downward resize
 *      pops worst-distance entries from the result-set until size <= new
 *      capacity. The unexplored-set is not affected. Used by
 *      IndexFactory to reuse descent-phase queue snapshots with a larger
 *      capacity during select-neighbors search.
 *
 * @tparam CandidateQueueImpl The candidate queue type to check.
 */
template <typename CandidateQueueImpl>
concept CandidateQueue =
std::constructible_from<CandidateQueueImpl, std::size_t> && requires(
    CandidateQueueImpl queue,
    const CandidateQueueImpl const_queue
) {

    // Type aliases. dist_func_t is per-method template arg now (DistFuncT),
    // not a queue-class member alias — removed from the concept too.
    typename CandidateQueueImpl::vertex_id_t;
    typename CandidateQueueImpl::distance_t;
    typename CandidateQueueImpl::candidate_entry_t;
    typename CandidateQueueImpl::knn_results_t;
    typename CandidateQueueImpl::random_seq_t;
    typename CandidateQueueImpl::vec_ele_t;
    typename CandidateQueueImpl::vector_array_t;
    typename CandidateQueueImpl::visited_table_t;

    // Query operations (const)
    { const_queue.empty() }              -> std::convertible_to<bool>;
    { const_queue.size() }               -> std::convertible_to<std::size_t>;
    { const_queue.get_result_size() }    -> std::convertible_to<std::size_t>;
    { const_queue.get_unexplored_size()} -> std::convertible_to<std::size_t>;
    { const_queue.capacity() }           -> std::convertible_to<std::size_t>;

    // Mutating operations. Note: try_push is variadic and cannot be checked
    // by this concept — it is enforced via a static_assert inside each queue
    // implementation at the call site.
    { queue.clear() }               -> std::same_as<void>;
    { queue.pop_best_unexplored() } -> std::same_as<std::pair<
        typename CandidateQueueImpl::vertex_id_t,
        typename CandidateQueueImpl::distance_t
    >>;
    { queue.pop_best_unexplored_entry() } -> std::same_as<
        typename CandidateQueueImpl::candidate_entry_t
    >;
    { queue.extract_results(std::declval<std::size_t>()) } -> std::same_as<
        typename CandidateQueueImpl::knn_results_t
    >;
    { queue.should_terminate() } -> std::convertible_to<bool>;

    // Iteration: provides mutable access to result-set entries.
    // Iteration order is implementation-defined (not necessarily sorted).
    { queue.begin() };
    { queue.end() };

    // Deep copy: returns an independent copy of the queue with identical state.
    { queue.clone() } -> std::same_as<CandidateQueueImpl>;

    // Capacity adjustment: resize the result-set capacity; upward resize
    // preserves contents; downward resize drops worst-distance entries.
    { queue.set_capacity(std::declval<std::size_t>()) } -> std::same_as<void>;
};

}   // namespace cpu
}   // namespace artea
