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

namespace artea {
namespace cpu {

/**
 * @brief Concept defining the required interface for candidate queues
 *        used in proximity graph routing.
 *
 * All candidate queue implementations (LinearCandidateQueue, StdCandidateQueue,
 * FHCandidateQueue) must satisfy this concept to be used as a template parameter
 * in MonolayerGraphRouter.
 *
 * Design Philosophy:
 * This concept is completely decoupled from candidate_entry_t. All operations
 * work directly with vertex_id_t and distance_t pairs, making the interface
 * simpler and more flexible.
 *
 * @tparam CandidateQueueImpl The candidate queue type to check.
 */
template <typename CandidateQueueImpl>
concept CandidateQueue =
std::constructible_from<CandidateQueueImpl, std::size_t> && requires(
    CandidateQueueImpl queue,
    const CandidateQueueImpl const_queue
) {

    // Type aliases
    typename CandidateQueueImpl::vertex_id_t;
    typename CandidateQueueImpl::distance_t;
    typename CandidateQueueImpl::random_seq_t;
    typename CandidateQueueImpl::dist_func_t;
    typename CandidateQueueImpl::vec_ele_t;
    typename CandidateQueueImpl::vector_array_t;

    // Query operations (const)
    { const_queue.empty() }              -> std::convertible_to<bool>;
    { const_queue.size() }               -> std::convertible_to<std::size_t>;
    { const_queue.get_result_size() }    -> std::convertible_to<std::size_t>;
    { const_queue.get_unexplored_size()} -> std::convertible_to<std::size_t>;
    { const_queue.capacity() }           -> std::convertible_to<std::size_t>;

    // Mutating operations
    { queue.clear() }               -> std::same_as<void>;
    { queue.try_push(
        std::declval<typename CandidateQueueImpl::vertex_id_t>(),
        std::declval<typename CandidateQueueImpl::distance_t>()
    ) } -> std::convertible_to<bool>;
    { queue.pop_best_unexplored() } -> std::same_as<std::pair<
        typename CandidateQueueImpl::vertex_id_t,
        typename CandidateQueueImpl::distance_t
    >>;
    { queue.extract_results(std::declval<std::size_t>()) } -> std::same_as<std::vector<std::pair<
        typename CandidateQueueImpl::vertex_id_t,
        typename CandidateQueueImpl::distance_t
    >>>;
    { queue.extract_result_ids(std::declval<std::size_t>()) } -> std::same_as<std::vector<
        typename CandidateQueueImpl::vertex_id_t
    >>;
    { queue.should_terminate() } -> std::convertible_to<bool>;

    // Initialization operations
    { queue.random_initialize(
        std::declval<typename CandidateQueueImpl::random_seq_t&>(),
        std::declval<const typename CandidateQueueImpl::dist_func_t&>(),
        std::declval<const typename CandidateQueueImpl::vec_ele_t*>(),
        std::declval<const typename CandidateQueueImpl::vector_array_t&>()
    ) } -> std::same_as<void>;
};

}   // namespace cpu
}   // namespace artea
