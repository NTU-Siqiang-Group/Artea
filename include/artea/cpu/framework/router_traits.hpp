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
 * @FilePath: /Artea/include/artea/cpu/framework/router_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <artea/cpu/router/candidate_queue_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
template <typename RouterTraitsT, typename DerivedClassT> class VectorRouter;
template <typename RouterTraitsT> class BruteforceRouter;
template <typename RouterTraitsT, CandidateQueue CandidateQueueImpl, VisitedTable VisitedTableImpl> class ProximityGraphRouter;
template <typename RouterTraitsT> struct CandidateEntry;
template <typename RouterTraitsT> struct CandidateEntryComparator;
template <typename RouterTraitsT> struct StatefulCandidateEntry;
template <typename RouterTraitsT> struct StatefulCandidateEntryComparator;
template <typename RouterTraitsT> class StdCandidateQueue;
template <typename RouterTraitsT> class LinearCandidateQueue;
template <typename RouterTraitsT> class FHCandidateQueue;
template <typename RouterTraitsT, VisitedTable VisitedTableT> class VisitedTablePool;

/** @brief Traits for routing to queried vectors */
template <typename ComputerTraitsT, typename IndexTraitsT, bool IntraQueryParallel = false>
struct RouterTraits : public ComputerTraitsT, public IndexTraitsT
{
    /** ------ Self Traits ------ **/
    using router_traits_t = RouterTraits<ComputerTraitsT, IndexTraitsT, IntraQueryParallel>;

    /** @brief Type for candidate entry. */
    using candidate_entry_t = CandidateEntry<router_traits_t>;

    /** @brief Type for candidate vector. */
    using candidate_vec_t = std::vector<candidate_entry_t>;

    /** @brief Invalid candidate entry constant (max distance, for min-heap sentinel). */
    static constexpr candidate_entry_t invalid_candidate_entry =
        candidate_entry_t::make_invalid_entry();

    /** @brief Min candidate entry constant (min distance, for max-heap sentinel). */
    static constexpr candidate_entry_t min_candidate_entry =
        candidate_entry_t::make_min_entry();

    /** @brief Type for candidate entry comparator. */
    using entry_comp_t = CandidateEntryComparator<router_traits_t>;

    /** @brief Type for stateful candidate entry (with embedded explored status). */
    using stateful_candidate_entry_t = StatefulCandidateEntry<router_traits_t>;

    /** @brief Type for stateful candidate entry comparator. */
    using stateful_entry_comp_t = StatefulCandidateEntryComparator<router_traits_t>;

    /** @brief 4-ary heap type for candidate entries, parameterized by comparator. */
    template <typename Compare>
    using four_ary_heap_t = FourAryHeap<candidate_entry_t, cache_aligned_container_t<candidate_entry_t>, Compare>;

    template <typename DerivedClassT>
    using vector_router_t = VectorRouter<router_traits_t, DerivedClassT>;

    /** @brief Type for bruteforce router. */
    using bruteforce_router_t = BruteforceRouter<router_traits_t>;

    /** @brief Type for standard candidate queue. */
    using std_candidate_queue_t = StdCandidateQueue<router_traits_t>;

    /** @brief Type for linear candidate queue */
    using linear_candidate_queue_t = LinearCandidateQueue<router_traits_t>;

    /** @brief Type for four-ary heap candidate queue */
    using fh_candidate_queue_t = FHCandidateQueue<router_traits_t>;

    /** @brief Type for visited table pool (default uses version_tag_table_t). */
    template <VisitedTable VisitedTableImpl = typename router_traits_t::version_tag_table_t>
    using visited_table_pool_t = VisitedTablePool<router_traits_t, VisitedTableImpl>;

    /** @brief Type for proximity graph router. */
    template <CandidateQueue CandidateQueueImpl = std_candidate_queue_t, VisitedTable VisitedTableImpl = typename router_traits_t::thread_local_bitmap_t>
    using proximity_graph_router_t = ProximityGraphRouter<router_traits_t, CandidateQueueImpl, VisitedTableImpl>;

    /** @brief Indicates whether to enable intra-query parallelism. */
    static constexpr bool intra_query_parallel = IntraQueryParallel;

};  // struct RouterTraits

}   // namespace cpu
}   // namespace artea
