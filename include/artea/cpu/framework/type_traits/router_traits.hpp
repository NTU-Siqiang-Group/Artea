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

#include <artea/cpu/router/data_structures/candidate_queue_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
template <typename RouterTraitsT, typename DerivedClassT> class VectorRouter;
template <typename RouterTraitsT> class BruteforceRouter;
namespace compact { template <typename RouterTraitsT> class DescentGraphRouter; }
namespace dynamic { template <typename RouterTraitsT> class DescentGraphRouter; }
namespace dynamic { template <typename RouterTraitsT> class InternalGraphRouter; }
namespace dynamic { template <typename RouterTraitsT> class HierarchicalGraphRouter; }
template <typename RouterTraitsT> struct DnbrCandidateEntry;
template <typename RouterTraitsT> struct DnbrCandidateEntryComparator;
template <typename RouterTraitsT> struct LnbrCandidateEntry;
template <typename RouterTraitsT> struct LnbrCandidateEntryComparator;
template <typename RouterTraitsT, typename EntryT> class StdCandidateQueue;
template <typename RouterTraitsT, typename EntryT> class LinearCandidateQueue;
template <typename RouterTraitsT, typename EntryT> class FHCandidateQueue;
template <typename RouterTraitsT, VisitedTable VisitedTableT> class VisitedTablePool;

/** @brief Traits for routing to queried vectors */
template <typename ComputerTraitsT, typename IndexTraitsT, bool IntraQueryParallel = false>
struct RouterTraits : virtual public ComputerTraitsT, virtual public IndexTraitsT
{
    /** ------ Self Traits ------ **/
    using router_traits_t = RouterTraits<ComputerTraitsT, IndexTraitsT, IntraQueryParallel>;

    // --- Candidate-entry types ---

    /** @brief Dnbr-flavored candidate entry (vertex_id + distance, 8 bytes). */
    using dnbr_candidate_entry_t      = DnbrCandidateEntry<router_traits_t>;
    using dnbr_candidate_entry_comp_t = DnbrCandidateEntryComparator<router_traits_t>;

    /** @brief Lnbr-flavored candidate entry (base_vid + layer_vid + distance). */
    using lnbr_candidate_entry_t      = LnbrCandidateEntry<router_traits_t>;
    using lnbr_candidate_entry_comp_t = LnbrCandidateEntryComparator<router_traits_t>;

    /** @brief Default candidate entry alias (dnbr). Kept for backward compat
     *         with every existing router, test, and benchmark. */
    using candidate_entry_t = dnbr_candidate_entry_t;

    /** @brief Type for result entry (alias for candidate_entry_t). */
    using result_entry_t = candidate_entry_t;

    /** @brief Flat KNN results: num_queries * topk result entries in row-major order. */
    using knn_results_t = std::vector<result_entry_t>;

    /** @brief Type for candidate vector. */
    using candidate_vec_t = std::vector<candidate_entry_t>;

    /** @brief Invalid candidate entry constant (max distance, for min-heap sentinel). */
    static constexpr candidate_entry_t invalid_candidate_entry =
        candidate_entry_t::make_invalid_entry();

    /** @brief Min candidate entry constant (min distance, for max-heap sentinel). */
    static constexpr candidate_entry_t min_candidate_entry =
        candidate_entry_t::make_min_entry();

    /** @brief Type for candidate entry comparator (dnbr). */
    using entry_comp_t = dnbr_candidate_entry_comp_t;

    /** @brief 4-ary heap type for candidate entries, parameterized by comparator. */
    template <typename Compare>
    using four_ary_heap_t = FourAryHeap<candidate_entry_t, cache_aligned_container_t<candidate_entry_t>, Compare>;

    template <typename DerivedClassT>
    using vector_router_t = VectorRouter<router_traits_t, DerivedClassT>;

    /** @brief Type for bruteforce router. */
    using bruteforce_router_t = BruteforceRouter<router_traits_t>;

    // --- Candidate queues (default EntryT = dnbr_candidate_entry_t) ---

    /** @brief Type for standard candidate queue (dnbr-backed by default). */
    using std_candidate_queue_t = StdCandidateQueue<router_traits_t, dnbr_candidate_entry_t>;

    /** @brief Type for linear candidate queue (dnbr-backed by default). */
    using linear_candidate_queue_t = LinearCandidateQueue<router_traits_t, dnbr_candidate_entry_t>;

    /** @brief Type for four-ary heap candidate queue (dnbr-backed by default). */
    using fh_candidate_queue_t = FHCandidateQueue<router_traits_t, dnbr_candidate_entry_t>;

    /** @brief Type for candidate queue */
    using candidate_queue_t = typename router_traits_t::linear_candidate_queue_t;

    /** @brief Lnbr-backed std candidate queue, used by the internal-graph
     *         router family to carry both base_vid and layer_vid through
     *         beam search without a side-channel lookup table. */
    using std_lnbr_candidate_queue_t = StdCandidateQueue<router_traits_t, lnbr_candidate_entry_t>;

    /** @brief Default visited table type (used by candidate queues and routers). */
    using visited_table_t = typename router_traits_t::thread_local_bitmap_t;

    /** @brief Type for visited table pool (default uses thread_local_bitmap_t). */
    using visited_table_pool_t = VisitedTablePool<router_traits_t, visited_table_t>;

    /** @brief Router + graph types grouped by mode. Inherits graph types from IndexTraits. */
    struct compact : IndexTraitsT::compact {
        using descent_graph_router_t = cpu::compact::DescentGraphRouter<router_traits_t>;
    };
    struct dynamic : IndexTraitsT::dynamic {
        using descent_graph_router_t      = cpu::dynamic::DescentGraphRouter<router_traits_t>;
        using internal_graph_router_t     = cpu::dynamic::InternalGraphRouter<router_traits_t>;
        using hierarchical_graph_router_t = cpu::dynamic::HierarchicalGraphRouter<router_traits_t>;
    };

    /** @brief Indicates whether to enable intra-query parallelism. */
    static constexpr bool intra_query_parallel = IntraQueryParallel;

};  // struct RouterTraits

}   // namespace cpu
}   // namespace artea
