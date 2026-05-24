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
#include <artea/cpu/router/data_structures/visited_table_concept.hpp>

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
// Router classes carry a DistFuncT template parameter (the concrete
// SIMDDistance<...,VecDim> they're instantiated with). Consumers obtain
// this type by calling std::visit on a SIMDDistanceDispatcher's variant.
template <typename RouterTraitsT, typename DistFuncT, typename DerivedClassT> class VectorRouter;
template <typename RouterTraitsT, typename DistFuncT> class BruteforceRouter;
template <typename RouterTraitsT, typename DistFuncT> class SingleLayerRouter;
template <typename RouterTraitsT, typename DistFuncT> class HierarchicalGraphRouter;
template <typename RouterTraitsT, typename DistFuncT> class SLRouterProfiler;
template <typename RouterTraitsT, typename DistFuncT> class HGRouterProfiler;
// Candidate entries, queues, visited-table pool and the stateless
// sample-utils class do NOT hold a dist_func member — they take it (when
// needed) via per-method template parameters. Their class templates stay
// dim-agnostic.
template <typename RouterTraitsT> struct CandidateEntry;
template <typename RouterTraitsT, typename EntryT> class StdCandidateQueue;
template <typename RouterTraitsT, typename EntryT> class LinearCandidateQueue;
template <typename RouterTraitsT, typename EntryT> class FHCandidateQueue;
template <typename RouterTraitsT, typename EntryT> class BoostCandidateQueue;
template <typename RouterTraitsT, VisitedTable VisitedTableT> class VisitedTablePool;
template <typename RouterTraitsT> class CandidateSampleUtils;

/** @brief Traits for routing to queried vectors */
template <typename ComputerTraitsT, typename IndexTraitsT, bool IntraQueryParallel = false>
struct RouterTraits : virtual public ComputerTraitsT, virtual public IndexTraitsT
{
    /** ------ Self Traits ------ **/
    using router_traits_t = RouterTraits<ComputerTraitsT, IndexTraitsT, IntraQueryParallel>;

    // --- Candidate / result entry (unified 8-byte type) ---

    /** @brief Unified candidate entry (vid + explored bit + distance). */
    using candidate_entry_t = CandidateEntry<router_traits_t>;

    /** @brief Result entry is the same unified type. */
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

    /** @brief 4-ary heap type for candidate entries, parameterized by comparator. */
    template <typename Compare>
    using four_ary_heap_t = FourAryHeap<candidate_entry_t, cache_aligned_container_t<candidate_entry_t>, Compare>;

    // Router class template aliases. Every router carries a DistFuncT
    // template param — consumers must supply the concrete SIMDDistance
    // type, obtained by std::visit on a SIMDDistanceDispatcher's variant.
    template <typename DistFuncT, typename DerivedClassT>
    using vector_router_t = VectorRouter<router_traits_t, DistFuncT, DerivedClassT>;

    /** @brief Type for bruteforce router. */
    template <typename DistFuncT>
    using bruteforce_router_t = BruteforceRouter<router_traits_t, DistFuncT>;

    // --- Candidate queues (all use unified candidate_entry_t) ---

    /** @brief Type for standard candidate queue. */
    using std_candidate_queue_t = StdCandidateQueue<router_traits_t, candidate_entry_t>;

    /** @brief Type for linear candidate queue. */
    using linear_candidate_queue_t = LinearCandidateQueue<router_traits_t, candidate_entry_t>;

    /** @brief Type for four-ary heap candidate queue. */
    using fh_candidate_queue_t = FHCandidateQueue<router_traits_t, candidate_entry_t>;

    /** @brief Type for boost d-ary heap candidate queue. */
    using boost_candidate_queue_t = BoostCandidateQueue<router_traits_t, candidate_entry_t>;

    /** @brief Default candidate queue type. */
    using candidate_queue_t = typename router_traits_t::linear_candidate_queue_t;

    /** @brief Default visited table type (used by candidate queues and routers).
     *  VersionTagTable's clear() is O(1) (version bump), so callers can
     *  clear freely between hierarchical layers without a memset cost. */
    using visited_table_t = typename router_traits_t::version_tag_table_t;

    /** @brief Type for visited table pool. */
    using visited_table_pool_t = VisitedTablePool<router_traits_t, visited_table_t>;

    /** @brief Stateless apex-bucket samplers shared by all routers. */
    using candidate_sample_utils_t = CandidateSampleUtils<router_traits_t>;

    /** @brief Unified single-level router. Graph-storage-agnostic at the
     *         class level — the graph type enters as a per-method
     *         template parameter. */
    template <typename DistFuncT>
    using single_layer_router_t = cpu::SingleLayerRouter<router_traits_t, DistFuncT>;

    /** @brief Unified multi-level router. Same graph-agnostic design. */
    template <typename DistFuncT>
    using hierarchical_graph_router_t = cpu::HierarchicalGraphRouter<router_traits_t, DistFuncT>;

    /** @brief Standalone per-hop 1-NN profiler for a single flat layer. */
    template <typename DistFuncT>
    using sl_router_profiler_t = cpu::SLRouterProfiler<router_traits_t, DistFuncT>;

    /** @brief Standalone per-hop 1-NN profiler for the full hierarchy. */
    template <typename DistFuncT>
    using hg_router_profiler_t = cpu::HGRouterProfiler<router_traits_t, DistFuncT>;

    /** @brief Pass-through of the per-mode graph types from IndexTraits;
     *         router types are no longer per-mode (lifted to top-level). */
    struct compact : IndexTraitsT::compact {};
    struct dynamic : IndexTraitsT::dynamic {};

    /** @brief Indicates whether to enable intra-query parallelism. */
    static constexpr bool intra_query_parallel = IntraQueryParallel;

};  // struct RouterTraits

}   // namespace cpu
}   // namespace artea
