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
 * @FilePath: /Artea/include/artea/cpu/framework/refiner_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

namespace artea {
namespace cpu {

/**
 * @brief Policy for IVF partitions construction strategy.
 */
enum class IVFConstructPolicyT {
    serial,      ///< Serial construction (single-threaded)
    parallel     ///< Parallel construction (multi-threaded with TBB)
};

// ----- Forward Declaration  ------ //
// Refiners that hold or call dist_func carry a DistFuncT template
// parameter (the concrete SIMDDistance type from the consumer's
// std::visit lambda). IVFPartitions and RefinerUtils are dim-agnostic.
template <typename RefinerTraitsT, typename DistFuncT, typename DerivedClassT> class NeighborUpdater;
template <typename RefinerTraitsT, typename DistFuncT> class TriangleUpdater;
template <typename RefinerTraitsT, typename DistFuncT> class PruningUpdater;
template <typename RefinerTraitsT, typename DistFuncT> class HierarchicalPruningUpdater;
template <typename RefinerTraitsT, typename DistFuncT> class ReverseUpdater;
template <typename RefinerTraitsT, typename DistFuncT> class RandomUpdater;
template <typename RefinerTraitsT, typename DistFuncT> class RoutingUpdater;
template <typename RefinerTraitsT, typename DistFuncT> class TruncateUpdater;
template <typename RefinerTraitsT, typename DistFuncT> class ARCUpdater;
template <typename RefinerTraitsT, typename DistFuncT> class RandomEG;
template <typename RefinerTraitsT, typename DistFuncT> class PropagateEngine;
template <typename RefinerTraitsT> class IVFPartitions;
template <typename RefinerTraitsT> class RefinerUtils;

template <typename ComputerTraitsT, typename BufferTraitsT, typename IndexTraitsT, typename RouterTraitsT>
struct RefinerTraits :
    virtual public ComputerTraitsT,
    virtual public BufferTraitsT,
    virtual public IndexTraitsT,
    virtual public RouterTraitsT
{

    /** ------ Self Traits ------ **/
    using refiner_traits_t = RefinerTraits<ComputerTraitsT, BufferTraitsT, IndexTraitsT, RouterTraitsT>;

    /** @brief IVF construction policy type. */
    using ivf_construct_policy_t = IVFConstructPolicyT;

    // Refiner template aliases. Each refiner that holds or calls dist_func
    // carries a DistFuncT template param. Consumers obtain DistFuncT from
    // their std::visit lambda over a SIMDDistanceDispatcher.
    template <typename DistFuncT, typename DerivedClassT>
    using neighbor_updater_t = NeighborUpdater<refiner_traits_t, DistFuncT, DerivedClassT>;

    /** @brief Triangle updater. */
    template <typename DistFuncT>
    using triangle_updater_t = TriangleUpdater<refiner_traits_t, DistFuncT>;

    /** @brief Pruning updater (no log writes). */
    template <typename DistFuncT>
    using pruning_updater_t = PruningUpdater<refiner_traits_t, DistFuncT>;

    /** @brief Hierarchical pruning updater — operates on nbr_t, not
     *         nbr_t, and takes max_nbr_size as a per-call argument.
     *         Consumed by stacked_rgraph::IndexFactory. */
    template <typename DistFuncT>
    using hierarchical_pruning_updater_t =
        HierarchicalPruningUpdater<refiner_traits_t, DistFuncT>;

    /** @brief Reverse edge updater. */
    template <typename DistFuncT>
    using reverse_updater_t = ReverseUpdater<refiner_traits_t, DistFuncT>;

    /** @brief Random neighbor updater. */
    template <typename DistFuncT>
    using random_updater_t = RandomUpdater<refiner_traits_t, DistFuncT>;

    /** @brief Routing-based neighbor updater (uses construct-mode router). */
    template <typename DistFuncT>
    using routing_updater_t = RoutingUpdater<refiner_traits_t, DistFuncT>;

    /** @brief Truncate updater: trims neighbor arrays to max_nbr_size. */
    template <typename DistFuncT>
    using truncate_updater_t = TruncateUpdater<refiner_traits_t, DistFuncT>;

    /** @brief Arc-radius pruning updater: drops edges longer than the
     *         configured arc_radius. */
    template <typename DistFuncT>
    using arc_updater_t = ARCUpdater<refiner_traits_t, DistFuncT>;

    /** @brief Random edge generator. */
    template <typename DistFuncT>
    using random_eg_t = RandomEG<refiner_traits_t, DistFuncT>;

    /** @brief Type for propagation engine. */
    template <typename DistFuncT>
    using propagate_engine_t = PropagateEngine<refiner_traits_t, DistFuncT>;

    /** @brief Neighbor array checker. */
    using nbr_arr_checker_t = typename IndexTraitsT::nbr_arr_checker_t;

    /** @brief IVF partitions for partition-based operations. */
    using ivf_partitions_t = IVFPartitions<refiner_traits_t>;

    /** @brief Bridge between dynamic::HierarchicalGraph and
     *         dynamic::RefiningGraph (layer fill / writeback helpers). */
    using refiner_utils_t = RefinerUtils<refiner_traits_t>;

};  // struct RefinerTraits

}   // namespace cpu
}   // namespace artea
