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
template <typename RefinerTraitsT, typename DerivedClassT> class NeighborUpdater;
template <typename RefinerTraitsT> class TriangleUpdater;
template <typename RefinerTraitsT> class PruningUpdater;
template <typename RefinerTraitsT> class HierarchicalPruningUpdater;
template <typename RefinerTraitsT> class ReverseUpdater;
template <typename RefinerTraitsT> class RandomUpdater;
template <typename RefinerTraitsT> class RoutingUpdater;
template <typename RefinerTraitsT> class TruncateUpdater;
template <typename RefinerTraitsT> class RandomEG;
template <typename RefinerTraitsT> class PropagateEngine;
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

    template <typename DerivedClassT>
    using neighbor_updater_t = NeighborUpdater<refiner_traits_t, DerivedClassT>;

    /** @brief Triangle updater. */
    using triangle_updater_t = TriangleUpdater<refiner_traits_t>;

    /** @brief Pruning updater (no log writes). */
    using pruning_updater_t = PruningUpdater<refiner_traits_t>;

    /** @brief Hierarchical pruning updater — operates on nbr_t, not
     *         nbr_t, and takes max_nbr_size as a per-call argument.
     *         Consumed by stacked_rgraph::IndexFactory. */
    using hierarchical_pruning_updater_t =
        HierarchicalPruningUpdater<refiner_traits_t>;

    /** @brief Reverse edge updater. */
    using reverse_updater_t = ReverseUpdater<refiner_traits_t>;

    /** @brief Random neighbor updater. */
    using random_updater_t = RandomUpdater<refiner_traits_t>;

    /** @brief Routing-based neighbor updater (uses construct-mode router). */
    using routing_updater_t = RoutingUpdater<refiner_traits_t>;

    /** @brief Truncate updater: trims neighbor arrays to max_nbr_size. */
    using truncate_updater_t = TruncateUpdater<refiner_traits_t>;

    /** @brief Aspect-Ratio-Constrained (ARC) updater — an alias of
     *         @c truncate_updater_t.
     *
     *         Capping a vertex's maximum out-degree is itself an approximate
     *         enforcement of the Aspect Ratio Constraint: bounding the
     *         out-degree forces pruning to retain only the closest neighbors,
     *         which discards exactly the long-range, large-aspect-ratio edges
     *         that an explicit ARC sweep was meant to drop. Truncating to
     *         @c max_nbr_size therefore subsumes ARC. */
    using arc_updater_t = truncate_updater_t;

    /** @brief Random edge generator. */
    using random_eg_t = RandomEG<refiner_traits_t>;

    /** @brief Type for propagation engine. */
    using propagate_engine_t = PropagateEngine<refiner_traits_t>;

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
