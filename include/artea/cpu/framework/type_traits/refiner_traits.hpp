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
template <typename RefinerTraitsT, typename BottomGraphT, typename DerivedClassT> class NeighborUpdater;
template <typename RefinerTraitsT, typename BottomGraphT> class TriangleUpdater;
template <typename RefinerTraitsT, typename BottomGraphT> class PruningUpdater;
template <typename RefinerTraitsT> class HierarchicalPruningUpdater;
template <typename RefinerTraitsT, typename BottomGraphT> class ReverseUpdater;
template <typename RefinerTraitsT, typename BottomGraphT> class RandomUpdater;
template <typename RefinerTraitsT, typename BottomGraphT> class RoutingUpdater;
template <typename RefinerTraitsT, typename BottomGraphT> class TruncateUpdater;
template <typename RefinerTraitsT> class RandomEG;
template <typename RefinerTraitsT, typename BottomGraphT, bool SelectiveSchedule> class PropagateEngine;
template <typename RefinerTraitsT> class IVFPartitions;

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

    template <typename BottomGraphT, typename DerivedClassT>
    using neighbor_updater_t = NeighborUpdater<refiner_traits_t, BottomGraphT, DerivedClassT>;

    /** @brief Triangle updater. */
    template <typename BottomGraphT>
    using triangle_updater_t = TriangleUpdater<refiner_traits_t, BottomGraphT>;

    /** @brief Pruning updater (no log writes). */
    template <typename BottomGraphT>
    using pruning_updater_t = PruningUpdater<refiner_traits_t, BottomGraphT>;

    /** @brief Hierarchical pruning updater — operates on nbr_t, not
     *         bnbr_t, and takes max_nbr_size as a per-call argument.
     *         Consumed by stacked_rgraph::IndexFactory. */
    using hierarchical_pruning_updater_t =
        HierarchicalPruningUpdater<refiner_traits_t>;

    /** @brief Reverse edge updater. */
    template <typename BottomGraphT>
    using reverse_updater_t = ReverseUpdater<refiner_traits_t, BottomGraphT>;

    /** @brief Random neighbor updater. */
    template <typename BottomGraphT>
    using random_updater_t = RandomUpdater<refiner_traits_t, BottomGraphT>;

    /** @brief Routing-based neighbor updater (uses construct-mode router). */
    template <typename BottomGraphT>
    using routing_updater_t = RoutingUpdater<refiner_traits_t, BottomGraphT>;

    /** @brief Truncate updater: trims neighbor arrays to max_nbr_size. */
    template <typename BottomGraphT>
    using truncate_updater_t = TruncateUpdater<refiner_traits_t, BottomGraphT>;

    /** @brief Random edge generator. */
    using random_eg_t = RandomEG<refiner_traits_t>;

    /** @brief Type for propagation engine. */
    template <typename BottomGraphT, bool SelectiveSchedule>
    using propagate_engine_t = PropagateEngine<refiner_traits_t, BottomGraphT, SelectiveSchedule>;

    /** @brief Neighbor array checker. */
    using nbr_arr_checker_t = typename IndexTraitsT::nbr_arr_checker_t;

    /** @brief IVF partitions for partition-based operations. */
    using ivf_partitions_t = IVFPartitions<refiner_traits_t>;

};  // struct RefinerTraits

}   // namespace cpu
}   // namespace artea
