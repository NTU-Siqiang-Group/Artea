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
 * @FilePath: /Artea/include/artea/cpu/framework/edge_generator_traits.hpp
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
template <typename EdgeGeneratorTraitsT, typename FlatGraphT, typename DerivedClassT> class NeighborUpdater;
template <typename EdgeGeneratorTraitsT, typename FlatGraphT> class TriangleUpdater;
template <typename EdgeGeneratorTraitsT, typename FlatGraphT> class SilentTriangleUpdater;
template <typename EdgeGeneratorTraitsT, typename FlatGraphT> class ReverseUpdater;
template <typename EdgeGeneratorTraitsT, typename FlatGraphT> class RandomUpdater;
template <typename EdgeGeneratorTraitsT, typename FlatGraphT> class RoutingUpdater;
template <typename EdgeGeneratorTraitsT, typename FlatGraphT> class TruncateUpdater;
template <typename EdgeGeneratorTraitsT> class RandomEG;
template <typename EdgeGeneratorTraitsT, typename FlatGraphT, bool SelectiveSchedule> class PropagateEngine;
template <typename EdgeGeneratorTraitsT> class IVFPartitions;

template <typename ComputerTraitsT, typename BufferTraitsT, typename IndexTraitsT, typename RouterTraitsT>
struct EdgeGeneratorTraits :
    virtual public ComputerTraitsT,
    virtual public BufferTraitsT,
    virtual public IndexTraitsT,
    virtual public RouterTraitsT
{

    /** ------ Self Traits ------ **/
    using edge_generator_traits_t = EdgeGeneratorTraits<ComputerTraitsT, BufferTraitsT, IndexTraitsT, RouterTraitsT>;

    /** @brief IVF construction policy type. */
    using ivf_construct_policy_t = IVFConstructPolicyT;

    template <typename FlatGraphT, typename DerivedClassT>
    using neighbor_updater_t = NeighborUpdater<edge_generator_traits_t, FlatGraphT, DerivedClassT>;

    /** @brief Triangle updater. */
    template <typename FlatGraphT>
    using triangle_updater_t = TriangleUpdater<edge_generator_traits_t, FlatGraphT>;

    /** @brief Silent triangle updater (no log writes). */
    template <typename FlatGraphT>
    using silent_triangle_updater_t = SilentTriangleUpdater<edge_generator_traits_t, FlatGraphT>;

    /** @brief Reverse edge updater. */
    template <typename FlatGraphT>
    using reverse_updater_t = ReverseUpdater<edge_generator_traits_t, FlatGraphT>;

    /** @brief Random neighbor updater. */
    template <typename FlatGraphT>
    using random_updater_t = RandomUpdater<edge_generator_traits_t, FlatGraphT>;

    /** @brief Routing-based neighbor updater (uses construct-mode router). */
    template <typename FlatGraphT>
    using routing_updater_t = RoutingUpdater<edge_generator_traits_t, FlatGraphT>;

    /** @brief Truncate updater: trims neighbor arrays to max_nbr_size. */
    template <typename FlatGraphT>
    using truncate_updater_t = TruncateUpdater<edge_generator_traits_t, FlatGraphT>;

    /** @brief Random edge generator. */
    using random_eg_t = RandomEG<edge_generator_traits_t>;

    /** @brief Type for propagation engine. */
    template <typename FlatGraphT, bool SelectiveSchedule>
    using propagate_engine_t = PropagateEngine<edge_generator_traits_t, FlatGraphT, SelectiveSchedule>;

    /** @brief Neighbor array checker. */
    using nbr_arr_checker_t = typename IndexTraitsT::nbr_arr_checker_t;

    /** @brief IVF partitions for partition-based operations. */
    using ivf_partitions_t = IVFPartitions<edge_generator_traits_t>;

};  // struct EdgeGeneratorTraits

}   // namespace cpu
}   // namespace artea
