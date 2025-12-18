// Copyright 2025 Weitang Ye
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
 * @FilePath: /Artea/include/artea/cpu/framework/template_context.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <vector>
#include <utility>
#include <mutex>

#include <tbb/spin_mutex.h>

#include <artea/cpu/containers/locked_buffer.hpp>
#include <artea/cpu/containers/tbb_buffer.hpp>
#include <artea/common/definitions.hpp>

namespace artea {
namespace cpu {

/** ------ Forward Declarations ------ **/

template <typename type_context_t> class RNGUpdater;
template <typename type_context_t> class PropagateEngine;
template <typename type_context_t> class IndexGraph;
template <typename type_context_t> class ClusterRouter;
template <typename type_context_t> class BruteforceRouter;
template <typename type_context_t> class ProximityGraphRouter;
template <typename type_context_t> class ClusterEvaluator;
template <typename type_context_t> class StreamClustering;
template <typename type_context_t> struct PartitionMetrics;


/* ------ Router Definition ------ */

template <router_policy_t router_policy, typename type_context_t>
struct RouterImplSelector;

template <typename type_context_t>
struct RouterImplSelector<router_policy_t::BRUTEFORCE_ROUTER, type_context_t> {
    using type = BruteforceRouter<type_context_t>;
};

template <typename type_context_t>
struct RouterImplSelector<router_policy_t::PROXIMITY_GRAPH_ROUTER, type_context_t> {
    using type = ProximityGraphRouter<type_context_t>;
};

template <
    typename VertexNumT,
    typename VecEleT,
    buffer_policy_t BufferPolicy,
    router_policy_t RouterPolicy,
    std::size_t BufCapacity
>
struct StrategicTraits {

    using this_t = StrategicTraits<VertexNumT, VecEleT, BufferPolicy, RouterPolicy, BufCapacity>;

    /** @brief Type definitions for convenience. */

    /** ------ Complex Type ------ **/

    /** @brief Type for RNG-based neighbor updaters. */
    using rng_updater_t = RNGUpdater<this_t>;

    /** @brief Type for the propagation engine. */
    using propagate_engine_t = PropagateEngine<this_t>;

    /** @brief Type for the index graph. */
    using index_graph_t = IndexGraph<this_t>;

    /** @brief Type for the cluster router. */
    using cluster_router_t = ClusterRouter<this_t>;

    /** @brief Type for the specific router implementation. */
    using router_impl_t = typename RouterImplSelector<RouterPolicy, this_t>::type;

    /** @brief Whether to enable intra-query parallelism in cluster routing. */
    static constexpr bool intra_query_parallel = false;

    using cluster_evaluator_t = ClusterEvaluator<this_t>;

    using stream_clustering_t = StreamClustering<this_t>;

    using partition_metrics_t = PartitionMetrics<this_t>;

};  //  struct StrategicTraits

}   // namespace cpu
}   // namespace artea