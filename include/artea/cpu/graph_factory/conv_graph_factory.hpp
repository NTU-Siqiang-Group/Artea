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
 * @FilePath: /Artea/include/artea/cpu/graph_factory/conv_graph_factory.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-02-11 09:56:36
 * @Date: 2025-11-15 20:36:29
 * @Description:
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>

namespace artea {
namespace cpu {

template <typename GraphFactoryTraitsT>
class ConvGraphFactory :
    public GraphFactoryTraitsT::template flat_graph_factory_t<ConvGraphFactory<GraphFactoryTraitsT>>
{

    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vec_ele_t = typename GraphFactoryTraitsT::vec_ele_t;
    using iter_t = typename GraphFactoryTraitsT::iter_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using flat_graph_t = typename GraphFactoryTraitsT::flat_graph_t;
    using vector_dataset_t = typename GraphFactoryTraitsT::vector_dataset_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    // Using propagate_engine_t with no selective scheduling currently.
    using random_eg_t = typename GraphFactoryTraitsT::random_eg_t;
    using propagate_engine_t = typename GraphFactoryTraitsT::template propagate_engine_t<false>;
    using triangle_updater_t = typename GraphFactoryTraitsT::triangle_updater_t;
    using reverse_updater_t = typename GraphFactoryTraitsT::reverse_updater_t;

public:
    ConvGraphFactory() {}

    /** @brief construct a new convergent graph */
    auto construct_graph_impl(
        const vector_dataset_t& dataset,
        const vertex_num_t max_nbr_size,
        const vertex_num_t reserved_nbr_size,
        const ratio_t scale_coeffs,
        const ratio_t shifted_coeffs,
        const iter_t num_outer_iters,   // recommend param: 4
        const iter_t num_inner_iters    // recommend param: 14
    ) -> flat_graph_t {
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(dataset.get_num_base_vecs());
        flat_graph_t flat_graph(
            /* vecs_data =          */ dataset.get_base_vecs(),
            /* num_vertices =       */ num_vertices,
            /* max_nbr_size =       */ max_nbr_size,
            /* reserved_nbr_size =  */ reserved_nbr_size
        );
        dist_func_t dist_func(dataset.get_vec_dim());

        // generate random edges first
        random_eg_t random_eg(dist_func);
        random_eg.generate(flat_graph, /* init_nbr_size = */ max_nbr_size);
        propagate_engine_t propagate_engine(num_vertices, dist_func);
        propagate_engine.set_graph(flat_graph);
        // Create triangle updater and reverse updater
        auto triangle_updater = propagate_engine.template make_updater<triangle_updater_t>(scale_coeffs, shifted_coeffs);
        auto reverse_updater = propagate_engine.template make_updater<reverse_updater_t>();
        // run propagation engine to refine the graph
        for (iter_t outer_iter = 0; outer_iter < num_outer_iters; ++outer_iter) {
            propagate_engine.run(num_inner_iters, triangle_updater);
            propagate_engine.run(1, reverse_updater);
        }

        return flat_graph;
    }

};  // class ConvGraphFactory


}   // namespace cpu
}   // namespace artea
