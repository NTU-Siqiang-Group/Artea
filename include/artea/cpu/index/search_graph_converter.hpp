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
 * @FilePath: /Artea/include/artea/cpu/index/search_graph_converter.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-03-13
 * @Description: Unified converter for transforming graphs to search graphs.
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename IndexTraitsT>
class SearchGraphConverter {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using layer_id_t = typename IndexTraitsT::layer_id_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using index_t = typename IndexTraitsT::conv_graph::index_t;
    using flat_search_graph_t = typename IndexTraitsT::flat_search_graph_t;
    using hierarchical_search_graph_t = typename IndexTraitsT::hierarchical_search_graph_t;
    using nbr_arr_checker_t = typename IndexTraitsT::nbr_arr_checker_t;

public:
    /**
     * @brief Convert a FlatGraph to FlatSearchGraph in parallel.
     * @param flat_graph The source flat graph to convert from.
     * @param extracted_nbr_size Fixed number of neighbors per vertex in the flat search graph.
     * @return A new FlatSearchGraph instance.
     */
    template <typename FlatGraphT>
    static auto from_flat_graph(
        const FlatGraphT& flat_graph,
        const vertex_num_t extracted_nbr_size
    ) -> flat_search_graph_t {
        const vertex_num_t max_nbr_size = flat_graph.layer_config().max_nbr_size();

        // Validate extracted_nbr_size does not exceed max_nbr_size
        if (extracted_nbr_size > max_nbr_size) {
            ARTEA_ERROR(fmt::format(
                "extracted_nbr_size ({}) cannot exceed max_nbr_size ({})",
                extracted_nbr_size, max_nbr_size
            ));
        }

        const vertex_num_t num_vertices = flat_graph.get_num_vertices();
        const auto& vecs_data = flat_graph.get_vecs_data();
        const auto& nbrs_arr = flat_graph.get_nbrs_arr();

        // Create the flat search graph
        flat_search_graph_t flat_search_graph(vecs_data, extracted_nbr_size);

        // Copy neighbors from FlatGraph to FlatSearchGraph in parallel
        auto& csr_nbrs = flat_search_graph.get_csr_nbrs();
        const vertex_id_t invalid_id = IndexTraitsT::invalid_vertex_id;

        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    const auto& nbrs = nbrs_arr[vid];
                    vertex_id_t* dst_nbrs = &csr_nbrs[static_cast<size_t>(vid) * extracted_nbr_size];
                    // Copy up to extracted_nbr_size neighbors
                    const vertex_num_t copy_count = std::min(static_cast<vertex_num_t>(nbrs.size()), extracted_nbr_size);
                    for (vertex_num_t i = 0; i < extracted_nbr_size; ++i) {
                        dst_nbrs[i] = (i < copy_count) ? nbrs[i].get_id() : invalid_id;
                    }

                    #ifndef NDEBUG
                    if (!nbr_arr_checker_t::invalid_id_suffix_check(dst_nbrs, extracted_nbr_size)) {
                        ARTEA_ERROR("Error: Invalid search graph row suffix layout.");
                    }
                    #endif
                }
            }
        );

        return flat_search_graph;
    }

    /**
     * @brief Convert HierarchicalGraph to HierarchicalSearchGraph.
     * @param hierarchical_graph The source hierarchical graph to convert from.
     *        When passed as an rvalue (std::move), inter-layer links are moved
     *        instead of copied.
     * @param bl_extracted_nbr_size Fixed number of neighbors for bottom layer.
     * @param ul_extracted_nbr_size Fixed number of neighbors for upper layers.
     * @return A new HierarchicalSearchGraph instance.
     */
    template <typename HierGraphT>
    static auto from_hierarchical_graph(
        HierGraphT&& hierarchical_graph,
        const vertex_num_t bl_extracted_nbr_size,
        const vertex_num_t ul_extracted_nbr_size
    ) -> hierarchical_search_graph_t {
        const auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();
        const auto num_layers = hierarchical_graph.get_num_layers();

        hierarchical_search_graph_t hier_search_graph(
            hier_vecs_manager,
            bl_extracted_nbr_size,
            ul_extracted_nbr_size
        );

        hier_search_graph.resize(num_layers);

        // Convert bottom layer (layer_id = 0)
        const auto& bottom_flat_graph = hierarchical_graph.get_layer_graph(0);
        auto bottom_search_graph = from_flat_graph(
            bottom_flat_graph,
            bl_extracted_nbr_size
        );
        hier_search_graph.set_layer_graph(0, std::move(bottom_search_graph));

        // Convert upper layers (layer_id > 0)
        for (layer_id_t layer_id = 1; layer_id < num_layers; ++layer_id) {
            const auto& upper_flat_graph = hierarchical_graph.get_layer_graph(layer_id);
            auto upper_search_graph = from_flat_graph(
                upper_flat_graph,
                ul_extracted_nbr_size
            );
            hier_search_graph.set_layer_graph(layer_id, std::move(upper_search_graph));
        }

        // Transfer inter-layer links (move when rvalue, copy when lvalue) and entry point
        hier_search_graph.get_inter_layer_links() =
            std::move(std::forward<HierGraphT>(hierarchical_graph).get_inter_layer_links());
        hier_search_graph.set_entry_point(hierarchical_graph.get_entry_point());

        return hier_search_graph;
    }

    /**
     * @brief Load FlatGraph from file and convert to FlatSearchGraph.
     * @param file_path Path to the flat graph index file.
     * @param extracted_nbr_size Fixed number of neighbors per vertex in the flat search graph.
     * @param vecs_data Reference to the vector data.
     * @return A new FlatSearchGraph instance.
     */
    static auto from_index_file(
        const std::string& file_path,
        const vertex_num_t extracted_nbr_size,
        const vector_array_t& vecs_data
    ) -> flat_search_graph_t {
        // Load FlatGraph from file
        index_t flat_graph = index_t::restore(file_path, vecs_data);

        // Convert to FlatSearchGraph
        return from_flat_graph(flat_graph, extracted_nbr_size);
    }

};  // class SearchGraphConverter

}   // namespace cpu
}   // namespace artea