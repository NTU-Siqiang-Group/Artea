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
 * @FilePath: /Artea/include/artea/cpu/index/compactor/descent_graph_compactor.hpp
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
class DescentGraphCompactor {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using layer_id_t = typename IndexTraitsT::layer_id_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using index_t = typename IndexTraitsT::conv_graph::index_t;
    using compact = typename IndexTraitsT::compact;
    using nbr_arr_checker_t = typename IndexTraitsT::nbr_arr_checker_t;

public:
    /**
     * @brief Convert a DescentGraph to CompactDescentGraph in parallel.
     * @param descent_graph The source descent graph to convert from.
     * @param extracted_nbr_size Fixed number of neighbors per vertex in the flat search graph.
     * @return A new CompactDescentGraph instance.
     */
    template <typename DescentGraphT>
    static auto from_descent_graph(
        const DescentGraphT& descent_graph,
        const vertex_num_t extracted_nbr_size
    ) -> typename compact::descent_graph_t {
        const vertex_num_t max_nbr_size = descent_graph.layer_config().max_nbr_size();

        // Validate extracted_nbr_size does not exceed max_nbr_size
        if (extracted_nbr_size > max_nbr_size) {
            ARTEA_ERROR(fmt::format(
                "extracted_nbr_size ({}) cannot exceed max_nbr_size ({})",
                extracted_nbr_size, max_nbr_size
            ));
        }

        const vertex_num_t num_vertices = descent_graph.get_num_vertices();
        const auto& vecs_data = descent_graph.get_vecs_data();
        const auto& nbrs_arr = descent_graph.get_nbrs_arr();

        // Create the flat search graph
        typename compact::descent_graph_t compact_descent_graph(vecs_data, extracted_nbr_size);

        // Copy neighbors from DescentGraph to CompactDescentGraph in parallel
        auto& csr_nbrs = compact_descent_graph.get_csr_nbrs();
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

        return compact_descent_graph;
    }

    /**
     * @brief Load DescentGraph from file and convert to CompactDescentGraph.
     * @param file_path Path to the descent graph index file.
     * @param extracted_nbr_size Fixed number of neighbors per vertex in the flat search graph.
     * @param vecs_data Reference to the vector data.
     * @return A new CompactDescentGraph instance.
     */
    static auto from_index_file(
        const std::string& file_path,
        const vertex_num_t extracted_nbr_size,
        const vector_array_t& vecs_data
    ) -> typename compact::descent_graph_t {
        // Load DescentGraph from file
        index_t descent_graph = index_t::restore(file_path, vecs_data);

        // Convert to CompactDescentGraph
        return from_descent_graph(descent_graph, extracted_nbr_size);
    }

};  // class DescentGraphCompactor

}   // namespace cpu
}   // namespace artea