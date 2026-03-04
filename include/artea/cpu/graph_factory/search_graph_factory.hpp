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

#pragma once

#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename IndexTraitsT>
class SearchGraphFactory {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using flat_graph_t = typename IndexTraitsT::flat_graph_t;
    using search_graph_t = typename IndexTraitsT::search_graph_t;

public:
    /**
     * @brief Factory method to convert a FlatGraph to SearchGraph in parallel.
     * @param flat_graph The source flat graph to convert from.
     * @param extracted_nbr_size Fixed number of neighbors per vertex in the search graph.
     * @return A new SearchGraph instance.
     */
    static auto from_flat_graph(
        const flat_graph_t& flat_graph,
        const vertex_num_t extracted_nbr_size
    ) -> search_graph_t {
        const vertex_num_t max_nbr_size = flat_graph.get_max_nbr_size();

        // Validate extracted_nbr_size does not exceed max_nbr_size
        if (extracted_nbr_size > max_nbr_size) {
            logger.error(fmt::format(
                "extracted_nbr_size ({}) cannot exceed max_nbr_size ({})",
                extracted_nbr_size, max_nbr_size
            ));
        }

        const vertex_num_t num_vertices = flat_graph.get_num_vertices();
        const auto& vecs_data = flat_graph.get_vecs_data();
        const auto& nbrs_arr = flat_graph.get_nbrs_arr();

        // Create the search graph
        search_graph_t search_graph(num_vertices, extracted_nbr_size, vecs_data);

        // Copy neighbors from FlatGraph to SearchGraph in parallel
        auto& csr_nbrs = search_graph.get_csr_nbrs();
        const vertex_id_t invalid_id = IndexTraitsT::invalid_vertex_id;

        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    const auto& nbrs = nbrs_arr[vid];
                    vertex_id_t* dst_nbrs = &csr_nbrs[static_cast<size_t>(vid) * extracted_nbr_size];

                    // Copy up to extracted_nbr_size neighbors
                    const vertex_num_t copy_count = std::min(
                        static_cast<vertex_num_t>(nbrs.size()),
                        extracted_nbr_size
                    );

                    for (vertex_num_t i = 0; i < copy_count; ++i) {
                        dst_nbrs[i] = nbrs[i].get_id();
                    }

                    // Fill remaining slots with invalid vertex ID if needed
                    for (vertex_num_t i = copy_count; i < extracted_nbr_size; ++i) {
                        dst_nbrs[i] = invalid_id;
                    }
                }
            }
        );

        return search_graph;
    }

};  // class SearchGraphFactory

}   // namespace cpu
}   // namespace artea