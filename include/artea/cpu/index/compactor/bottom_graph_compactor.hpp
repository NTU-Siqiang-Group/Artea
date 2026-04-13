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
 * @FilePath: /Artea/include/artea/cpu/index/compactor/bottom_graph_compactor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Compactor: dynamic BottomGraph -> compact BottomGraph.
 */

#pragma once

#include <algorithm>
#include <string>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename IndexTraitsT>
class BottomGraphCompactor {

    using vertex_num_t   = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t    = typename IndexTraitsT::vertex_id_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using compact        = typename IndexTraitsT::compact;
    using nbr_arr_checker_t = typename IndexTraitsT::nbr_arr_checker_t;

public:
    /**
     * @brief Compact a BottomGraph-like structure into a compact::bottom_graph_t.
     *
     * Reads neighbor arrays and vector data from @p src, copies up to
     * @p extracted_nbr_size neighbors per vertex into a flat CSR layout.
     *
     * @param src                The source graph (any type exposing
     *                           get_num_vertices, get_vecs_data, get_nbrs_arr,
     *                           layer_config).
     * @param extracted_nbr_size Fixed neighbor count per vertex in the output.
     * @return A new compact::bottom_graph_t.
     */
    template <typename BottomGraphT>
    static auto compact_graph(
        const BottomGraphT& src,
        const vertex_num_t extracted_nbr_size
    ) -> typename compact::bottom_graph_t {
        const vertex_num_t max_nbr_size = src.layer_config().max_nbr_size();

        if (extracted_nbr_size > max_nbr_size) {
            ARTEA_ERROR(fmt::format(
                "extracted_nbr_size ({}) cannot exceed max_nbr_size ({})",
                extracted_nbr_size, max_nbr_size
            ));
        }

        const vertex_num_t num_vertices = src.get_num_vertices();
        const auto& vecs_data = src.get_vecs_data();
        const auto& nbrs_arr = src.get_nbrs_arr();

        typename compact::bottom_graph_t result(vecs_data, extracted_nbr_size);
        auto& csr_nbrs = result.get_csr_nbrs();
        const vertex_id_t invalid_id = IndexTraitsT::invalid_vertex_id;

        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    const auto& nbrs = nbrs_arr[vid];
                    vertex_id_t* dst = &csr_nbrs[static_cast<size_t>(vid) * extracted_nbr_size];
                    const vertex_num_t copy_count = std::min(
                        static_cast<vertex_num_t>(nbrs.size()), extracted_nbr_size);
                    for (vertex_num_t i = 0; i < extracted_nbr_size; ++i) {
                        dst[i] = (i < copy_count) ? nbrs[i].get_vid() : invalid_id;
                    }

                    #ifndef NDEBUG
                    if (!nbr_arr_checker_t::invalid_id_suffix_check(dst, extracted_nbr_size)) {
                        ARTEA_ERROR("Error: Invalid search graph row suffix layout.");
                    }
                    #endif
                }
            }
        );

        return result;
    }

    /**
     * @brief Load a conv_graph index from file and compact it.
     */
    static auto from_index_file(
        const std::string& file_path,
        const vertex_num_t extracted_nbr_size,
        const vector_array_t& vecs_data
    ) -> typename compact::bottom_graph_t {
        using index_t = typename IndexTraitsT::conv_graph::index_t;
        index_t bottom_graph = index_t::restore(file_path, vecs_data);
        return compact_graph(bottom_graph, extracted_nbr_size);
    }

};  // class BottomGraphCompactor

}   // namespace cpu
}   // namespace artea
