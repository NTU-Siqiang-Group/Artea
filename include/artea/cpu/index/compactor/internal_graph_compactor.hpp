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
 * @FilePath: /Artea/include/artea/cpu/index/compactor/internal_graph_compactor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Compactor: dynamic InternalGraph -> compact InternalGraph.
 */

#pragma once

#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Compactor that converts a dynamic::InternalGraph into a
 *        compact::InternalGraph by copying neighbors into a fixed-stride
 *        sentinel-terminated layout.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class InternalGraphCompactor {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t  = typename IndexTraitsT::vertex_id_t;
    using lnbr_t       = typename IndexTraitsT::lnbr_t;
    using compact      = typename IndexTraitsT::compact;
    using dynamic      = typename IndexTraitsT::dynamic;

public:
    /**
     * @brief Compact a dynamic::InternalGraph into a compact::InternalGraph.
     *
     * @param src                The source internal graph (read-only).
     * @param extracted_nbr_size Fixed neighbor slots per vertex in the output.
     * @return A new compact::InternalGraph.
     */
    static auto compact_graph(
        const typename dynamic::internal_graph_t& src,
        const vertex_num_t extracted_nbr_size
    ) -> typename compact::internal_graph_t {
        const vertex_num_t src_max_nbr_size = src.max_nbr_size();

        if (extracted_nbr_size > src_max_nbr_size) {
            ARTEA_ERROR(fmt::format(
                "extracted_nbr_size ({}) cannot exceed source max_nbr_size ({})",
                extracted_nbr_size, src_max_nbr_size
            ));
        }

        const vertex_num_t num_vertices = src.get_num_vertices();
        typename compact::internal_graph_t result(num_vertices, extracted_nbr_size);

        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    auto src_block = src.fetch_nbrs(vid);
                    const uint64_t valid_count = src.num_valid_nbrs(vid);
                    const lnbr_t* src_nbrs = src_block.data() + 1;  // skip header

                    auto dst_nbrs = result.fetch_nbrs(vid);

                    const vertex_num_t copy_count = std::min(
                        static_cast<vertex_num_t>(valid_count),
                        extracted_nbr_size
                    );
                    for (vertex_num_t i = 0; i < copy_count; ++i) {
                        dst_nbrs[i] = src_nbrs[i];
                    }
                    for (vertex_num_t i = copy_count; i < extracted_nbr_size; ++i) {
                        dst_nbrs[i] = IndexTraitsT::invalid_lnbr;
                    }

                    result.set_inter_layer_link(
                        vid, src.get_inter_layer_link(vid)
                    );
                }
            }
        );

        return result;
    }

};  // class InternalGraphCompactor

}   // namespace cpu
}   // namespace artea
