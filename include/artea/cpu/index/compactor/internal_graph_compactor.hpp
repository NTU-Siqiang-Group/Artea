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
 * @Description: Single-layer compactor: InternalGraph -> CompactInternalGraph.
 */

#pragma once

#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Compactor that converts a single-layer @c InternalGraph into a
 *        @c CompactInternalGraph by copying neighbors into a fixed-stride
 *        sentinel-terminated layout.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class InternalGraphCompactor {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using lnbr_t = typename IndexTraitsT::lnbr_t;
    using internal_graph_t = typename IndexTraitsT::internal_graph_t;
    using compact_internal_graph_t = typename IndexTraitsT::compact_internal_graph_t;

public:
    /**
     * @brief Convert an @c InternalGraph to a @c CompactInternalGraph in parallel.
     *
     * The source graph is read-only; the resulting @c CompactInternalGraph owns
     * a freshly allocated CSR array. The first @c valid_count neighbors of each
     * vertex are copied into the compact layout, and the remaining slots up to
     * @p extracted_nbr_size are filled with @c IndexTraitsT::invalid_lnbr so
     * that the sentinel-based traversal pattern can be used by readers.
     *
     * Inter-layer links are also copied 1:1.
     *
     * @param internal_graph     The source internal graph (read-only).
     * @param extracted_nbr_size Fixed number of neighbor slots per vertex in
     *                           the resulting compact graph. Must not exceed
     *                           @c internal_graph.max_nbr_size().
     * @return A new CompactInternalGraph with the same number of vertices.
     */
    static auto from_internal_graph(
        const internal_graph_t& internal_graph,
        const vertex_num_t extracted_nbr_size
    ) -> compact_internal_graph_t {
        const vertex_num_t src_max_nbr_size = internal_graph.max_nbr_size();

        if (extracted_nbr_size > src_max_nbr_size) {
            ARTEA_ERROR(fmt::format(
                "extracted_nbr_size ({}) cannot exceed source max_nbr_size ({})",
                extracted_nbr_size, src_max_nbr_size
            ));
        }

        const vertex_num_t num_vertices = internal_graph.get_num_vertices();
        compact_internal_graph_t compact_graph(num_vertices, extracted_nbr_size);

        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    // Source: full block [header | nbr_0 | ... | nbr_{src_max-1}]
                    auto src_block = internal_graph.fetch_nbrs(vid);
                    const uint64_t valid_count = internal_graph.num_valid_nbrs(vid);
                    const lnbr_t* src_nbrs = src_block.data() + 1;  // skip header

                    // Destination: contiguous extracted_nbr_size lnbr_t slots.
                    auto dst_nbrs = compact_graph.fetch_nbrs(vid);

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

                    // Copy inter-layer link
                    compact_graph.set_inter_layer_link(
                        vid, internal_graph.get_inter_layer_link(vid)
                    );
                }
            }
        );

        return compact_graph;
    }

};  // class InternalGraphCompactor

}   // namespace cpu
}   // namespace artea
