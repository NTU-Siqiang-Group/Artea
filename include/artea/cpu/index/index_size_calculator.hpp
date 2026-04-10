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
 * @FilePath: /Artea/include/artea/cpu/index/index_size_calculator.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Utility for calculating index sizes of various graph structures
 */

#pragma once

#include <cstddef>

namespace artea {
namespace cpu {

struct IndexSizeInfo {
    size_t total_bytes = 0;
    double total_mb = 0.0;
};

template <typename IndexTraitsT>
class IndexSizeCalculator {
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using compact = typename IndexTraitsT::compact;

public:
    /**
     * @brief Calculate size of a descent graph (uses max_nbr_size)
     * Each vertex has max_nbr_size neighbors.
     */
    template <typename DescentGraphT>
    static auto calculate_descent_graph_size(const DescentGraphT& graph) -> IndexSizeInfo {
        IndexSizeInfo info;

        vertex_num_t num_vertices = graph.get_num_vertices();
        vertex_num_t max_nbr_size = graph.layer_config().max_nbr_size();

        info.total_bytes = static_cast<size_t>(num_vertices) * max_nbr_size * sizeof(vertex_id_t);
        info.total_mb = info.total_bytes / (1024.0 * 1024.0);

        return info;
    }

    /**
     * @brief Calculate size of a flat search graph (uses extracted_nbr_size)
     * CSR format: num_vertices * extracted_nbr_size.
     */
    static auto calculate_size(const compact::descent_graph_t& search_graph) -> IndexSizeInfo {
        IndexSizeInfo info;

        vertex_num_t num_vertices = search_graph.get_num_vertices();
        vertex_num_t extracted_nbr_size = search_graph.get_extracted_nbr_size();

        info.total_bytes = static_cast<size_t>(num_vertices) * extracted_nbr_size * sizeof(vertex_id_t);
        info.total_mb = info.total_bytes / (1024.0 * 1024.0);

        return info;
    }
};

} // namespace cpu
} // namespace artea
