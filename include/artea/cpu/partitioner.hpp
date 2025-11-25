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
 * @FilePath: /Artea/include/artea/cpu/partitioner.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-24 11:15:51
 * @Date: 2025-11-24 11:15:46
 * @Description:
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <limits>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/cluster_router.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <typename vertex_num_t, typename vec_ele_t>
struct PartitionedVectors {

    using vertex_id_t = vertex_num_t;
    using offset_t = vertex_num_t;

    /** @brief Partition offsets indicating the start of each partition */
    std::vector<offset_t> part_offsets;  // size: num_partitions + 1

    /** @brief Reordered vector array for better locality */
    VectorArray<vertex_num_t, vec_ele_t> reordered_arr;

    /** @brief Original IDs mapping after reordering */
    std::vector<vertex_id_t> original_ids;
};

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename clustering_t,
    typename cluster_router_t,
    typename dist_func_t
>
class Partitioner {

    using vertex_id_t = vertex_num_t;
    using chunk_size_t = vertex_num_t;
    using chunk_id_t = vertex_num_t;
    using offset_t = vertex_num_t;

public:

    Partitioner(
        const clustering_t& clustering
    ) :
        _centroids(clustering.get_centroids()),
        _router(clustering.get_router()),
        _num_clusters(clustering.get_num_clusters())
    {}

    auto partition_and_reorder(const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr)
        -> PartitionedVectors<vertex_num_t, vec_ele_t>
    {
        if (_centroids.get_num_vecs() == 0) {
            logger.error("Model not trained. Call Clustering::fit() first.");
            throw std::runtime_error("Model not trained. Call Clustering::fit() first.");
        }

        const vertex_num_t num_vecs = vecs_arr.get_num_vecs();
        const vec_dim_t vec_dim = vecs_arr.get_vec_dim();

        // --- Assign Labels using the ClusterRouter ---
        _router.initialize();
        std::vector<cluster_id_t> labels = _router.batch_query(vecs_arr);

        // --- Count Histogram (Parallel) ---
        const uint32_t num_threads = tbb::this_task_arena::max_concurrency();
        const uint32_t num_chunks = num_threads * 4;
        const chunk_size_t chunk_size = (num_vecs + num_chunks - 1) / num_chunks;

        /** @note chunk_counts[chunk_idx][k] gives the count of cluster k in chunk chunk_idx */
        std::vector<std::vector<vertex_num_t>> chunk_counts(
            num_chunks,
            std::vector<vertex_num_t>(this->_num_clusters, 0)
        );

        tbb::parallel_for(
            static_cast<chunk_id_t>(0),
            num_chunks,
            [&](chunk_id_t chunk_idx) {
                vertex_id_t chunk_start = static_cast<vertex_id_t>(chunk_idx * chunk_size);
                vertex_id_t chunk_end = std::min(static_cast<vertex_id_t>(chunk_start + chunk_size), num_vecs);

                auto& local_counts = chunk_counts[chunk_idx];
                for (vertex_id_t i = chunk_start; i < chunk_end; ++i) {
                    local_counts[labels[i]]++;
                }
            }
        );

        // --- Compute Offsets (Serial prefix sum) ---
        std::vector<offset_t> part_offsets(this->_num_clusters + 1, 0);
        std::vector<std::vector<offset_t>> chunk_write_pos(
            num_chunks,
            std::vector<offset_t>(this->_num_clusters, 0)
        );
        offset_t accumulated_pos = 0;
        for (vertex_num_t k = 0; k < this->_num_clusters; ++k) {
            // record the starting offset for each partition
            part_offsets[k] = accumulated_pos;
            // record the starting write position for each chunk in this partition
            for (chunk_id_t c = 0; c < num_chunks; ++c) {
                chunk_write_pos[c][k] = accumulated_pos;
                accumulated_pos += static_cast<offset_t>(chunk_counts[c][k]);
            }
        }

        // final offset
        part_offsets[this->_num_clusters] = static_cast<offset_t>(accumulated_pos);
        assert(accumulated_pos == static_cast<offset_t>(num_vecs));

        // Scatter / Reorder Data (Parallel)
        VectorArray<vertex_num_t, vec_ele_t> reordered_vecs(num_vecs, vec_dim);
        std::vector<vertex_id_t> original_ids(num_vecs, 0);

        vec_ele_t* dest_base = reordered_vecs.get_all();
        const vec_ele_t* src_base = vecs_arr.get_all();

        tbb::parallel_for(
            static_cast<chunk_id_t>(0),
            num_chunks,
            [&](chunk_id_t chunk_idx) {
                vertex_id_t chunk_start = static_cast<vertex_id_t>(chunk_idx * chunk_size);
                vertex_id_t chunk_end = std::min(static_cast<vertex_id_t>(chunk_start + chunk_size), num_vecs);
                auto local_write_pos = chunk_write_pos[chunk_idx];

                for (vertex_id_t i = chunk_start; i < chunk_end; ++i) {
                    cluster_id_t label = labels[i];
                    vertex_id_t dest_idx = local_write_pos[label]++;

                    const vec_ele_t* src_ptr = src_base + i * vec_dim;
                    vec_ele_t* dest_ptr = dest_base + dest_idx * vec_dim;
                    std::copy(src_ptr, src_ptr + vec_dim, dest_ptr);

                    original_ids[dest_idx] = i;
                }
            }
        );

        return {
            std::move(part_offsets),
            std::move(reordered_vecs),
            std::move(original_ids)
        };
    }

    __attribute__((always_inline))
    auto operator()(
        const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr
    ) -> PartitionedVectors<vertex_num_t, vec_ele_t> {
        return this->partition_and_reorder(vecs_arr);
    }

private:

    const cluster_num_t _num_clusters;

    /** @brief Use VectorArray for aligned SIMD access */
    const VectorArray<vertex_num_t, vec_ele_t>& _centroids;

    /** @brief Cluster router for efficient nearest centroid search. */
    cluster_router_t& _router;

};  // class Partitioner

}   // namespace cpu
}   // namespace artea