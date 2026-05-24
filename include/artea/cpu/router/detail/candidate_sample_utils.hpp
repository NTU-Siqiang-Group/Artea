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
 * @FilePath: /Artea/include/artea/cpu/router/detail/candidate_sample_utils.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Stateless apex-bucket samplers shared by dynamic / compact
 *               routers. Each sampler has a queue-push form and a
 *               return-value form; both are implemented independently
 *               (no mutual delegation).
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename RouterTraitsT>
class CandidateSampleUtils {

    using vertex_id_t           = typename RouterTraitsT::vertex_id_t;
    using layer_id_t            = typename RouterTraitsT::layer_id_t;
    using vec_ele_t             = typename RouterTraitsT::vec_ele_t;
    using distance_t            = typename RouterTraitsT::distance_t;
    // dist_func type is per-method template arg (DistFuncT); deduced from caller.
    using vector_array_t        = typename RouterTraitsT::vector_array_t;
    using std_candidate_queue_t = typename RouterTraitsT::std_candidate_queue_t;

public:
    /** @brief Multi-seed apex sampling into a caller-owned queue. */
    template <typename HierarchicalGraphT, typename DistFuncT>
    static auto sample_entries(
        const vector_array_t&     vecs_data,
        const DistFuncT&          dist_func,
        const HierarchicalGraphT& hier_graph,
        const vec_ele_t*          query_vec,
        std_candidate_queue_t&    candidate_queue
    ) -> void {
        const layer_id_t top_level_id = hier_graph.top_occupied_level_id();
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id) {
            ARTEA_ERROR("sample_entries: hierarchy has no occupied levels");
        }
        const auto& bucket = hier_graph.get_vids_with_highest_level(top_level_id);
        if (bucket.empty()) {
            ARTEA_ERROR("sample_entries: top-level bucket is empty");
        }
        std::atomic_thread_fence(std::memory_order_acquire);

        const std::size_t pool_size = bucket.size();
        const std::size_t queue_cap = candidate_queue.capacity();
        const std::size_t start  = _draw_random_index(pool_size);
        const std::size_t stride = (pool_size <= queue_cap) ? 1 : (pool_size / queue_cap);
        const std::size_t take   = std::min(queue_cap, pool_size);

        for (std::size_t i = 0; i < take; ++i) {
            const std::size_t random_idx = (start + i * stride) % pool_size;
            const vertex_id_t sampled_vid  = bucket[random_idx];
            const distance_t  sampled_dist = dist_func(query_vec, vecs_data.get(sampled_vid));
            candidate_queue.try_push(sampled_vid, sampled_dist);
        }
    }

    /** @brief Multi-seed apex sampling returned directly as a vector. */
    template <typename HierarchicalGraphT, typename DistFuncT>
    static auto sample_entries(
        const vector_array_t&     vecs_data,
        const DistFuncT&          dist_func,
        const HierarchicalGraphT& hier_graph,
        const vec_ele_t*          query_vec,
        const std::size_t         num_entries
    ) -> std::vector<std::pair<vertex_id_t, distance_t>> {
        const layer_id_t top_level_id = hier_graph.top_occupied_level_id();
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id) {
            ARTEA_ERROR("sample_entries: hierarchy has no occupied levels");
        }
        const auto& bucket = hier_graph.get_vids_with_highest_level(top_level_id);
        if (bucket.empty()) {
            ARTEA_ERROR("sample_entries: top-level bucket is empty");
        }
        std::atomic_thread_fence(std::memory_order_acquire);

        const std::size_t pool_size = bucket.size();
        const std::size_t start  = _draw_random_index(pool_size);
        const std::size_t stride = (pool_size <= num_entries) ? 1 : (pool_size / num_entries);
        const std::size_t take   = std::min(num_entries, pool_size);

        std::vector<std::pair<vertex_id_t, distance_t>> seeds;
        seeds.reserve(take);
        for (std::size_t i = 0; i < take; ++i) {
            const std::size_t random_idx = (start + i * stride) % pool_size;
            const vertex_id_t sampled_vid  = bucket[random_idx];
            const distance_t  sampled_dist = dist_func(query_vec, vecs_data.get(sampled_vid));
            seeds.emplace_back(sampled_vid, sampled_dist);
        }
        return seeds;
    }

    /** @brief One-seed apex sampling pushed into a caller-owned queue. */
    template <typename HierarchicalGraphT, typename DistFuncT>
    static auto sample_single_entry(
        const vector_array_t&     vecs_data,
        const DistFuncT&          dist_func,
        const HierarchicalGraphT& hier_graph,
        const vec_ele_t*          query_vec,
        std_candidate_queue_t&    candidate_queue
    ) -> void {
        const layer_id_t top_level_id = hier_graph.top_occupied_level_id();
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id) {
            ARTEA_ERROR("sample_single_entry: hierarchy has no occupied levels");
        }
        const auto& bucket = hier_graph.get_vids_with_highest_level(top_level_id);
        if (bucket.empty()) {
            ARTEA_ERROR("sample_single_entry: top-level bucket is empty");
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const std::size_t random_idx = _draw_random_index(bucket.size());
        const vertex_id_t vid  = bucket[random_idx];
        const distance_t  dist = dist_func(query_vec, vecs_data.get(vid));
        candidate_queue.try_push(vid, dist);
    }

    /** @brief One-seed apex sampling returned as a (vid, dist) pair. */
    template <typename HierarchicalGraphT, typename DistFuncT>
    static auto sample_single_entry(
        const vector_array_t&     vecs_data,
        const DistFuncT&          dist_func,
        const HierarchicalGraphT& hier_graph,
        const vec_ele_t*          query_vec
    ) -> std::pair<vertex_id_t, distance_t> {
        const layer_id_t top_level_id = hier_graph.top_occupied_level_id();
        if (top_level_id == HierarchicalGraphT::unassigned_highest_level_id) {
            ARTEA_ERROR("sample_single_entry: hierarchy has no occupied levels");
        }
        const auto& bucket = hier_graph.get_vids_with_highest_level(top_level_id);
        if (bucket.empty()) {
            ARTEA_ERROR("sample_single_entry: top-level bucket is empty");
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const std::size_t random_idx = _draw_random_index(bucket.size());
        const vertex_id_t vid  = bucket[random_idx];
        const distance_t  dist = dist_func(query_vec, vecs_data.get(vid));
        return {vid, dist};
    }

private:
    static auto _draw_random_index(const std::size_t upper_bound) -> std::size_t {
        thread_local std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<std::size_t> dist(0, upper_bound - 1);
        return dist(rng);
    }

};  // class CandidateSampleUtils

}   // namespace cpu
}   // namespace artea
