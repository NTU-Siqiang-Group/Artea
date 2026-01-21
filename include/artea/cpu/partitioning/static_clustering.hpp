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
 * @FilePath: /Artea/include/artea/cpu/partitioning/static_clustering.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <limits>

#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/partitioning/vector_router.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename dist_func_t,
    typename cluster_router_t,
    typename vector_sampler_t,
    typename derived_class_t
>
class StaticClustering {

public:
    StaticClustering(
        const vertex_num_t num_vertices,
        const cluster_num_t num_clusters,
        const vec_dim_t vec_dim,
        const dist_func_t& dist_func,
        const vector_sampler_t& sampler
    ) :
        _num_vertices(num_vertices),
        _num_clusters(num_clusters),
        _vec_dim(vec_dim),
        _centroids(num_clusters, vec_dim),
        _dist_func(dist_func),
        _router(_centroids, dist_func),
        _sampler(sampler)
    {}

    /**
     * @brief Fit the clustering model to the provided dataset.
     *
     * @param vecs_arr The source vector data container.
     * @param sampling_ratio Ratio of data to use for training (0.0 to 1.0).
     */
    auto fit(
        const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr,
        const float sampling_ratio = 0.1f
    ) -> void {
        static_cast<derived_class_t*>(this)->fit_impl(vecs_arr, sampling_ratio);
    }

    /** --- Accessors --- **/

    __attribute__((always_inline))
    auto get_num_clusters() const -> const cluster_num_t {
        return _num_clusters;
    }

    __attribute__((always_inline))
    auto get_vec_dim() const -> const vec_dim_t {
        return _vec_dim;
    }

    __attribute__((always_inline))
    auto get_num_vertices() const -> const vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto get_centroids() const -> const VectorArray<cluster_num_t, vec_ele_t>& {
        return _centroids;
    }

    __attribute__((always_inline))
    auto get_centroids() -> VectorArray<cluster_num_t, vec_ele_t>& {
        return _centroids;
    }

    __attribute__((always_inline))
    auto get_router() const -> cluster_router_t& {
        return _router;
    }

    __attribute__((always_inline))
    auto get_sampler() const -> const vector_sampler_t& {
        return _sampler;
    }

protected:

    /** @brief Dimension of each vector. */
    const vec_dim_t _vec_dim;

    /** @brief Total number of vectors in the dataset. */
    const vertex_num_t _num_vertices;

    /** @brief Target number of clusters (K). */
    const cluster_num_t _num_clusters;

    /** @brief Use VectorArray for aligned SIMD access */
    VectorArray<cluster_num_t, vec_ele_t> _centroids;

    /** @brief Reference to the injected distance function functor. */
    const dist_func_t& _dist_func;

    /** @brief Cluster router for efficient nearest centroid search. */
    mutable cluster_router_t _router;

    /** @brief Vector sampler for random sampling of input data. */
    const vector_sampler_t& _sampler;

};

}   // namespace cpu
}   // namespace artea
