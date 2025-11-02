/*
 * @FilePath: /Artea/include/artea/cpu/fast_1nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-10-23 18:56:28
 * @Date: 2025-10-23 16:48:28
 * @Description: 
 */

#pragma once

#include <artea/types.hpp>
#include <artea/config.hpp>
#include <artea/cpu/vector_array.hpp>
#include <omp.h>

namespace artea {
namespace cpu {

enum class EngineType{
    ANN_BASED,
    BRUTE_FORCE
};

template <
    typename vecs_num_t,
    typename vec_ele_t,
    typename vec_id_t = vecs_num_t
>
class Fast1NNEngine {
    
public:
    Fast1NNEngine() = default;

    ~Fast1NNEngine() = default;

    template <EngineType engine_type>
    auto compute(
        const VectorArray<vecs_num_t, vec_ele_t>* vecs,
        const vec_dim_t vec_dim,
        const VectorArray<part_num_t, vec_ele_t>* centroids,
        const part_num_t num_clusters,
        const part_id_t* results
    ) -> void {

        omp_set_num_threads(fast_1nn_threads);
        #pragma omp parallel for schedule(static)
        for (vecs_num_t vec_id = 0; vec_id < _vecs->size(); vec_id++) {
            /** implementaion logic:
            *      vec_ele_t* vec = vecs->get(vec_id);
            *      part_id_t best_cluster_id = 0;
            *      vec_ele_t best_dist = std::numeric_limits<vec_ele_t>::max();
            *      for (part_id_t cluster_id = 0; cluster_id < num_clusters; cluster_id++) {
            *          vec_ele_t* centroid_vec = centroids->get(cluster_id);
            *          vec_ele_t dist = 0;
            *          for (vec_dim_t dim = 0; dim < vec_dim; dim++) {
            *              dist += (vec[dim] - centroid_vec[dim]) * (vec[dim] - centroid_vec[dim]);
            *          }
            *      }
            *      if (dist < best_dist) {
            *          best_dist = dist;
            *          best_cluster_id = cluster_id;
            *      }
            *      return best_cluster_id;
            */
        }
    }

};  // class Fast1NNEngine

}   // namespace cpu
}   // namespace artea