/*
 * @FilePath: /Artea/include/artea/cpu/simple_propagate.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-10 01:50:01
 * @Date: 2025-11-09 19:11:21
 * @Description: 
 */

#pragma once

#include <artea/cpu/index_graph.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/recommended_nn.hpp>
#include <artea/cpu/random_nn.hpp>
#include <artea/cpu/types.hpp>
#include <artea/cpu/simd_distance.hpp>
#include <range/v3/view/zip.hpp>

namespace artea {
namespace cpu {

template <typename vertex_num_t, typename vec_ele_t>
class SimplePropagate {

    using distance_t = IndexGraph<vertex_num_t, vec_ele_t>::distance_t;
    using vertex_id_t = IndexGraph<vertex_num_t, vec_ele_t>::vertex_id_t;
    using nbr_t = IndexGraph<vertex_num_t, vec_ele_t>::nbr_t;
    using nbr_arr_t = IndexGraph<vertex_num_t, vec_ele_t>::nbr_arr_t;

public:

    /** @brief Initialize the neighbors of each vertex in the graph. */
    auto init_strategy() -> void override {
        // intialize neighbors array with random_nn
        tbb::parallel_for(tbb::blocked_range<vertex_id_t>(0, this->_num_vertices),
        [=](const tbb::blocked_range<vertex_id_t>& r) {
            for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                random_nn.generate(this->_nbrs_arr[vid], this->_num_nbrs_per_vertex);
            }
        });
    }

    /** @brief Update the neighbors of each vertex in the graph. */
    auto update_strategy() -> void override {
        // update neighbors array with recom_nn
        tbb::parallel_for(tbb::blocked_range<vertex_id_t>(0, this->_num_vertices),
        [=](const tbb::blocked_range<vertex_id_t>& r) {
            for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                nbr_arr_t recom_nbrs = this->_recom_nn.get_recom_nbrs(vid);
                nbr_arr_t& original_nbrs = this->_nbrs_arr[vid];
                nbr_arr_t reserved_nbrs(this->_num_nbrs_per_vertex);
                // RNG strategy
                uint32_t recom_idx = 0, org_idx = 0;
                Array<bool> org_flags(this->_num_nbrs_per_vertex);
                while (recom_idx < recom_nbrs.size() and 
                       org_idx < original_nbrs.size() and
                       reserved_nbrs.size() < this->_num_nbrs_per_vertex
                ) {
                    vertex_id_t recom_vid = recom_nbrs[recom_idx].dest,
                                org_vid = original_nbrs[org_idx].dest;
                    distance_t recom_dist = recom_nbrs[recom_idx].distance,
                                org_dist = original_nbrs[org_idx].distance;
                    if (recom_dist < org_dist) {
                        if (_check_comflict_for_recom_nbr(recom_nbrs[recom_idx], reserved_nbrs)) {
                            reserved_nbrs.push_back(recom_nbrs[recom_idx]);
                            org_flags.push_back(false);
                        }
                        ++recom_idx;
                    } else {
                        if (_check_comflict_for_org_nbr(original_nbrs[org_idx], reserved_nbrs, org_flags)) {
                            reserved_nbrs.push_back(original_nbrs[org_idx]);
                            org_flags.push_back(true);
                        }
                        ++org_idx;
                    }
                }
                this->_nbrs_arr[vid] = std::move(reserved_nbrs);
            }
        });
    }

private:

    // TODO: check_conflict functions should be static

    auto _check_comflict_for_org_nbr(
        nbr_t& nbr,
        const nbr_arr_t& reserved_nbrs,
        const Array<bool>& org_flags
    ) {
        vec_ele_t* vec = this->_vecs_arr.get_vec(nbr.dest);
        for (auto const& [rnbr, org_flag] : ranges::views::zip(reserved_nbrs, org_flags)) {
            if (org_flag) {
                continue;
            }
            vec_ele_t* rvec = this->_vecs_arr.get_vec(rnbr.dest);
            distance_t dist = simd_distance(vec, rvec, this->_vec_dim);
            if (dist < nbr.distance) {
                return false;
            }
        }
        return true;
    }

    auto _check_comflict_for_recom_nbr(
        nbr_t& nbr,
        const nbr_arr_t& reserved_nbrs
    ) {
        vec_ele_t* vec = this->_vecs_arr.get_vec(nbr.dest);
        for (auto const& rnbr : reserved_nbrs) {
            vec_ele_t* rvec = this->_vecs_arr.get_vec(rnbr.dest);
            distance_t dist = simd_distance(vec, rvec, this->_vec_dim);
            if (dist < nbr.distance) {
                return false;
            }
        }
        return true;
    }

    /** @brief The recommended nearest neighbor generator. */
    RecommendedNN<vertex_num_t, vec_ele_t>& _recom_nn;

    /** @brief The random nearest neighbor generator. */
    RandomNN<vertex_num_t, vec_ele_t>& _random_nn;

};  // class SimplePropagate

}   // namespace cpu
}   // namespace artea