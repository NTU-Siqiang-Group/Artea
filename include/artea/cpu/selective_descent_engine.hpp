/*
 * @FilePath: /Artea/include/artea/cpu/selective_descent_engine.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-15 13:24:13
 * @Date: 2025-11-09 19:11:21
 * @Description:
 */

#pragma once

#include <utility>
#include <vector>

#include <tbb/parallel_for.h>

#include <artea/cpu/index_graph.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/recommended_nn.hpp>
#include <artea/cpu/random_nn.hpp>
#include <artea/cpu/definitions.hpp>
#include <artea/cpu/conflicts.hpp>
#include <artea/cpu/bytemap.hpp>

namespace artea {
namespace cpu {

template <typename vertex_num_t, typename vec_ele_t, DistanceMetrics dist_type = DistanceMetrics::EUCLIDEAN>
class SelectiveDescentEngine:
    public DescentEngine<vertex_num_t, vec_ele_t,
            SelectiveDescentEngine<vertex_num_t, vec_ele_t, dist_type>>
{

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;
    using conflict_t = Conflicts<vertex_num_t, vec_ele_t, dist_type>;

public:
    SelectiveDescentEngine(
        IndexGraph<vertex_num_t, vec_ele_t>& graph,
        conflict_t& conflict,
        RecommendedNN<vertex_num_t, vec_ele_t>& recom_nn,
        RandomNN<vertex_num_t, vec_ele_t>& random_nn
    ) : DescentEngine(graph),
        _conflict(conflict),
        _recom_nn(recom_nn),
        _random_nn(random_nn),
        _active_map(graph.get_num_vertices())
    {}

    __attribute__((always_inline))
    auto initialize_op(const vertex_id_t vid) -> void override {
        this->_random_nn.generate(
            this->_nbrs_arr[vid],
            this->_edges_per_vertex
        );
    }

    /** @brief Update the neighbors of a single vertex in the graph.
      * @param vid The vertex id to update.
      * @note Update strategy should be implemented here.
    */
    auto propagate_op(const vertex_id_t vid) -> void override {
        const nbr_arr_t& recom_nbrs = _recom_nn.get_recom_nbrs(vid);
        const nbr_arr_t& original_nbrs = this->_nbrs_arr[vid];

        nbr_arr_t reserved_nbrs = nbr_arr_t(this->_edges_per_vertex);


        // RNG strategy
        std::size_t recom_idx = 0, org_idx = 0;
        Array<bool> org_flags = Array<bool>::alloc(this->_edges_per_vertex);
        while (recom_idx < recom_nbrs.size() and
                org_idx < original_nbrs.size() and
                reserved_nbrs.size() < this->_edges_per_vertex
        ) {
            distance_t recom_dist = recom_nbrs[recom_idx].distance,
                        org_dist = original_nbrs[org_idx].distance;
            if (recom_dist < org_dist) {
                const auto [accepted, result_dist] = _conflict.rng_strategy(
                    recom_nbrs[recom_idx],
                    reserved_nbrs
                );
                if (accepted) {
                    reserved_nbrs.push_back(recom_nbrs[recom_idx]);
                    org_flags.push_back(false);
                }
                else /* not accepted */ {
                    this->_recom_nn.append_edge(
                        vid,
                        recom_nbrs[recom_idx].dest,
                        result_dist
                    );
                }
                ++recom_idx;
            }
            else {
                const auto [accepted, result_dist] = _conflict.rng_strategy(
                    original_nbrs[org_idx],
                    reserved_nbrs,
                    org_flags
                );
                if (accepted) {
                    reserved_nbrs.push_back(original_nbrs[org_idx]);
                    org_flags.push_back(true);
                }
                else /* not accepted */ {
                    this->_recom_nn.append_edge(
                        vid,
                        original_nbrs[org_idx].dest,
                        result_dist
                    );
                }
                ++org_idx;
            }
        }

        // Write back to nbrs_arr (this operation is not in thread-race context)
        this->_nbrs_arr[vid] = std::move(reserved_nbrs);
    }

private:

    /** @brief The recommended nearest neighbor generator. */
    RecommendedNN<vertex_num_t, vec_ele_t>& _recom_nn;

    /** @brief The random nearest neighbor generator. */
    RandomNN<vertex_num_t, vec_ele_t>& _random_nn;

    /** @brief The conflict checker. */
    conflict_t& _conflict;

    Bytemap _active_map;

};  // class SelectiveDescentEngine

}   // namespace cpu
}   // namespace artea