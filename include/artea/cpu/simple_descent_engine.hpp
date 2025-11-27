/*
 * @FilePath: /Artea/include/artea/cpu/simple_descent_engine.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-27 11:53:04
 * @Date: 2025-11-09 19:11:21
 * @Description:
 */

#pragma once

#include <utility>
#include <vector>

#include <tbb/parallel_for.h>

#include <artea/cpu/index/index_graph.hpp>
#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/propagation/recommended_nn.hpp>
#include <artea/cpu/utils/random_seq.hpp>
#include <artea/cpu/conflicts.hpp>
#include <artea/cpu/utils/allocator.hpp>
#include <artea/common/element_pos.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <typename vertex_num_t, typename vec_ele_t, DistanceMetrics dist_type = DistanceMetrics::EUCLIDEAN>
class SimpleDescentEngine:
    public DescentEngine<vertex_num_t, vec_ele_t,
            SimpleDescentEngine<vertex_num_t, vec_ele_t, dist_type>>
{

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;
    using conflict_t = Conflicts<vertex_num_t, vec_ele_t, dist_type>;

public:

    SimpleDescentEngine(
        IndexGraph<vertex_num_t, vec_ele_t>& graph,
        conflict_t& conflict,
        RecommendedNN<vertex_num_t, vec_ele_t>& recom_nn,
        RandomSeq<vertex_num_t, vec_ele_t>& random_seq
    ) : DescentEngine(graph),
        _conflict(conflict),
        _recom_nn(recom_nn),
        _random_seq(random_seq)
    {}

    /** @brief Initialize the neighbors of a single vertex in the graph.
      * @param vid The vertex id to initialize.
    */
    __attribute__((always_inline))
    auto initialize_op(const vertex_id_t vid) -> void override {
        this->_random_seq.generate(
            this->_nbrs_arr[vid],
            this->_edges_limit
        );
    }

    /** @brief Update the neighbors of a single vertex in the graph.
      * @param vid The vertex id to update.
      * @param cur_iter The current iteration number.
      * @note Neighborhood update strategy should be implemented here.
      * TODO: implement add reverse edge interval.
    */
    auto propagate_op(const vertex_id_t vid, const iter_t cur_iter) -> void override {

        const nbr_arr_t& recom_nbrs = _recom_nn.get_recom_nbrs(vid),
            original_nbrs = this->_nbrs_arr[vid];

        nbr_arr_t reserved_nbrs;
        reserved_nbrs.reserve(this->_edges_limit);
        Array<bool> from_org_flags = Array<bool>::alloc(this->_edges_limit);

        std::size_t org_idx = 0, recom_idx = 0;
        std::size_t& idx_box [] = { org_idx, recom_idx };

        vertex_id_t last_selected_id = vid;

        // RNG strategy pruning and recommendation
        while (
            org_idx < original_nbrs.size() and
            recom_idx < recom_nbrs.size()
        ) {
            int selected_pos = min_element_pos<2>({
                original_nbrs[org_idx].distance,
                recom_nbrs[recom_idx].distance
            });

            std::size_t& selected_idx = { org_idx, recom_idx }[selected_pos];
            const nbr_t& selected_nbr = { original_nbrs, recom_nbrs }[selected_pos][selected_idx];

            // unique the selected neighbors
            if (selected_nbr.dest == last_selected_id) {
                ++selected_idx;
                continue;
            }
            last_selected_id = selected_nbr.dest;

            const auto [accepted, result_vid, result_dist] = _conflict.rng_strategy(
                selected_nbr,
                selected_pos == 0,
                reserved_nbrs,
                from_org_flags
            );

            if (accepted) {
                reserved_nbrs.push_back(selected_nbr);
                from_org_flags.push_back(selected_pos == 0);
            }
            else /* not accepted */ {
                this->_recom_nn.append_edge(
                    selected_nbr.dest,
                    result_vid,
                    result_dist
                );
            }

            ++selected_idx;
        }   // end while

        // Write back to nbrs_arr (this operation is not in thread-race context)
        this->_nbrs_arr[vid] = reserved_nbrs;

    }   // end function `propagate_op`

private:

    /** @brief The recommended nearest neighbor generator. */
    RecommendedNN<vertex_num_t, vec_ele_t>& _recom_nn;

    /** @brief The random nearest neighbor generator. */
    RandomSeq<vertex_num_t, vec_ele_t>& _random_seq;

    /** @brief The RNG-rule conflict detector. */
    Conflicts<vertex_num_t, vec_ele_t, dist_type>& _conflict;

};  // class SimplePropagate

}   // namespace cpu
}   // namespace artea