/*
 * @FilePath: /Artea/include/artea/cpu/propagation/concurrent_recom_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <vector>
#include <algorithm>

#include <omp.h>
#include <tbb/concurrent_vector.h>
#include <range/v3/view/zip.hpp>

#include <artea/definitions.hpp>
#include <artea/cpu/propagation/recommended_nn.hpp>

namespace artea {
namespace cpu {

template<
    typename vertex_num_t,
    typename vec_ele_t
>
class ConcurrentRecomNN final:
    public RecommendedNN<vertex_num_t, vec_ele_t,
        ConcurrentRecomNN<vertex_num_t, vec_ele_t>>
{
    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;
    using base_class_t = RecommendedNN<vertex_num_t, vec_ele_t,
        ConcurrentRecomNN<vertex_num_t, vec_ele_t>>;

public:
    /** @brief Construct a new Concurrent Recom NN object. */
    ConcurrentRecomNN(
        const vertex_num_t num_vertices,
        const vertex_num_t recom_buf_size
    ) : base_class_t(num_vertices, recom_buf_size) {
        _recom_buf.resize(num_vertices);
        for (auto& recom_arr : _recom_buf) {
            recom_arr.reserve(recom_buf_size);
        }
    }

    /** @brief Write an edge to the recommendation buffer. */
    auto append_edge_impl(const vertex_id_t src, const nbr_t& nbr) -> void {
        _recom_buf[src].push_back(nbr);
    }

    auto append_edge_impl(const vertex_id_t src, const vertex_id_t dest, const distance_t dist) -> void {
        _recom_buf[src].emplace_back(dest, dist);
    }

    /** @brief Get the recommended neighbors of a vertex.
      * @param src The source vertex.
      * @return nbr_arr_t The recommended neighbors of the source vertex.
      * @note The neighbors are sorted by distance in ascending order.
      * @note This function must be used in non-concurrent environment.
     */
    auto get_recom_nbrs_impl(const vertex_id_t src) -> nbr_arr_t& {
        return this->_ro_recom_buf[src];
    }

    auto flush_impl() -> void {
        tbb::parallel_for(tbb::blocked_range<vertex_id_t>(0, this->_num_vertices),
        [=](const tbb::blocked_range<vertex_id_t>& r) {
            for (vertex_id_t i = r.begin(); i != r.end(); ++i) {
                this->_ro_recom_buf[i].assign(_recom_buf[i].begin(), _recom_buf[i].end());
                std::sort(
                    this->_ro_recom_buf[i].begin(),
                    this->_ro_recom_buf[i].end(),
                    NeighborComparator<vertex_id_t, vec_ele_t>
                );
            }
        });
        clear_impl();
    }

    /** @brief Clear the recommendation buffer.
      * @note This function must be used in non-concurrent environment.
     */
    auto clear_impl() -> void {
        // TODO test the performance with or without parallelization
        tbb::parallel_for(tbb::blocked_range<vertex_id_t>(0, this->_num_vertices),
        [=](const tbb::blocked_range<vertex_id_t>& r) {
            for (vertex_id_t i = r.begin(); i != r.end(); ++i) {
                _recom_buf[i].clear();
            }
        });
        // for (vertex_id_t i = 0; i < this->_num_vertices; ++i) {
        //     _recom_buf[i].clear();
        // }
    }

private:

    /** @brief Buffer for appending neighbors. */
    std::vector<tbb::concurrent_vector<nbr_t>> _recom_buf;

};  // class ConcurrentRecomNN

}   // namespace cpu
}   // namespace artea