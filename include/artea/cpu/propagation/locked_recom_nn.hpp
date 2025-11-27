/*
 * @FilePath: /Artea/include/artea/cpu/propagation/locked_recom_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-27 11:41:34
 * @Date: 2025-11-13 09:28:38
 * @Description:
 */

#include <vector>
#include <algorithm>
#include <mutex>
#include <memory>

#include <omp.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <artea/definitions.hpp>
#include <artea/cpu/propagation/recommended_nn.hpp>

namespace artea {
namespace cpu {

template<
    typename vertex_num_t,
    typename vec_ele_t
>
class LockedRecomNN final:
    public RecommendedNN<vertex_num_t, vec_ele_t,
        LockedRecomNN<vertex_num_t, vec_ele_t>>
{

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;
    using base_class_t = RecommendedNN<vertex_num_t, vec_ele_t,
        LockedRecomNN<vertex_num_t, vec_ele_t>>;

public:
    /** @brief Construct a new LockedRecomNN object. */
    LockedRecomNN(
        const vertex_num_t num_vertices,
        const vertex_num_t& recom_buf_size
    ): base_class_t(num_vertices, recom_buf_size) {
        _recom_buf.resize(num_vertices);
        for (auto& vec : _recom_buf) {
            vec.reserve(recom_buf_size);
        }
        _locks.reserve(num_vertices);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            _locks.emplace_back(std::make_unique<std::mutex>());
        }
    }

    /** @brief Write an edge to the recommendation buffer with thread-safety. */
    __attribute__((always_inline))
    auto append_edge_impl(const vertex_id_t src, const nbr_t& nbr) -> void {
        std::lock_guard<std::mutex> lock(*_locks[src]);
        _recom_buf[src].push_back(nbr);
    }

    /** @brief Write an edge to the recommendation buffer with thread-safety. */
    __attribute__((always_inline))
    auto append_edge_impl(const vertex_id_t src, const vertex_id_t dest, const distance_t dist) -> void {
        std::lock_guard<std::mutex> lock(*_locks[src]);
        _recom_buf[src].emplace_back(dest, dist);
    }

    /** @brief Get the recommended neighbors of a vertex.
      * @param src The source vertex.
      * @return nbr_arr_t& The recommended neighbors of the source vertex.
      * @note The neighbors are sorted by distance in ascending order.
      * @note Ensure that the recommendation buffer is not modified while this function is being called.
     */
    auto get_recom_nbrs_impl(const vertex_id_t& src) -> nbr_arr_t& {
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
      * @note Ensure that the recommendation buffer is not modified while this function is being called.
     */
    auto clear_impl() -> void {
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

    /** @brief Buffer for appending neighbors. Uses standard vectors. */
    std::vector<std::vector<nbr_t>> _recom_buf;

    /** @brief A mutex for each buffer entry to allow fine-grained locking. */
    std::vector<std::unique_ptr<std::mutex>> _locks;

};  // class LockedRecomNN


}   // namespace cpu
}   // namespace artea