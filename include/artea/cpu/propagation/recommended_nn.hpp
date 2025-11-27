/*
 * @FilePath: /Artea/include/artea/cpu/propagation/recommended_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-27 11:41:52
 * @Date: 2025-11-02 21:18:08
 * @Description:
 */

/*
 * @FilePath: /Artea/include/artea/cpu/utils/random_seq.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-05 19:18:18
 * @Date: 2025-11-02 19:41:19
 * @Description: Refactored to use generation type as a class template parameter.
 */

#pragma once

#include <random>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/index/index_graph.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename derived_class
>
class RecommendedNN {

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;

public:
    RecommendedNN(const vertex_num_t num_vertices, const vertex_num_t recom_buf_size) :
        _num_vertices(num_vertices),
        _recom_buf_size(recom_buf_size),
        _ro_recom_buf(num_vertices)
    {
        for (auto& ro_recom_arr : _ro_recom_buf) {
            ro_recom_arr.reserve(recom_buf_size);
        }
    }

    ~RecommendedNN() = default;

    __attribute__((always_inline))
    auto append_edge(const vertex_id_t src, const nbr_t& nbr) -> void {
        static_cast<derived_class*>(this)->append_edge_impl(src, nbr);
    }

    __attribute__((always_inline))
    auto append_edge(const vertex_id_t src, const vertex_id_t dest, const distance_t dist) -> void {
        static_cast<derived_class*>(this)->append_edge_impl(src, dest, dist);
    }

    __attribute__((always_inline))
    auto get_recom_nbrs(const vertex_id_t src) -> nbr_arr_t& {
        return static_cast<derived_class*>(this)->get_recom_nbrs_impl(src);
    }

    /**
     * @brief Flush the recommendation buffer (implemented by the derived class, which supports
     * thread-safe concurrent access) to a read-only buffer.
     */
    __attribute__((always_inline))
    auto flush() -> void {
        static_cast<derived_class*>(this)->flush_impl();
    }

    __attribute__((always_inline))
    auto clear() -> void {
        static_cast<derived_class*>(this)->clear_impl();
    }

protected:

    /** @brief Number of vertices. */
    vertex_num_t _num_vertices;

    /** @brief Size of the append buffer. */
    vertex_num_t _recom_buf_size;

    /** @brief Read-only recommendation buffer for each vertex. */
    std::vector<nbr_arr_t> _ro_recom_buf;

};  // class RecommendedNN

}   // namespace cpu
}   // namespace artea