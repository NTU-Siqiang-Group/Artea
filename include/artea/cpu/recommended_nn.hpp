/*
 * @FilePath: /Artea/include/artea/cpu/recommended_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-09 17:07:03
 * @Date: 2025-11-02 21:18:08
 * @Description: 
 */

/*
 * @FilePath: /Artea/include/artea/cpu/random_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-05 19:18:18
 * @Date: 2025-11-02 19:41:19
 * @Description: Refactored to use generation type as a class template parameter.
 */

#pragma once

#include <random>
#include <stdexcept>
#include <type_traits>

#include <artea/types.hpp>
#include <artea/config.hpp>
#include <artea/cpu/array.hpp>
#include <artea/cpu/vector_array.hpp>

namespace artea {
namespace cpu {

template <typename vec_num_t>
class RecommendedNN {

public:

    using vec_id_t = vec_num_t;

    RecommendedNN(const vertex_num_t& recom_buf_size, bool enabled = false) : 
        _recom_buf_size(recom_buf_size) {}

    ~RecommendedNN() = default;

    virtual auto append_edge(const vertex_id_t& src, const nbr_t& nbr) -> void = 0;

    virtual auto get_recom_nbrs(const vertex_id_t& src) -> Array<nbr_t> = 0;

protected:

    /** @brief The size of the append buffer. */
    const vertex_num_t& _recom_buf_size;

};  // class RecommendedNN

}   // namespace cpu
}   // namespace artea