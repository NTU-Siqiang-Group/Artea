/*
 * @FilePath: /Artea/include/artea/cpu/recommended_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-02 21:20:54
 * @Date: 2025-11-02 21:18:08
 * @Description: 
 */

#pragma once

#include <cstdint>

#include <artea/types.hpp>
#include <artea/config.hpp>
#include <artea/cpu/vector_array.hpp>

namespace artea {
namespace cpu {

template <
    typename vec_num_t, 
    typename vec_id_t = vec_num_t
>
class RecommendedNN {

public:
    /**
     * @brief Construct a new RecommendedNN object.
     * @param num_vecs The upper bound (exclusive) for the random numbers to be generated.
     */
    RecommendedNN(const vec_num_t& num_vecs) : _num_vecs(num_vecs) {
        
    }

private:
    const vec_num_t& _num_vecs;

};  // class RecommendedNN

}   // namespace cpu
}   // namespace artea