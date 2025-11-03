/*
 * @FilePath: /Artea/include/artea/cpu/speculative_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-02 21:20:17
 * @Date: 2025-10-23 10:54:01
 * @Description: 
 */

#pragma once

#include <cstdint>

#include <artea/types.hpp>
#include <artea/cpu/vector_array.hpp>

namespace artea {
namespace cpu {

template <
    typename vec_num_t, 
    typename vec_ele_t,
    typename vec_id_t = vec_num_t,
    typename distance_t = vec_ele_t
>
class SpeculativeNN {

public:
    SpeculativeNN() : _is_built(false) {}
    ~SpeculativeNN() = default;

    auto build(const VectorArray<vec_num_t, vec_ele_t, vec_dim>* vecs_array) -> void = 0;

    auto batch_update() -> void = 0;

    auto stream_update() -> void = 0;
    
    auto speculate(vec_id_t vec_id, vec_id_t* spec_nbrs, vec_num_t spec_nbrs_num) -> void = 0;

    __attribute__((always_inline))
    auto generate(vec_id_t vec_id, vec_id_t* spec_nbrs, vec_num_t spec_nbrs_num) {
        speculate(vec_id, spec_nbrs, spec_nbrs_num);
    }

protected:

    bool _is_built = false;

};  // class SpeculativeNN

}  // namespace cpu
}  // namespace artea