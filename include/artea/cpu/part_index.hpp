/*
 * @FilePath: /Artea/include/artea/cpu/part_index.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-05 16:15:24
 * @Date: 2025-11-01 17:04:26
 * @Description: 
 */

#pragma once

#include <artea/types.hpp>
#include <artea/cpu/array.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/speculative_nn.hpp>

namespace artea {
namespace cpu {

template <
    typename vec_num_t,
    typename vec_ele_t,
    typename distance_t = vec_ele_t,
    typename vec_id_t = vec_num_t
>
class PartMethod {

public:

private:
    

};  // class PartMethod

template <
    typename vec_num_t,
    typename vec_ele_t,
    typename distance_t = vec_ele_t,
    typename vec_id_t = vec_num_t
>
class PartIndex : public SpeculativeNN<vec_num_t, vec_ele_t>  {

public:
    PartIndex(part_num_t num_parts, vec_num_t num_vecs, vec_dim_t dim) : _num_parts(num_parts) {
        SpeculativeNN<vec_num_t, vec_ele_t>();
        _centroids = VectorArray<part_num_t, vec_ele_t>(num_parts, dim);
        _vid2pid = Array<part_id_t>(num_vecs);
    }

    ~PartIndex() = default;

    auto build(const VectorArray<vec_num_t, vec_ele_t, vec_dim>* vecs_array) -> void override {
        
    }

    auto speculate(vec_id_t vec_id) -> part_id_t override {
        
    }

private:
    part_num_t _num_parts;
    VectorArray<part_num_t, vec_ele_t> _centroids;
    Array<part_id_t> _vid2pid;
    
};  // class PartIndex

}   // namespace cpu
}   // namespace artea