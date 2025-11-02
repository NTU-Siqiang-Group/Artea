/*
 * @FilePath: /Artea/include/artea/cpu/part_index.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-02 15:02:19
 * @Date: 2025-11-01 17:04:26
 * @Description: 
 */

#pragma once

#include <artea/types.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/speculative_nn.hpp>

namespace artea {
namespace cpu {

template <
    typename vecs_num_t,
    typename vec_ele_t,
    typename distance_t = vec_ele_t,
    typename vec_id_t = vecs_num_t
>
class PartIndex : public SpeculativeNN<vecs_num_t, vec_ele_t, vec_dim>  {

public:
    PartIndex(part_num_t num_parts, vecs_num_t num_vecs, vec_dim_t dim) : _num_parts(num_parts) {
        SpeculativeNN<vecs_num_t, vec_ele_t, vec_dim>();
        _centroids = new VectorArray<part_num_t, vec_ele_t>(num_parts, dim);
        _vid2pid = new part_id_t[num_vecs];
    }

    ~PartIndex() {
        if (_centroids != nullptr)
            delete _centroids;
        if (_vid2pid != nullptr)
            delete[] _vid2pid;
        _centroids = nullptr;
        _vid2pid = nullptr;
    }

    auto build(const VectorArray<vecs_num_t, vec_ele_t, vec_dim>* vecs_array) -> void override {
        
    }

    auto speculate(vec_id_t vec_id) -> part_id_t override {
        
    }

private: 
    part_num_t _num_parts;
    VectorArray<part_num_t, vec_ele_t>* _centroids;
    part_id_t* _vid2pid;
    
};  // class PartIndex   

}   // namespace cpu
}   // namespace artea