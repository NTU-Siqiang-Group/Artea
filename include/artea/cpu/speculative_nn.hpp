// speculative_nn.hpp

#pragma once

#include <cstdint>

#include <artea/types.hpp>
#include <artea/cpu/vector_array.hpp>

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
    
    auto speculate(vec_id_t vec_id) = 0;

protected:

    bool _is_built = false;

};  // class SpeculativeNN