// speculative_nn.hpp

#pragma once

#include <cstdint>

#include <artea/types.hpp>
#include <artea/cpu/vector_array.hpp>

template <typename vec_num_t, typename vec_ele_t, vec_dim_t vec_dim>
class SpeculativeNN {

public:
    SpeculativeNN() = default;
    ~SpeculativeNN() = default;

    /**
     * @brief Build an approximate index.
     * 
     * @param vecs_array input vector array.
     */
    auto build(const VectorArray<vec_num_t, vec_ele_t, vec_dim>& vecs_array) -> void = 0;

    auto query(vec_ele_t* query_vec) -> vec_num_t = 0;

    auto query(vec_id_t vec_id) -> vec_num_t = 0;

    auto speculate(vec_id_t vec_id) = 0;

private:

};  // class SpeculativeNN