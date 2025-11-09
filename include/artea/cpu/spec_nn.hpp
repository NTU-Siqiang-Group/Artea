/*
 * @FilePath: /Artea/include/artea/cpu/speculative_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-05 19:37:57
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
    SpeculativeNN(bool enabled = false) : _is_built(false), _enabled(enabled) {}
    ~SpeculativeNN() = default;

    auto build(const VectorArray<vec_num_t, vec_ele_t, vec_dim>* vecs_array) -> void = 0;

    auto batch_update() -> void = 0;

    auto stream_update() -> void = 0;
    
    /**
     * @brief Speculate the nearest neighbors of a vector.
     * @param vec_id The ID of the vector to speculate.
     * @param spec_nbrs The buffer to store the speculated nearest neighbors.
     * @param spec_nbrs_num The number of nearest neighbors to speculate.
     */
    auto speculate(vec_id_t vec_id, vec_id_t* spec_nbrs, vec_num_t spec_nbrs_num) -> void = 0;

    /**
     * @brief Generate the nearest neighbors of a vector.
     * @param vec_id The ID of the vector to generate.
     * @param spec_nbrs The buffer to store the generated nearest neighbors.
     * @param spec_nbrs_num The number of nearest neighbors to generate.
     */
    __attribute__((always_inline))
    auto generate(vec_id_t vec_id, vec_id_t* spec_nbrs, vec_num_t spec_nbrs_num) -> void {
        speculate(vec_id, spec_nbrs, spec_nbrs_num);
    }

    /**
    * @brief Check if the speculative nearest neighbors generator is enabled.
    * @return true If the speculative nearest neighbors generator is enabled.
    * @return false If the speculative nearest neighbors generator is disabled.
    */
    __attribute__((always_inline))
    auto enabled() const -> bool {
        return _enabled;
    }

    /**
    * @brief Disable the speculative nearest neighbors generator.
    */
    __attribute__((always_inline))
    auto disable() -> void {
        _enabled = false;
    }

    /**
    * @brief Activate the speculative nearest neighbors generator.
    */
    __attribute__((always_inline))
    auto activate() -> void {
        _enabled = true;
    }

protected:

    bool _is_built = false;
    bool _enabled = false;

};  // class SpeculativeNN

}  // namespace cpu
}  // namespace artea