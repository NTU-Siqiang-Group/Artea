/*
 * @FilePath: /Artea/include/artea/cpu/spec_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-12 14:57:03
 * @Date: 2025-10-23 10:54:01
 * @Description:
 */

#pragma once

#include <cstdint>

#include <artea/definitions.hpp>
#include <artea/cpu/containers/vector_array.hpp>

namespace artea {
namespace cpu {

// TODO: reconstruct this class
template <typename vec_num_t, typename vec_ele_t>
class SpeculativeNN {

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;

public:
    SpeculativeNN(bool enabled = false) : _is_built(false) {}
    ~SpeculativeNN() = default;

    virtual auto build(const VectorArray<vec_num_t, vec_ele_t, vec_dim>* vecs_array) -> void = 0;

    virtual auto batch_update() -> void = 0;

    virtual auto stream_update() -> void = 0;

    /**
     * @brief Speculate the nearest neighbors of a vector.
     * @param vec_id The ID of the vector to speculate.
     * @param spec_nbrs The buffer to store the speculated nearest neighbors.
     * @param spec_nbrs_num The number of nearest neighbors to speculate.
     */
    virtual auto speculate(vec_id_t vec_id, vec_id_t* spec_nbrs, vec_num_t spec_nbrs_num) -> void = 0;

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

protected:

    bool _is_built = false;

};  // class SpeculativeNN

}  // namespace cpu
}  // namespace artea