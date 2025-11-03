/*
 * @FilePath: /Artea/include/artea/cpu/stable_descent.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-02 21:16:51
 * @Date: 2025-11-02 08:39:50
 * @Description: 
 */

#pragma once

#include <artea/types.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/index_graph.hpp>
#include <artea/cpu/speculative_nn.hpp>
#include <artea/cpu/random_nn.hpp>

namespace artea {
namespace cpu {

class StableDescent {

public:
    StableDescent(
        IndexGraph<vec_num_t, vec_id_t>* index_graph, 
        SpeculativeNN<vec_num_t, vec_ele_t>* speculative_nn, 
        RandomNN<vec_num_t, vec_id_t>* random_nn
    ) : _index_graph(index_graph), _speculative_nn(speculative_nn), _random_nn(random_nn) {
        
    }

    void run(iter_t num_iters) {
        
    }

private:
    IndexGraph<vec_num_t, vec_id_t>* _index_graph;
    SpeculativeNN<vec_num_t, vec_ele_t>* _speculative_nn;
    RandomNN<vec_num_t, vec_id_t>* _random_nn;
    

};  // class StableDescent

}   // namespace cpu
}   // namespace artea
