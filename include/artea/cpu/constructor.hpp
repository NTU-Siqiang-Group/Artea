/*
 * @FilePath: /Artea/include/artea/cpu/constructor.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-19 15:11:06
 * @Date: 2025-11-15 20:36:29
 * @Description:
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include <artea/cpu/index_graph.hpp>
#include <artea/cpu/recommended_nn.hpp>
#include <artea/cpu/random_seq.hpp>
#include <artea/cpu/vector_dataset.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/cpu/descent_engine.hpp>

#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    DistanceMetrics dist_type = DistanceMetrics::EUCLIDEAN
>
class GraphConstructor {

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;

public:
    GraphConstructor(const std::string& config_path, const std::string& dataset_name) :
        _dataset(config_path, dataset_name)
    {}


private:
    VectorDataset<vertex_num_t, vec_ele_t> _dataset;

};  // class GraphConstructor


}   // namespace cpu
}   // namespace artea