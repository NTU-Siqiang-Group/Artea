/*
 * @FilePath: /Artea/include/artea/cpu/index/symmetric_knn_graph/index_structure.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Symmetric KNN graph index structure, reusing conv_graph::IndexStructure.
 */

#pragma once

#include <artea/cpu/index/conv_graph/index_structure.hpp>

namespace artea {
namespace cpu {
namespace symmetric_knn_graph {

template <typename IndexTraitsT>
using IndexStructure = conv_graph::IndexStructure<IndexTraitsT>;

}   // namespace symmetric_knn_graph
}   // namespace cpu
}   // namespace artea
