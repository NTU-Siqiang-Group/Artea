/*
 * @FilePath: /Artea/include/artea/cpu/index/knn_graph/configs.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration aliases for KNN graph construction.
 */

#pragma once

#include <artea/cpu/index/conv_graph/configs.hpp>

namespace artea {
namespace cpu {
namespace knn_graph {

/** @brief KNN graph uses the same PropagateConfig as conv_graph. */
template <typename BaseTraitsT>
using PropagateConfig = conv_graph::PropagateConfig<BaseTraitsT>;

}   // namespace knn_graph
}   // namespace cpu
}   // namespace artea
