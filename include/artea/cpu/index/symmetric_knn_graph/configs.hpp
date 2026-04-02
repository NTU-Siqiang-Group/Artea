/*
 * @FilePath: /Artea/include/artea/cpu/index/symmetric_knn_graph/configs.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration aliases for symmetric KNN graph construction.
 */

#pragma once

#include <artea/cpu/index/conv_graph/configs.hpp>

namespace artea {
namespace cpu {
namespace symmetric_knn_graph {

/** @brief Symmetric KNN graph uses the same PropagateConfig as conv_graph. */
template <typename BaseTraitsT>
using PropagateConfig = conv_graph::PropagateConfig<BaseTraitsT>;

/** @brief Symmetric KNN graph uses the same PruningConfig as conv_graph. */
template <typename BaseTraitsT>
using PruningConfig = conv_graph::PruningConfig<BaseTraitsT>;

}   // namespace symmetric_knn_graph
}   // namespace cpu
}   // namespace artea
