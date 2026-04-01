/*
 * @FilePath: /Artea/include/artea/cpu/index/artea_graph/configs.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration aliases for Artea graph construction.
 */

#pragma once

#include <artea/cpu/index/conv_graph/configs.hpp>

namespace artea {
namespace cpu {
namespace artea_graph {

/** @brief Artea graph uses the same PropagateConfig as conv_graph. */
template <typename BaseTraitsT>
using PropagateConfig = conv_graph::PropagateConfig<BaseTraitsT>;

/** @brief Artea graph uses the same PruningConfig as conv_graph. */
template <typename BaseTraitsT>
using PruningConfig = conv_graph::PruningConfig<BaseTraitsT>;

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
