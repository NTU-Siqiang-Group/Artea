// Copyright 2026 Weitang Ye
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * @FilePath: /Artea/include/artea/cpu/framework/vertex_generator_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
template <typename VertexGeneratorTraitsT, typename DerivedClassT> class VertexGenerator;
template <typename VertexGeneratorTraitsT> class OrthoLSHGenerator;
template <typename VertexGeneratorTraitsT> class PStableLSHGenerator;
template <typename VertexGeneratorTraitsT> class LSHTable;
template <typename VertexGeneratorTraitsT> class MBGreedyVG;
template <typename VertexGeneratorTraitsT> class LBGreedyVG;
template <typename VertexGeneratorTraitsT> class RandomVG;

template <typename ComputerTraitsT>
struct VertexGeneratorTraits : virtual public ComputerTraitsT {

    /** ------ Self Traits ------ **/
    using vertex_generator_traits_t = VertexGeneratorTraits<ComputerTraitsT>;

    /** @brief Vertex subset result type. */
    using vertex_subset_t = typename ComputerTraitsT::vertex_subset_t;

    /** @brief Approximate r-net result type (alias for vertex_subset_t). */
    using approx_rnet_t = vertex_subset_t;

    /** @brief Ortho LSH generator. */
    using ortho_lsh_generator_t = OrthoLSHGenerator<vertex_generator_traits_t>;

    /** @brief P-Stable LSH generator. */
    using pstable_lsh_generator_t = PStableLSHGenerator<vertex_generator_traits_t>;

    /** @brief LSH function table type. */
    using lsh_table_t = LSHTable<vertex_generator_traits_t>;

    /** @brief Mini batch greedy vertex generator. */
    using mb_greedy_vg_t = MBGreedyVG<vertex_generator_traits_t>;

    /** @brief Large batch greedy vertex generator. */
    using lb_greedy_vg_t = LBGreedyVG<vertex_generator_traits_t>;

    /** @brief Random vertex generator. */
    using random_vg_t = RandomVG<vertex_generator_traits_t>;

    template <typename DerivedClassT>
    using vertex_generator_t = VertexGenerator<vertex_generator_traits_t, DerivedClassT>;

};  // struct VertexGeneratorTraits

}   // namespace cpu
}   // namespace artea