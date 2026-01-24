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

#include <artea/cpu/containers/allocator.hpp>

#pragma once

namespace artea {
namespace cpu {

/** ------ Forward Declaration  ------ **/
template <typename VertexGeneratorTraitsT> class OrthoLSHGenerator;
template <typename VertexGeneratorTraitsT> class LSHTable;

template <typename ComputerTraitsT, typename BufferTraitsT>
struct VertexGeneratorTraits : public ComputerTraitsT, public BufferTraitsT {

    /** ------ Self Traits ------ **/
    using vertex_generator_traits_t = VertexGeneratorTraits<ComputerTraitsT, BufferTraitsT>;

    /** @brief Ortho LSH generator. */
    using ortho_lsh_generator_t = OrthoLSHGenerator<vertex_generator_traits_t>;

    /** @brief LSH function table type. */
    using lsh_table_t = LSHTable<vertex_generator_traits_t>;

};  // struct VertexGeneratorTraits

}   // namespace cpu
}   // namespace artea