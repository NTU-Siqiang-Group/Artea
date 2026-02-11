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
 * @FilePath: /Artea/include/artea/cpu/framework/edge_generator_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

namespace artea {
namespace cpu {

// ----- Forward Declaration  ------ //
template <typename EdgeGeneratorTraitsT, typename DerivedClassT> class NeighborUpdater;
template <typename EdgeGeneratorTraitsT> class TriangleUpdater;
template <typename EdgeGeneratorTraitsT> class ReverseUpdater;
template <typename EdgeGeneratorTraitsT> class RandomUpdater;
template <typename EdgeGeneratorTraitsT> class RandomEG;

template <typename ComputerTraitsT, typename BufferTraitsT, typename IndexTraitsT>
struct EdgeGeneratorTraits :
    virtual public ComputerTraitsT,
    virtual public BufferTraitsT,
    virtual public IndexTraitsT
{

    /** ------ Self Traits ------ **/
    using edge_generator_traits_t = EdgeGeneratorTraits<ComputerTraitsT, BufferTraitsT, IndexTraitsT>;

    template <typename DerivedClassT>
    using neighbor_updater_t = NeighborUpdater<edge_generator_traits_t, DerivedClassT>;

    /** @brief Triangle updater. */
    using triangle_updater_t = TriangleUpdater<edge_generator_traits_t>;

    /** @brief Reverse edge updater. */
    using reverse_updater_t = ReverseUpdater<edge_generator_traits_t>;

    /** @brief Random neighbor updater. */
    using random_updater_t = RandomUpdater<edge_generator_traits_t>;

    /** @brief Random edge generator. */
    using random_eg_t = RandomEG<edge_generator_traits_t>;

};  // struct EdgeGeneratorTraits

}   // namespace cpu
}   // namespace artea
