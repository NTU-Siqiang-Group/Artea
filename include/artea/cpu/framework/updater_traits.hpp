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
 * @FilePath: /Artea/include/artea/cpu/framework/updater_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

namespace artea {
namespace cpu {

// ----- Forward Declaration  ------ //
template <typename UpdaterTraitsT, typename DerivedClassT> class NeighborUpdater;
template <typename UpdaterTraitsT> class RNGUpdater;

template <typename ComputerTraitsT, typename BufferTraitsT>
struct UpdaterTraits : public ComputerTraitsT, public BufferTraitsT {

    /** ------ Self Traits ------ **/
    using updater_traits_t = UpdaterTraits<ComputerTraitsT, BufferTraitsT>;

    template <typename DerivedClassT>
    using neighbor_updater_t = NeighborUpdater<updater_traits_t, DerivedClassT>;

    /** @brief RNG updater. */
    using rng_updater_t = RNGUpdater<updater_traits_t>;

};  // struct UpdaterTraits

}   // namespace cpu
}   // namespace artea