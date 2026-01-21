// Copyright 2025 Weitang Ye
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

#pragma once

#include <cstddef>
#include <vector>
#include <utility>

// #include <artea/cpu/framework/base_traits.hpp>
// #include <artea/cpu/framework/buffer_traits.hpp>
// #include <artea/cpu/framework/computer_traits.hpp>

namespace artea {
namespace cpu {

/* ------ Forward Declarations ------ */
template <typename UpdaterTraitsT> class NeighborUpdater;

template <typename UpdaterTraitsT> class RNGUpdater;

/* ------ Updater Traits Definition ------ */

enum class updater_policy_t : uint8_t {
    RNG_UPDATER = 0
};  // enum class updater_policy_t

template <updater_policy_t UpdaterPolicy, typename UpdaterTraitsT>
struct UpdaterImplSelector;

template <typename UpdaterTraitsT>
struct UpdaterImplSelector<updater_policy_t::RNG_UPDATER, typename UpdaterTraitsT> {
    using type = RNGUpdater<UpdaterTraitsT>;
};

template <
    typename ComputerTraitsT,
    typename BufferTraitsT,
    updater_policy_t UpdaterPolicy
>
struct UpdaterTraits : public ComputerTraitsT, public BufferTraitsT {

public:

    using computer_traits_t = ComputerTraitsT;

    using buffer_traits_t = BufferTraitsT;

    using base_traits_t = typename ComputerTraitsT::base_traits_t;

    using updater_traits_t = UpdaterTraits<ComputerTraitsT, BufferTraitsT, UpdaterPolicy>;

    using updater_impl_t = typename UpdaterImplSelector<UpdaterPolicy, updater_traits_t>::type;

    static constexpr updater_policy_t updater_policy = UpdaterPolicy;

};  // struct UpdaterTraits

}   // namespace cpu
}   // namespace artea