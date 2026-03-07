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

#pragma once

namespace artea {
namespace cpu {

/**
 * @brief Base class for vertex generators using CRTP (Curiously Recurring Template Pattern).
 * @tparam VertexGeneratorTraitsT The vertex generator traits type.
 * @tparam DerivedClassT The derived class type (CRTP).
 */
template <typename VertexGeneratorTraitsT, typename DerivedClassT>
class VertexGenerator {

public:
    /**
     * @brief Generate method that delegates to the derived class's generate_impl.
     * This enables static polymorphism.
     */
    template <typename... Args>
    auto generate(Args&&... args) {
        return static_cast<DerivedClassT*>(this)->generate_impl(std::forward<Args>(args)...);
    }

};  // class VertexGenerator

}   // namespace cpu
}   // namespace artea

