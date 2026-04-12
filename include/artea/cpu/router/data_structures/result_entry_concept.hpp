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
 * @FilePath: /Artea/include/artea/cpu/router/data_structures/result_entry_concept.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Concept for result entries returned by routers. Any type that
 *               exposes get_base_id() and get_distance() can be consumed by
 *               recall estimation and other result-processing utilities.
 */

#pragma once

#include <concepts>
#include <type_traits>

namespace artea {
namespace cpu {

/**
 * @brief Concept for router result entries.
 *
 * A ResultEntry must expose:
 *   - @c get_base_id()  — the base-dataset vertex identity.
 *   - @c get_distance() — the distance to the query.
 *
 * @c CandidateEntry satisfies this concept.
 */
template <typename EntryT>
concept ResultEntry = requires(const EntryT ce) {
    typename EntryT::vertex_id_t;
    typename EntryT::distance_t;

    { ce.get_base_id() }  -> std::convertible_to<typename EntryT::vertex_id_t>;
    { ce.get_distance() } -> std::convertible_to<typename EntryT::distance_t>;
};

}   // namespace cpu
}   // namespace artea
