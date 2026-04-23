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

#include <concepts>
#include <cstddef>

namespace artea {
namespace cpu {

/**
 * @brief Concept defining the required interface for visited tables
 *        used in proximity graph search.
 *
 * A VisitedTable tracks which vertices have been visited during a single
 * beam search query. Implementations (e.g., ThreadLocalBitmap) must provide
 * bit-level set/test/clear operations.
 *
 * @tparam VisitedTableImpl The visited table type to check.
 */
template <typename VisitedTableImpl>
concept VisitedTable = requires(VisitedTableImpl table, const VisitedTableImpl const_table, std::size_t idx) {

    // Mark a vertex as visited
    { table.set(idx) } -> std::same_as<void>;

    // Check if a vertex has been visited
    { const_table.test(idx) } -> std::convertible_to<bool>;

    // Clear all visited marks
    { table.clear() } -> std::same_as<void>;
};

}   // namespace cpu
}   // namespace artea
