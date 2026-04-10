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
 * @FilePath: /Artea/include/artea/cpu/router/data_structures/candidate_entry_concept.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Concept describing the interface every candidate-entry type
 *               must satisfy so it can be carried by the templated candidate
 *               queues (StdCandidateQueue / LinearCandidateQueue /
 *               FHCandidateQueue).
 */

#pragma once

#include <concepts>
#include <type_traits>
#include <utility>

namespace artea {
namespace cpu {

/**
 * @brief Concept defining the interface every candidate-entry type must
 *        satisfy so it can be carried by the candidate queues.
 *
 * Two concrete entry types currently satisfy this concept:
 *   - @c DnbrCandidateEntry: 8-byte packed (vertex_id + explored bit +
 *     distance). Used by the dnbr/descent-graph router family.
 *   - @c LnbrCandidateEntry: 12-byte (layer_vid + explored bit + base_vid +
 *     distance). Used by the internal-graph router family, which needs to
 *     carry both @c base_vid and @c layer_vid through the beam-search queue
 *     without a side-channel lookup table.
 *
 * Design notes:
 *   - The concept deliberately says nothing about constructors. Different
 *     entry types accept different constructor argument lists; the candidate
 *     queues forward their @c try_push arguments into the entry via variadic
 *     perfect-forwarding, so the constructor signature is enforced at the
 *     call site (not by this concept).
 *   - @c get_base_id() returns the base-dataset identity (for recall).
 *   - @c get_layer_id() returns the per-graph dedup key handed to the
 *     visited table. For Dnbr, layer_id == base_id; for Lnbr they differ.
 *   - Ordering is distance-based. Comparators sort by @c get_distance().
 *
 * @tparam EntryT The candidate-entry type to check.
 */
template <typename EntryT>
concept CandidateEntry = requires(EntryT e, const EntryT ce) {

    // --- Nested type aliases ---
    typename EntryT::vertex_id_t;
    typename EntryT::distance_t;

    // --- Layer-local identity (dedup key for visited table / candidate queue) ---
    { ce.get_layer_id() } -> std::convertible_to<typename EntryT::vertex_id_t>;

    // --- Distance accessors ---
    { ce.get_distance() } -> std::convertible_to<typename EntryT::distance_t>;
    { e.set_distance(std::declval<typename EntryT::distance_t>()) }
        -> std::same_as<void>;

    // --- Explored / unexplored state (lazy-deletion support in queues) ---
    { ce.is_explored()       } -> std::convertible_to<bool>;
    { ce.is_unexplored()     } -> std::convertible_to<bool>;
    { ce.is_invalid()        } -> std::convertible_to<bool>;
    { e.mark_as_explored()   } -> std::same_as<void>;
    { e.mark_as_unexplored() } -> std::same_as<void>;

    // --- Sentinels (static factories) ---
    { EntryT::make_invalid_entry() } -> std::convertible_to<EntryT>;
    { EntryT::make_min_entry()     } -> std::convertible_to<EntryT>;

    // --- Distance-based ordering ---
    { ce <  ce } -> std::convertible_to<bool>;
    { ce >  ce } -> std::convertible_to<bool>;
    { ce == ce } -> std::convertible_to<bool>;
};

}   // namespace cpu
}   // namespace artea
