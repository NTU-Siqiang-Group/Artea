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
 * @FilePath: /Artea/include/artea/cpu/utils/simd_distance_dispatcher.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Runtime-dim → static-dim adapter for SIMDDistance. Holds a
 *               std::variant of pre-compiled SIMDDistance<...,VecDim>
 *               alternatives over a fixed set of common dims. Only exposes
 *               the ctor (selects the alternative once) and variant()
 *               (lets callers do per-loop std::visit). No per-call
 *               operator() / fast_euclidean — every call site must wrap
 *               in std::visit at method/loop entry so dispatch is amortized
 *               over many distance calls.
 */

#pragma once

#include <cstddef>
#include <variant>
#include <fmt/format.h>
#include <artea/common/logger.hpp>
#include <artea/cpu/utils/simd_distance.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Variant-based dispatcher mapping a runtime @c vec_dim to one of a
 *        compile-time-fixed set of @c SIMDDistance<...,VecDim> alternatives.
 *
 *        Selects the alternative once at construction; thereafter callers
 *        access it via @c variant() and wrap their loops in @c std::visit.
 *        No per-call dispatch is provided — that pattern adds ~2 cycles per
 *        distance call without any kernel benefit. Per-loop dispatch
 *        amortizes the cost over all iterations.
 *
 *        Usage pattern at every call site:
 *        @code
 *          std::visit([&](const auto& simd_distance) {
 *              // Loop body using `simd_distance(p, q)` and/or
 *              // `simd_distance.fast_euclidean(p, q, p_norm)`.
 *              // The concrete SIMDDistance<...,VecDim> is fully inlined here.
 *          }, dispatcher.variant());
 *        @endcode
 *
 *        Extending the supported dim set is a two-line edit:
 *        1. Add @c simd_distance_t<D> to @c simd_distance_variant_t.
 *        2. Add @c case D: return simd_distance_t<D>{}; to @c _make_variant.
 *
 *  @tparam ComputerTraitsT  Trait chain (provides vec_ele_t, vec_dim_t, ...).
 *  @tparam UnrollSize       SIMD inner unroll factor; forwarded to all
 *                           variant alternatives.
 */
template <typename ComputerTraitsT, std::size_t UnrollSize = 1>
class SIMDDistanceDispatcher {

    using vec_dim_t  = typename ComputerTraitsT::vec_dim_t;

    template <std::size_t VecDim>
    using simd_distance_t = SIMDDistance<ComputerTraitsT, VecDim, UnrollSize>;

public:
    /**
     * @brief The list of supported static-dim SIMDDistance alternatives.
     *        Single source of truth for the supported dim set.
     */
    using simd_distance_variant_t = std::variant<
        simd_distance_t< 96>,
        simd_distance_t<100>,
        simd_distance_t<128>,
        simd_distance_t<300>,
        simd_distance_t<960>
    >;

    /**
     * @brief Select the variant alternative for @p vec_dim at construction.
     *        Unsupported dims hit ARTEA_ERROR with a message pointing at
     *        this file.
     */
    explicit SIMDDistanceDispatcher(vec_dim_t vec_dim)
      : _impl(_make_variant(vec_dim)) {}

    /**
     * @brief Access the underlying variant. Callers wrap their loop bodies
     *        in @c std::visit over this variant to obtain a concrete
     *        @c SIMDDistance<...,VecDim> inside the loop.
     */
    __attribute__((always_inline))
    auto variant() const -> const simd_distance_variant_t& { return _impl; }

    /**
     * @brief Sugar over @c std::visit + @c variant() — invoke @p body
     *        with the concrete @c SIMDDistance<...,VecDim> as its only
     *        argument. Pure-C++ alternative to the ARTEA_WITH_DIM macro
     *        pair (which is just a syntactic wrapper around this).
     *
     *        Example:
     *        @code
     *          dispatcher.dispatch([&](const auto& dist_func) {
     *              using DistFunc = std::decay_t<decltype(dist_func)>;
     *              bruteforce_router_t<router_traits_t, DistFunc> router(...);
     *              return router.batch_query(...);
     *          });
     *        @endcode
     */
    template <typename Body>
    __attribute__((always_inline))
    auto dispatch(Body&& body) const {
        return std::visit(std::forward<Body>(body), _impl);
    }

private:
    static auto _make_variant(vec_dim_t vec_dim) -> simd_distance_variant_t {
        switch (vec_dim) {
            case  96: return simd_distance_t< 96>{};
            case 100: return simd_distance_t<100>{};
            case 128: return simd_distance_t<128>{};
            case 300: return simd_distance_t<300>{};
            case 960: return simd_distance_t<960>{};
            default:
                ARTEA_ERROR(fmt::format(
                    "SIMDDistanceDispatcher: unsupported vec_dim {}. "
                    "Supported set: {{96, 100, 128, 300, 960}}. "
                    "Extend simd_distance_variant_t and _make_variant in "
                    "simd_distance_dispatcher.hpp to support more dims.",
                    vec_dim));
        }
    }

    simd_distance_variant_t _impl;
};

}   // namespace cpu
}   // namespace artea

// ----------------------------------------------------------------------
// Block-style sugar for the dispatcher visit pattern.
//
// Usage:
//   ARTEA_WITH_DIM(dispatcher, DistFunc, dist_func) {
//       // Inside the block:
//       //   `dist_func` is a const reference to the chosen
//       //     SIMDDistance<...,VecDim> instance.
//       //   `DistFunc`  is its type alias (== decltype of dist_func, decayed).
//       bruteforce_router_t<router_traits_t, DistFunc> router(base, dist_func, k);
//       auto results = router.batch_query(...);
//   } ARTEA_END_DIM(dispatcher);
//
// Notes:
//   - The block body lives inside a lambda. A bare `return` returns from
//     the lambda only — to propagate values out, capture by reference or
//     return from the lambda and assign at the call site.
//   - Throws inside the body propagate through std::visit as expected.
//   - For pure-C++ usage without macros, call dispatcher.dispatch(lambda)
//     directly — same code path.
// ----------------------------------------------------------------------

#define ARTEA_WITH_DIM(dispatcher_expr, dim_alias_name, dist_func_name)         \
    std::visit([&](const auto& dist_func_name) {                                \
        using dim_alias_name = std::decay_t<decltype(dist_func_name)>;

#define ARTEA_END_DIM(dispatcher_expr)                                          \
    }, (dispatcher_expr).variant())

