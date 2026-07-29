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
 * @FilePath: /Artea/include/artea/cpu/framework/type_context/infra_dispatcher.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Metric/dim-aware infra dispatcher.
 *
 *   Only ComputerTraits-dependent infra is dispatched here (everything that is
 *   parameterized by <Metric, Dim>). An `InfraKind` enum names each such type;
 *   `infra_t<Metric, Dim, Kind>` resolves the type and `acquire<Metric, Dim, Kind>(...)`
 *   constructs it. The kind->type mapping is realized by partial specializations
 *   of InfraResolver (function templates can't be partially specialized).
 *
 *   The two compile-time axes (distance metric + SIMD-padded dimension) are
 *   supplied by the caller: the metric is parsed from input (workload `metric`
 *   field / CLI flag) via `parse_metric()`, the padded dim is read off the
 *   loaded dataset. Both are then resolved together by a single
 *   `infra_dispatch(info, fn)` trampoline that calls `fn.template operator()<Metric, Dim>()`.
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <utility>

#include <fmt/format.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

namespace artea {
namespace cpu {

/** @brief Names each ComputerTraits-dependent infra type the dispatcher can
 *         hand out. Metric/dim-independent types (vector_dataset_t, the
 *         compactor, the compact graph, ...) are NOT listed — they are used
 *         directly without a metric/dim. */
enum class InfraKind : uint8_t {
    SimdDistance,
    SimdLinear,
    DistanceProber,
    RecallEstimator,
    AdrEstimator,
    SingleLayerRouter,
    HierRouter,
    ArteaIndex,
    ArteaFactory,
    ArteaRGraphConfig,
    ArteaPropagateConfig,
    ArteaPruningConfig,
    // extend as new ComputerTraits-dependent infra is needed
};

/** @brief Primary template — undefined; each InfraKind is specialized below.
 *
 *  Each specialization exposes `type` (the infra type for <Metric, Dim>) and a
 *  perfect-forwarding `make(args...)` factory. `make` is itself a template, so
 *  it is only instantiated when actually called — kinds used purely for their
 *  static API (e.g. ArteaFactory) never force a constructor instantiation. */
template <DistanceMetricsT Metric, vec_dim_t Dim, InfraKind Kind>
struct InfraResolver;

// For aliases that live directly in namespace artea::cpu.
#define ARTEA_DEFINE_INFRA_RESOLVER(KIND, ALIAS)                              \
    template <DistanceMetricsT Metric, vec_dim_t Dim>                            \
    struct InfraResolver<Metric, Dim, InfraKind::KIND> {                        \
        using type = ALIAS<Metric, Dim>;                                        \
        template <typename... Args>                                         \
        static type make(Args&&... args) {                                 \
            return type(std::forward<Args>(args)...);                      \
        }                                                                   \
    }

ARTEA_DEFINE_INFRA_RESOLVER(SimdDistance,      dist_func_t);
ARTEA_DEFINE_INFRA_RESOLVER(SimdLinear,        linear_func_t);
ARTEA_DEFINE_INFRA_RESOLVER(DistanceProber,    distance_prober_t);
ARTEA_DEFINE_INFRA_RESOLVER(RecallEstimator,   recall_estimator_t);
ARTEA_DEFINE_INFRA_RESOLVER(AdrEstimator,      adr_estimator_t);
ARTEA_DEFINE_INFRA_RESOLVER(SingleLayerRouter, single_layer_router_t);
ARTEA_DEFINE_INFRA_RESOLVER(HierRouter,        hierarchical_graph_router_t);

#undef ARTEA_DEFINE_INFRA_RESOLVER

// For aliases that live in the artea_graph nested namespace.
#define ARTEA_DEFINE_INFRA_RESOLVER_AG(KIND, ALIAS)                           \
    template <DistanceMetricsT Metric, vec_dim_t Dim>                            \
    struct InfraResolver<Metric, Dim, InfraKind::KIND> {                        \
        using type = artea_graph::ALIAS<Metric, Dim>;                          \
        template <typename... Args>                                         \
        static type make(Args&&... args) {                                 \
            return type(std::forward<Args>(args)...);                      \
        }                                                                   \
    }

ARTEA_DEFINE_INFRA_RESOLVER_AG(ArteaIndex,           index_t);
ARTEA_DEFINE_INFRA_RESOLVER_AG(ArteaFactory,         factory_t);
ARTEA_DEFINE_INFRA_RESOLVER_AG(ArteaRGraphConfig,    rgraph_config_t);
ARTEA_DEFINE_INFRA_RESOLVER_AG(ArteaPropagateConfig, propagate_config_t);
ARTEA_DEFINE_INFRA_RESOLVER_AG(ArteaPruningConfig,   pruning_config_t);

#undef ARTEA_DEFINE_INFRA_RESOLVER_AG

/** @brief Type access (no construction): infra_t<Metric, Dim, Kind>. */
template <DistanceMetricsT Metric, vec_dim_t Dim, InfraKind Kind>
using infra_t = typename InfraResolver<Metric, Dim, Kind>::type;

/** @brief Object construction: acquire<Metric, Dim, Kind>(ctor-args...). */
template <DistanceMetricsT Metric, vec_dim_t Dim, InfraKind Kind, typename... Args>
auto acquire(Args&&... args) -> infra_t<Metric, Dim, Kind> {
    return InfraResolver<Metric, Dim, Kind>::make(std::forward<Args>(args)...);
}

/** @brief The two compile-time axes resolved together by infra_dispatch(): the
 *         distance metric (caller-supplied, parsed from input) and the
 *         SIMD-padded vector dimension (read off the loaded dataset). */
struct DatasetInfra {
    DistanceMetricsT metric;
    vec_dim_t        dim;     // SIMD-padded dimension (ceil(orig/16)*16)
};

/** @brief Parse a distance-metric name (workload `metric` field / CLI flag)
 *         into its DistanceMetricsT. Canonical spellings are 'euclidean',
 *         'inner_product', and 'cosine'; common aliases are accepted. Throws
 *         on an unrecognized name. */
inline auto parse_metric(std::string_view name) -> DistanceMetricsT {
    if (name == "euclidean" || name == "l2") {
        return DistanceMetricsT::EUCLIDEAN;
    }
    if (name == "inner_product" || name == "dot" || name == "ip" || name == "mips") {
        return DistanceMetricsT::DOT;
    }
    if (name == "cosine" || name == "angular") {
        return DistanceMetricsT::COSINE;
    }
    ARTEA_ERROR(fmt::format("Unknown distance metric '{}': expected 'euclidean', "
                            "'inner_product', or 'cosine'", name));
}

/** @brief Canonical display name for a metric (the workload spelling). */
inline auto metric_name(DistanceMetricsT metric) -> const char* {
    switch (metric) {
        case DistanceMetricsT::EUCLIDEAN: return "euclidean";
        case DistanceMetricsT::DOT:       return "inner_product";
        case DistanceMetricsT::COSINE:    return "cosine";
    }
    return "unknown";
}

/** @brief Resolve a dataset's (metric, dim) to the compile-time pair <Metric, Dim>
 *         in one shot and invoke a generic callable:
 *             fn.template operator()<Metric, Dim>()
 *
 *  Both axes are dispatched together (a single entry point) so downstream code
 *  never juggles two separate trampolines. Each (metric, dim) case instantiates
 *  one full <Metric, Dim> stack in the calling TU, so keep the dim lists tight —
 *  they cover exactly the SIMD-padded dimensions of the benchmarked datasets.
 *  An unsupported (metric, dim) throws; add the padded dim under its metric
 *  here to support it. */
template <typename Fn>
decltype(auto) infra_dispatch(DatasetInfra info, Fn&& fn) {
    switch (info.metric) {
        case DistanceMetricsT::COSINE:
            switch (info.dim) {
                case 112: return fn.template operator()<DistanceMetricsT::COSINE, vec_dim_t{112}>();
                case 304: return fn.template operator()<DistanceMetricsT::COSINE, vec_dim_t{304}>();
                default: break;
            }
            break;
        case DistanceMetricsT::EUCLIDEAN:
            switch (info.dim) {
                case  96: return fn.template operator()<DistanceMetricsT::EUCLIDEAN, vec_dim_t{ 96}>();
                case 112: return fn.template operator()<DistanceMetricsT::EUCLIDEAN, vec_dim_t{112}>();
                case 128: return fn.template operator()<DistanceMetricsT::EUCLIDEAN, vec_dim_t{128}>();
                case 304: return fn.template operator()<DistanceMetricsT::EUCLIDEAN, vec_dim_t{304}>();
                case 960: return fn.template operator()<DistanceMetricsT::EUCLIDEAN, vec_dim_t{960}>();
                default: break;
            }
            break;
        case DistanceMetricsT::DOT:
            switch (info.dim) {
                case 208: return fn.template operator()<DistanceMetricsT::DOT, vec_dim_t{208}>();  // yandex-t2i-10m
                default: break;
            }
            break;
        default: break;
    }
    ARTEA_ERROR(fmt::format("Unsupported (metric={}, dim={}) in infra_dispatch(): add the "
                            "padded dim under its metric in infra_dispatcher.hpp",
                            metric_name(info.metric), info.dim));
}

}   // namespace cpu
}   // namespace artea

// Packages only the verbose generic-lambda header for infra_dispatch(). `infra_dispatch(`,
// its `)`, and the body braces are written explicitly at the call site, so
// every bracket in the source stays matched (no hidden parens):
//     return infra_dispatch(info, ARTEA_METRIC_LAMBDA(int) { ...Metric, Dim...; return n; });
//     infra_dispatch(info, ARTEA_METRIC_LAMBDA(void) { ...Metric, Dim...; });
// Inside the body `DistanceMetricsT Metric` and `vec_dim_t Dim` are in scope. The
// return type is variadic so it may itself contain commas, e.g.
//     infra_dispatch(info, ARTEA_METRIC_LAMBDA(std::pair<A, B>) { ... });
#define ARTEA_METRIC_LAMBDA(...)                                               \
    [&]<::artea::cpu::DistanceMetricsT Metric, ::artea::cpu::vec_dim_t Dim>() -> __VA_ARGS__
