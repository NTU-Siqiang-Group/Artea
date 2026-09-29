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

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>

using namespace artea::cpu;

static_assert(std::is_same_v<
    artea_graph::rgraph_config_t<DistanceMetricsT::EUCLIDEAN, 96>,
    stacked_rgraph::rgraph_config_t<DistanceMetricsT::EUCLIDEAN, 96>>,
    "ARTEA and stacked r-nets must share the same configuration type");

TEST(RGraphRadius, TauKSetsL1AndPreservesGeometricGrowth) {
    using config_t = artea_graph::rgraph_config_t<DistanceMetricsT::EUCLIDEAN, 96>;
    const config_t base(/*beta=*/2.0f, /*tau_k=*/0.0f, /*tau=*/0.0f, /*min_distance=*/0.5f, 16);
    EXPECT_FLOAT_EQ(base.tau_k(), 0.0f);
    EXPECT_FLOAT_EQ(base.radius_at(0), 0.5f);
    EXPECT_FLOAT_EQ(base.radius_at(1), 0.5f);
    EXPECT_FLOAT_EQ(base.radius_at(2), 1.0f);
    EXPECT_FLOAT_EQ(base.radius_at(3), 2.0f);

    const config_t expanded(2.0f, 3.0f, 0.0f, 0.5f, 16);
    EXPECT_FLOAT_EQ(expanded.tau_k(), 3.0f);
    EXPECT_FLOAT_EQ(expanded.radius_at(0), base.radius_at(0));
    EXPECT_FLOAT_EQ(expanded.radius_at(1), 2.0f);
    EXPECT_FLOAT_EQ(expanded.radius_at(2), 4.0f);
    EXPECT_FLOAT_EQ(expanded.radius_at(3), 8.0f);

    const config_t fractional(1.5f, 0.5f, 0.0f, 2.0f, 16);
    EXPECT_FLOAT_EQ(fractional.tau_k(), 0.5f);
    EXPECT_FLOAT_EQ(fractional.radius_at(0), 2.0f);
    EXPECT_FLOAT_EQ(fractional.radius_at(1), 3.0f);
    EXPECT_FLOAT_EQ(fractional.radius_at(2), 4.5f);
    EXPECT_FLOAT_EQ(fractional.radius_at(3), 6.75f);
    for (layer_num_t level = 1; level < 6; ++level) {
        EXPECT_FLOAT_EQ(fractional.radius_at(level + 1) / fractional.radius_at(level), 1.5f);
    }

    const config_t larger_beta(4.0f, 0.5f, 0.0f, 2.0f, 16);
    EXPECT_FLOAT_EQ(larger_beta.radius_at(0), fractional.radius_at(0));
    EXPECT_FLOAT_EQ(larger_beta.radius_at(1), fractional.radius_at(1));
    EXPECT_FLOAT_EQ(larger_beta.radius_at(2), 12.0f);
    EXPECT_FLOAT_EQ(larger_beta.radius_at(3), 48.0f);
}

TEST(RGraphRadius, RejectsInvalidParametersAndL1Overflow) {
    using config_t = artea_graph::rgraph_config_t<DistanceMetricsT::EUCLIDEAN, 96>;
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (const float beta : {0.0f, 0.9f, 1.0f, infinity, nan}) {
        EXPECT_THROW(config_t(beta, 0.0f, 0.0f, 0.5f, 16), std::runtime_error);
    }
    for (const float invalid : {-1.0f, infinity, nan}) {
        EXPECT_THROW(config_t(2.0f, invalid, 0.0f, 0.5f, 16), std::runtime_error);
        EXPECT_THROW(config_t(2.0f, 0.0f, invalid, 0.5f, 16), std::runtime_error);
    }
    for (const float invalid : {0.0f, -1.0f, infinity, nan}) {
        EXPECT_THROW(config_t(2.0f, 0.0f, 0.0f, invalid, 16), std::runtime_error);
    }
    EXPECT_THROW(config_t(2.0f, 0.0f, 0.0f, 0.5f, 0), std::runtime_error);
    EXPECT_THROW(config_t(2.0f, 0.0f, 0.0f, 0.5f, 16, 0), std::runtime_error);
    EXPECT_THROW(config_t(2.0f, 0.0f, 0.0f, 0.5f, 16, 16, 0), std::runtime_error);
    const float largest = std::numeric_limits<float>::max();
    EXPECT_THROW(config_t(2.0f, largest, 0.0f, 2.0f, 16), std::runtime_error);
    EXPECT_THROW(config_t(2.0f, 1.0f, 0.0f, largest, 16), std::runtime_error);
    // Beta scales only L2 and above; it cannot make the L1 radius overflow.
    const config_t largest_beta(largest, 1.0f, 0.0f, 1.0f, 16);
    EXPECT_FLOAT_EQ(largest_beta.radius_at(1), 2.0f);
    const config_t boundary(2.0f, 0.0f, 0.0f, largest, 16);
    EXPECT_FLOAT_EQ(boundary.radius_at(1), largest);
    // Compute in double so a large coefficient with a small distance scale remains valid.
    const config_t representable(2.0f, largest, 0.0f, 1e-20f, 16);
    EXPECT_TRUE(std::isfinite(representable.radius_at(1)));
    EXPECT_GT(representable.radius_at(1), 0.0f);
}

TEST(RGraphRadius, PruningShiftDoesNotAffectRadii) {
    using config_t = artea_graph::rgraph_config_t<DistanceMetricsT::EUCLIDEAN, 96>;
    const config_t original(2.0f, 0.5f, 0.0f, 0.5f, 16);
    for (const float shift : {0.0f, 0.5f, 1.0f, 2.0f, std::numeric_limits<float>::max()}) {
        const config_t config(2.0f, 0.5f, shift, 0.5f, 16);
        EXPECT_FLOAT_EQ(config.tau_k(), 0.5f);
        EXPECT_FLOAT_EQ(config.tau(), shift);
        for (layer_num_t level = 0; level < 6; ++level) {
            EXPECT_FLOAT_EQ(config.radius_at(level), original.radius_at(level));
        }
    }
}

TEST(RGraphRadius, L1MembershipDependsOnTauKNotBetaOrPruningShift) {
    constexpr auto metric = DistanceMetricsT::EUCLIDEAN;
    constexpr vec_dim_t dim = 96;
    dist_func_t<metric, dim> build_dist;
    const stacked_rgraph::pruning_config_t<metric, dim> pruning_config(1.0f, 0.0f);
    // R1=4 admits point 5 to L1 and covers point 9; R1=6 or 8 covers point 5 and admits point 9.
    struct Case { float beta; float tau_k; float shift; layer_id_t second; layer_id_t third; };
    for (const auto settings : {Case{2.0f, 0.0f, 0.0f, 1, 0}, Case{2.0f, 1.0f, 0.0f, 0, 1},
                               Case{4.0f, 0.0f, 0.0f, 1, 0}, Case{2.0f, 0.0f, 1.0f, 1, 0},
                               Case{2.0f, 0.5f, 0.0f, 0, 1}}) {
        SCOPED_TRACE(::testing::Message() << "beta=" << settings.beta << ", tau_k=" << settings.tau_k
                                         << ", shift=" << settings.shift);
        const stacked_rgraph::rgraph_config_t<metric, dim> config(
            settings.beta, settings.tau_k, settings.shift, 4.0f, 16);
        stacked_rgraph::index_t<metric, dim> graph(3, config, pruning_config);
        EXPECT_FLOAT_EQ(graph.tau_k(), settings.tau_k);
        EXPECT_FLOAT_EQ(graph.tau(), settings.shift);
        EXPECT_FLOAT_EQ(graph.radius_at(1), config.radius_at(1));
        // Separate batches make insertion order deterministic with parallel workers.
        for (const float coordinate : {0.0f, 5.0f, 9.0f}) {
            vector_array_t point(1, dim);
            std::fill_n(point.get(0), dim, 0.0f);
            point.get(0)[0] = coordinate;
            stacked_rgraph::factory_t<metric, dim>::add_vertices(
                graph, std::move(point), build_dist, /*insert_on_L0=*/false);
        }
        const auto& hierarchy = graph.get_hierarchical_graph();
        EXPECT_EQ(hierarchy.get_highest_level_id(0), 1);
        EXPECT_EQ(hierarchy.get_highest_level_id(1), settings.second);
        EXPECT_EQ(hierarchy.get_highest_level_id(2), settings.third);
    }
}

TEST(StageDistance, EuclideanAliasesSelectDifferentDistancesByStage) {
    std::array<float, 960> a{}, b{};
    for (const vec_dim_t dim : {96, 112, 128, 304, 384, 960}) {
        b.fill(0.0f);
        b.front() = 3.0f;
        b[dim - 1] = 4.0f;
        for (const char* name : {"euclidean", "l2", "euclidean_sqr", "l2_sqr"}) {
            SCOPED_TRACE(::testing::Message() << name << ", dim=" << dim);
            const DatasetInfra info{parse_metric(name), dim};
            const float build_distance = build_infra_dispatch(info, ARTEA_METRIC_LAMBDA(float) {
                EXPECT_EQ(Metric, DistanceMetricsT::EUCLIDEAN);
                EXPECT_EQ(Dim, dim);
                return dist_func_t<Metric, Dim>{}(a.data(), b.data());
            });
            const float search_distance = search_infra_dispatch(info, ARTEA_METRIC_LAMBDA(float) {
                EXPECT_EQ(Metric, DistanceMetricsT::EUCLIDEAN_SQR);
                EXPECT_EQ(Dim, dim);
                return dist_func_t<Metric, Dim>{}(a.data(), b.data());
            });
            EXPECT_FLOAT_EQ(build_distance, 5.0f);
            EXPECT_FLOAT_EQ(search_distance, 25.0f);
        }
    }
}

TEST(StageDistance, OtherMetricsAndExplicitDistanceDispatchKeepTheirSemantics) {
    for (const auto info : {DatasetInfra{DistanceMetricsT::COSINE, 112},
                           DatasetInfra{DistanceMetricsT::COSINE, 304},
                           DatasetInfra{DistanceMetricsT::DOT, 208}}) {
        build_infra_dispatch(info, ARTEA_METRIC_LAMBDA(void) {
            EXPECT_EQ(Metric, info.metric);
            EXPECT_EQ(Dim, info.dim);
        });
        search_infra_dispatch(info, ARTEA_METRIC_LAMBDA(void) {
            EXPECT_EQ(Metric, info.metric);
            EXPECT_EQ(Dim, info.dim);
        });
    }
    for (const auto metric : {DistanceMetricsT::EUCLIDEAN, DistanceMetricsT::EUCLIDEAN_SQR}) {
        infra_dispatch(DatasetInfra{metric, 128}, ARTEA_METRIC_LAMBDA(void) {
            EXPECT_EQ(Metric, metric);
        });
    }
}

TEST(StageDistance, EuclideanBuildCompactsAndSearchesWithSquaredDistances) {
    constexpr vec_dim_t dim = 96;
    constexpr vertex_num_t count = 768;  // Exercise refinement above min_layer_cap.
    constexpr vertex_num_t topk = 5;
    vector_array_t base(count, dim);
    for (vertex_id_t i = 0; i < count; ++i) {
        std::fill_n(base.get(i), dim, 0.0f);
        base.get(i)[0] = 3.0f * i;
        base.get(i)[1] = 0.4f * (i % 7);
    }
    const DatasetInfra info{DistanceMetricsT::EUCLIDEAN_SQR, dim};
    std::optional<compact::hierarchical_graph_t> compact_graph;
    build_infra_dispatch(info, ARTEA_METRIC_LAMBDA(void) {
        ASSERT_EQ(Metric, DistanceMetricsT::EUCLIDEAN);
        ASSERT_EQ(Dim, dim);
        dist_func_t<Metric, Dim> build_dist;
        artea_graph::rgraph_config_t<Metric, Dim> rgraph_config(
            2.0f, 0, 0.0f, 1.0f, 16, 16, 16, 8, 16);
        artea_graph::propagate_config_t<Metric, Dim> propagate_config(1, 1, 0.6f);
        artea_graph::pruning_config_t<Metric, Dim> pruning_config(1.1f, 0.0f);
        artea_graph::index_t<Metric, Dim> graph(
            count, rgraph_config, propagate_config, pruning_config);
        artea_graph::factory_t<Metric, Dim>::add_vertices(
            graph, base.extract_subset(0, count), build_dist,
            /*insert_on_L0=*/false, /*shuffle=*/false);

        // Check stored construction distances against an independent scalar L2
        // reference before compaction discards edge weights.
        const auto& dynamic_graph = graph.get_hierarchical_graph();
        ASSERT_NE(graph.top_occupied_level_id(), dynamic::hierarchical_graph_t::invalid_level_id);
        size_t num_edges = 0;
        for (vertex_id_t i = 0; i < count; ++i) {
            const auto highest_level = dynamic_graph.get_highest_level_id(i);
            ASSERT_NE(highest_level, dynamic::hierarchical_graph_t::invalid_level_id);
            for (layer_id_t level = 0; level <= highest_level; ++level) {
                for (const auto& neighbor : dynamic_graph.fetch_level_nbrs(i, level)) {
                    if (neighbor.is_invalid()) break;
                    const auto j = neighbor.get_vid();
                    const double dx = static_cast<double>(base.get(i)[0]) - base.get(j)[0];
                    const double dy = static_cast<double>(base.get(i)[1]) - base.get(j)[1];
                    const double expected = std::hypot(dx, dy);
                    EXPECT_NEAR(neighbor.get_distance(), expected, expected * 2e-6);
                    ++num_edges;
                }
            }
        }
        ASSERT_GT(num_edges, 0u);

        compact_graph.emplace(hierarchical_graph_compactor_t::compact_graph(
            dynamic_graph, base, build_dist));
        graph.release();
    });
    ASSERT_TRUE(compact_graph.has_value());

    search_infra_dispatch(info, ARTEA_METRIC_LAMBDA(void) {
        ASSERT_EQ(Metric, DistanceMetricsT::EUCLIDEAN_SQR);
        ASSERT_EQ(Dim, dim);
        dist_func_t<Metric, Dim> search_dist;
        // Explicit L2 reference router checks ordering independently of the search policy.
        dist_func_t<DistanceMetricsT::EUCLIDEAN, Dim> l2_dist;
        hierarchical_graph_router_t<DistanceMetricsT::EUCLIDEAN, Dim> l2_router(base, l2_dist, topk, count);
        hierarchical_graph_router_t<Metric, Dim> squared_router(base, search_dist, topk, count);
        l2_router.initialize();
        squared_router.initialize();

        for (const vertex_id_t anchor : {7u, 81u, 173u}) {
            std::array<float, dim> query{};
            query[0] = base.get(anchor)[0] + 0.75f;
            query[1] = base.get(anchor)[1] + 0.2f;
            const auto l2 = l2_router.template query<false>(query.data(), *compact_graph);
            const auto squared = squared_router.template query<false>(query.data(), *compact_graph);
            ASSERT_EQ(l2.size(), topk);
            ASSERT_EQ(squared.size(), topk);
            for (size_t k = 0; k < topk; ++k) {
                ASSERT_FALSE(squared[k].is_invalid());
                EXPECT_EQ(l2[k].get_vid(), squared[k].get_vid());
                const auto id = squared[k].get_vid();
                const double dx = static_cast<double>(query[0]) - base.get(id)[0];
                const double dy = static_cast<double>(query[1]) - base.get(id)[1];
                const double expected_squared = dx * dx + dy * dy;
                EXPECT_NEAR(squared[k].get_distance(), expected_squared, expected_squared * 2e-6);
                EXPECT_NEAR(l2[k].get_distance(), std::sqrt(expected_squared),
                            std::sqrt(expected_squared) * 2e-6);
            }
        }
    });
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
