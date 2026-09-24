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

#include <gtest/gtest.h>
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>

using namespace artea::cpu;

TEST(RGraphRadius, SkipLevelsSetL1AndPreserveGeometricGrowth) {
    using config_t = artea_graph::rgraph_config_t<DistanceMetricsT::EUCLIDEAN, 96>;
    const config_t config(/*beta=*/2.0f, /*num_skip_levels=*/0u, /*min_distance=*/0.5f, 16);
    EXPECT_EQ(config.num_skip_levels(), 0u);
    EXPECT_FLOAT_EQ(config.radius_at(0), 0.5f);
    EXPECT_FLOAT_EQ(config.radius_at(1), 1.0f);
    EXPECT_FLOAT_EQ(config.radius_at(2), 2.0f);
    EXPECT_FLOAT_EQ(config.radius_at(3), 4.0f);
    EXPECT_FLOAT_EQ(config.radius_at(4), 8.0f);

    const config_t larger_beta(4.0f, 0u, 0.5f, 16);
    EXPECT_FLOAT_EQ(larger_beta.radius_at(1), 2.0f);
    EXPECT_FLOAT_EQ(larger_beta.radius_at(2), 8.0f);
    EXPECT_FLOAT_EQ(larger_beta.radius_at(3), 32.0f);

    const config_t skipped(2.0f, 2u, 0.5f, 16);
    EXPECT_EQ(skipped.num_skip_levels(), 2u);
    EXPECT_FLOAT_EQ(skipped.radius_at(0), config.radius_at(0));
    EXPECT_FLOAT_EQ(skipped.radius_at(1), 4.0f);
    for (layer_num_t h = 1; h <= 4; ++h) {
        EXPECT_FLOAT_EQ(skipped.radius_at(h), config.radius_at(h + 2));
    }

    const config_t fractional_beta(1.5f, 3u, 2.0f, 16);
    EXPECT_FLOAT_EQ(fractional_beta.radius_at(1), 10.125f);
    EXPECT_FLOAT_EQ(fractional_beta.radius_at(2), 15.1875f);

    EXPECT_THROW(config_t(1.0f, 0u, 0.5f, 16), std::runtime_error);
    EXPECT_THROW(config_t(2.0f, 0u, 0.0f, 16), std::runtime_error);
    EXPECT_THROW(config_t(2.0f, std::numeric_limits<layer_num_t>::max(), 0.5f, 16), std::runtime_error);
    EXPECT_THROW(config_t(std::numeric_limits<float>::quiet_NaN(), 0u, 0.5f, 16), std::runtime_error);
}

TEST(RGraphRadius, L1MembershipChangesWithSkippedLevelsAndBeta) {
    constexpr auto metric = DistanceMetricsT::EUCLIDEAN;
    constexpr vec_dim_t dim = 96;
    dist_func_t<metric, dim> build_dist;
    const stacked_rgraph::pruning_config_t<metric, dim> pruning_config(1.0f, 0.0f);
    // With R1=4, points 0 and 5 enter L1 and cover point 9. With R1=8,
    // point 0 covers point 5, while point 9 must enter L1.
    struct Case { float beta; layer_num_t skips; layer_id_t second; layer_id_t third; };
    for (const auto c : {Case{2.0f, 0u, 1, 0}, Case{2.0f, 1u, 0, 1}, Case{4.0f, 0u, 0, 1}}) {
            SCOPED_TRACE(::testing::Message() << "beta=" << c.beta << ", skips=" << c.skips);
            const stacked_rgraph::rgraph_config_t<metric, dim> config(
                c.beta, c.skips, 2.0f, 16);
            stacked_rgraph::index_t<metric, dim> graph(3, config, pruning_config);
            EXPECT_EQ(graph.num_skip_levels(), c.skips);
            // Separate batches make the insertion order deterministic even
            // when construction uses parallel workers.
            for (const float x : {0.0f, 5.0f, 9.0f}) {
                vector_array_t point(1, dim);
                std::fill_n(point.get(0), dim, 0.0f);
                point.get(0)[0] = x;
                stacked_rgraph::factory_t<metric, dim>::add_vertices(
                    graph, std::move(point), build_dist, /*insert_on_L0=*/false);
            }
            const auto& hierarchy = graph.get_hierarchical_graph();
            EXPECT_EQ(hierarchy.get_highest_level_id(0), 1);
            EXPECT_EQ(hierarchy.get_highest_level_id(1), c.second);
            EXPECT_EQ(hierarchy.get_highest_level_id(2), c.third);
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
            2.0f, 2u, 1.0f, 16, 16, 16, 8, 16);
        artea_graph::propagate_config_t<Metric, Dim> propagate_config(1, 1, 0.6f);
        artea_graph::pruning_config_t<Metric, Dim> pruning_config(1.0f, 0.0f);
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
                for (const auto& neighbor : dynamic_graph.fetch_layer_nbrs(i, level)) {
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
            const auto l2 = l2_router.template query<false, false>(query.data(), *compact_graph);
            const auto squared = squared_router.template query<false, false>(query.data(), *compact_graph);
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
