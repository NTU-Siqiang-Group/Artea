// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>
#include <artea/cpu/framework/type_context/default_context.hpp>

namespace {
using namespace artea::cpu;
constexpr auto metric = DistanceMetricsT::EUCLIDEAN;
constexpr unsigned dim = 16;
using Traits = graph_factory_traits_t<metric, dim>;
using Index = Traits::exact_artea::index_t;
using Factory = Traits::exact_artea::factory_t;
using RNets = Traits::exact_artea::rnets_factory_t;
using Config = Traits::exact_artea::rgraph_config_t;
using Pruning = Traits::exact_artea::pruning_config_t;
const dist_func_t<metric, dim> distance;

static_assert(!std::is_base_of_v<Traits::stacked_rgraph::index_t, Index>);
static_assert(!std::is_base_of_v<Traits::stacked_rgraph::factory_t, Factory>);
static_assert(std::is_same_v<Config, Traits::artea_graph::rgraph_config_t>);

auto config(uint32_t old_capacity = 1) -> Config {
    return Config(2, 1, 1, 1, 16, 16, 16, old_capacity, old_capacity);
}
const Pruning pruning(2, 1, 1);

auto grid(unsigned width) -> vector_array_t {
    vector_array_t data(width * width, dim);
    for (uint32_t id = 0; id < data.get_num_vecs(); ++id) {
        std::fill_n(data.get(id), dim, 0.0f);
        data.get(id)[0] = id % width;
        data.get(id)[1] = id / width;
    }
    return data;
}

auto neighbor_ids(std::span<const nbr_t> neighbors) -> std::vector<uint32_t> {
    std::vector<uint32_t> result;
    for (const auto& neighbor : neighbors) result.push_back(neighbor.get_vid());
    return result;
}

auto verify_merged_rows(const Index& index) -> void {
    const auto& graph = index.get_hierarchical_graph();
    EXPECT_EQ(graph.get_num_vertices(), index.get_num_vertices());
    EXPECT_EQ(graph.top_occupied_level_id(), index.top_occupied_level_id());
    EXPECT_EQ(graph.entry_point_vid(), index.entry_point_vid());
    std::size_t total = 0;
    for (uint32_t vid = 0; vid < index.get_num_vertices(); ++vid) {
        ASSERT_EQ(graph.get_highest_level_id(vid), index.get_highest_level_id(vid));
        for (uint32_t h = 0; h <= index.get_highest_level_id(vid); ++h) {
            const auto expected = neighbor_ids(index.fetch_level_nbrs(vid, h));
            const auto actual = graph.fetch_level_nbrs(vid, h);
            EXPECT_EQ((std::vector<uint32_t>(actual.begin(), actual.end())), expected);
            total += expected.size();
        }
    }
    EXPECT_EQ(graph.neighbor_ids().size(), total);
}

TEST(ExactArteaBuild, PerLayerRefiningGraphsGrowAndMergeWithoutTrimmingOrTruncation) {
    auto data = grid(4);
    Index index(data, config(), pruning);
    EXPECT_FALSE(index.has_layers());
    EXPECT_THROW(index.get_hierarchical_graph(), std::logic_error);
    RNets::fps_generator(index, distance);
    ASSERT_GT(index.get_num_layers(), 1u);
    ASSERT_EQ(index.get_top_level_vids().size(), 1u);
    for (uint32_t h = 0; h < index.get_num_layers(); ++h) {
        auto& layer = index.get_layer_graph(h);
        EXPECT_EQ(&layer.get_vecs_data(), &data);
        const auto vids = index.layer_vids(h);
        EXPECT_EQ(layer.get_num_vertices(), vids.size());
        for (const auto vid : vids) {
            auto& row = layer.fetch_nbrs(vid);
            // Reverse ID order deliberately: the merge must preserve the source order.
            for (auto it = vids.rbegin(); it != vids.rend(); ++it) {
                if (*it != vid) row.emplace_back(*it, distance(data.get(vid), data.get(*it)), true);
            }
            EXPECT_EQ(row.size(), vids.size() - 1);
        }
    }
    Factory::merge_layers(index);
    verify_merged_rows(index);
    const auto& graph = index.get_hierarchical_graph();
    EXPECT_EQ(graph.fetch_level_nbrs(0, 0).size(), 15u);
    EXPECT_GT(graph.fetch_level_nbrs(0, 0).size(), index.rgraph_config().bl_max_nbr_size());
    EXPECT_EQ(graph.get_top_level_vids().size(), 1u);
    EXPECT_TRUE(graph.fetch_level_nbrs(index.entry_point_vid(), index.top_occupied_level_id()).empty());
}

TEST(ExactArteaBuild, ConstructsEachLayerFromOnlyItsOwnVerticesAndKeepsFullPrunedRows) {
    auto data = grid(4);
    Index index(data, config(), pruning);
    RNets::fps_generator(index, distance);
    for (uint32_t h = 0; h < index.get_num_layers(); ++h) {
        Factory::build_layer(index, h, distance);
        for (const auto vid : index.layer_vids(h)) {
            const auto expected = Factory::prune_arc_candidates(index, vid, h, index.layer_vids(h), distance);
            const auto actual = index.fetch_level_nbrs(vid, h);
            ASSERT_EQ(actual.size(), expected.size());
            for (std::size_t i = 0; i < actual.size(); ++i) {
                EXPECT_EQ(actual[i].get_vid(), expected[i].get_vid());
                EXPECT_EQ(actual[i].get_distance(), expected[i].get_distance());
                EXPECT_TRUE(actual[i].is_old());
                EXPECT_GE(index.get_highest_level_id(actual[i].get_vid()), h);
            }
        }
    }
    EXPECT_GT(index.fetch_level_nbrs(0, 0).size(), 1u);
    Factory::merge_layers(index);
    verify_merged_rows(index);
}

TEST(ExactArteaBuild, FullBuildHandlesEmptySingletonAndRepeatedBuildRequiresReset) {
    for (uint32_t count : {0u, 1u}) {
        vector_array_t data(count, dim);
        if (count) std::fill_n(data.get(0), dim, 0.0f);
        Index index(data, config(), pruning);
        Factory::build(index, distance);
        EXPECT_EQ(index.get_num_layers(), 1u);
        EXPECT_EQ(index.get_num_vertices(), count);
        EXPECT_TRUE(index.has_hierarchical_graph());
        verify_merged_rows(index);
        EXPECT_EQ(index.entry_point_vid(), count ? 0u : base_traits_t::invalid_vertex_id);
        EXPECT_THROW(Factory::build(index, distance), std::logic_error);
        index.prepare_build(config(), pruning);
        Factory::build(index, distance);
        verify_merged_rows(index);
    }
}

TEST(ExactArteaBuild, RetainsAllSparseLevelsIncludingSingletonApex) {
    vector_array_t data(2, dim);
    std::fill_n(data.get_all(), 2 * dim, 0.0f);
    data.get(1)[0] = 1024;
    Index index(data, Config(2, 0, 0, 1, 4, 4, 4, 1, 1), Pruning(2, 0, 1));
    Factory::build(index, distance);
    EXPECT_EQ(index.get_num_layers(), 12u);
    EXPECT_EQ(index.get_hierarchical_graph().top_occupied_level_id(), 11u);
    EXPECT_EQ(index.get_hierarchical_graph().get_highest_level_id(0), 11u);
    EXPECT_EQ(index.get_hierarchical_graph().get_highest_level_id(1), 10u);
    verify_merged_rows(index);
}

TEST(ExactArteaBuild, ThreadLimitsAndLegacyDegreeSettingsDoNotChangeExactTopology) {
    auto data = grid(9);
    std::vector<std::vector<uint32_t>> reference_rows;
    std::vector<uint32_t> reference_heights;
    for (uint32_t capacity : {1u, 128u}) {
        for (std::size_t threads : {1u, 8u}) {
            tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
            Index index(data, config(capacity), pruning);
            Factory::build(index, distance);
            verify_merged_rows(index);
            std::vector<std::vector<uint32_t>> rows;
            std::vector<uint32_t> heights;
            for (uint32_t vid = 0; vid < data.get_num_vecs(); ++vid) {
                heights.push_back(index.get_highest_level_id(vid));
                for (uint32_t h = 0; h <= heights.back(); ++h) rows.push_back(neighbor_ids(index.fetch_level_nbrs(vid, h)));
            }
            if (reference_rows.empty()) {
                reference_rows = rows;
                reference_heights = heights;
            } else {
                EXPECT_EQ(rows, reference_rows);
                EXPECT_EQ(heights, reference_heights);
            }
        }
    }
}

TEST(ExactArteaBuild, LayerMutationInvalidatesSnapshotAndMergeRejectsInvalidNeighbors) {
    auto data = grid(4);
    Index index(data, config(), pruning);
    Factory::build(index, distance);
    const auto h = index.top_occupied_level_id();
    auto& top_row = index.get_layer_graph(h).fetch_nbrs(index.entry_point_vid());
    EXPECT_FALSE(index.has_hierarchical_graph());
    uint32_t lower_vertex = 0;
    while (index.get_highest_level_id(lower_vertex) >= h) ++lower_vertex;
    top_row.emplace_back(lower_vertex, 1, true);
    EXPECT_THROW(Factory::merge_layers(index), std::invalid_argument);
    EXPECT_EQ(top_row.size(), 1u);
    EXPECT_FALSE(index.has_hierarchical_graph());
    top_row.clear();
    Factory::merge_layers(index);
    verify_merged_rows(index);
    auto& bottom_row = index.get_layer_graph(0).fetch_nbrs(0);
    bottom_row.push_back(nbr_t::make_invalid_nbr());
    EXPECT_THROW(Factory::merge_layers(index), std::invalid_argument);
    EXPECT_TRUE(bottom_row.back().is_invalid());
}

struct ThrowingComputer : computer_traits_t<metric, dim> {
    struct dist_func_t {
        auto operator()(const float*, const float*) const -> float {
            throw std::runtime_error("Injected distance failure");
        }
    };
};
using ThrowingRouter = RouterTraits<ThrowingComputer, index_traits_t>;
using ThrowingRefiner = RefinerTraits<ThrowingComputer, buffer_traits_t, index_traits_t, ThrowingRouter>;
using ThrowingFactory = GraphFactoryTraits<ThrowingRefiner, ThrowingRouter>::exact_artea::factory_t;

TEST(ExactArteaBuild, FailedLayerBuildPreservesRowsAndPublishedSnapshot) {
    auto data = grid(4);
    Index index(data, config(), pruning);
    Factory::build(index, distance);
    const auto* snapshot = &index.get_hierarchical_graph();
    const auto before = neighbor_ids(index.fetch_level_nbrs(0, 0));
    EXPECT_THROW(ThrowingFactory::build_layer(index, 0, ThrowingComputer::dist_func_t{}), std::runtime_error);
    EXPECT_EQ(&index.get_hierarchical_graph(), snapshot);
    EXPECT_EQ(neighbor_ids(index.fetch_level_nbrs(0, 0)), before);
    verify_merged_rows(index);
}

TEST(ExactArteaBuild, ExistingRouterQueriesMergedAndPerLayerGraphsWithOptionalEarlyStop) {
    auto data = grid(4);
    Index index(data, config(), pruning);
    Factory::build(index, distance);
    hierarchical_graph_router_t<metric, dim> router(data, distance, 3, 16);
    router.initialize();
    for (const std::array<float, 2> xy : {std::array{0.31f, 0.63f}, std::array{1.7f, 2.2f}, std::array{2.51f, 0.17f}}) {
        alignas(64) std::array<float, dim> query{};
        query[0] = xy[0];
        query[1] = xy[1];
        std::vector<uint32_t> expected(data.get_num_vecs());
        std::iota(expected.begin(), expected.end(), 0u);
        std::sort(expected.begin(), expected.end(), [&](auto a, auto b) {
            const auto da = distance(query.data(), data.get(a));
            const auto db = distance(query.data(), data.get(b));
            return da < db || (da == db && a < b);
        });
        const auto compact = router.query(query.data(), index.get_hierarchical_graph());
        const auto layered = router.query(query.data(), index);
        const auto early = router.query<false, true>(query.data(), index.get_hierarchical_graph(),
                                                    index.rgraph_config(), index.pruning_config());
        ASSERT_EQ(compact.size(), 3u);
        ASSERT_EQ(layered.size(), 3u);
        ASSERT_EQ(early.size(), 3u);
        for (unsigned i = 0; i < 3; ++i) {
            EXPECT_EQ(compact[i].get_vid(), expected[i]);
            EXPECT_EQ(layered[i].get_vid(), expected[i]);
            EXPECT_EQ(early[i].get_vid(), expected[i]);
        }
    }
}

TEST(ExactArteaBuild, OwnedDatasetAndExplicitRebuildKeepVectorStorageAlive) {
    auto dataset = std::make_unique<vector_dataset_t>();
    dataset->get_base_vecs() = grid(4);
    const auto* vectors = &dataset->get_base_vecs();
    Index index(std::move(dataset));
    EXPECT_THROW(Factory::build(index, distance), std::logic_error);
    EXPECT_THROW(Factory::build_layers(index, distance), std::logic_error);
    EXPECT_THROW(Factory::merge_layers(index), std::logic_error);
    index.prepare_build(config(), pruning);
    Factory::build(index, distance);
    EXPECT_TRUE(index.owns_dataset());
    EXPECT_EQ(&index.get_base_vecs(), vectors);
    verify_merged_rows(index);
    index.prepare_build(config(128), pruning);
    EXPECT_FALSE(index.has_layers());
    EXPECT_FALSE(index.has_hierarchical_graph());
    Factory::build(index, distance);
    EXPECT_EQ(&index.get_base_vecs(), vectors);
    verify_merged_rows(index);
}
} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
