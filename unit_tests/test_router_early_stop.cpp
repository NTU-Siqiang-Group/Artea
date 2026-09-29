// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include <artea/cpu/framework/type_traits/base_traits.hpp>
#include <artea/cpu/framework/type_traits/computer_traits.hpp>
#include <artea/cpu/framework/type_traits/index_traits.hpp>
#include <artea/cpu/framework/type_traits/router_traits.hpp>
#include <artea/cpu/containers/vector_dataset.hpp>
#include <artea/cpu/index/layer_config.hpp>
#include <artea/cpu/index/dynamic_structure/hierarchical_graph.hpp>
#include <artea/cpu/index/compact_structure/hierarchical_graph.hpp>
#include <artea/cpu/index/exact_artea/index_structure.hpp>
#include <artea/cpu/router/vector_router.hpp>
#include <artea/cpu/router/data_structures/candidate_entry.hpp>
#include <artea/cpu/router/data_structures/std_candidate_queue.hpp>
#include <artea/cpu/router/data_structures/visited_table_pool.hpp>
#include <artea/cpu/router/detail/candidate_sample_utils.hpp>
#include <artea/cpu/router/hierarchical_graph_router.hpp>
#include <artea/cpu/utils/simd_distance.hpp>

namespace {
using namespace artea::cpu;

// These paths seed from the graph (including its apex sampler), never the
// MKL random-initialization helper. Keep that unrelated dependency out of this test.
struct UnusedRandomSeq {};
struct Base : BaseTraits<uint32_t, float> { using random_seq_t = UnusedRandomSeq; };
using IT = IndexTraits<Base>;
using Graph = IT::compact::hierarchical_graph_t;
using Config = IT::exact_artea::rgraph_config_t;
using Pruning = IT::exact_artea::pruning_config_t;
using Vectors = Base::vector_array_t;

template <DistanceMetricsT Metric>
struct CountingComputer : ComputerTraits<Base, Metric, 16> {
    struct dist_func_t {
        std::atomic<unsigned>* calls;
        auto operator()(const float* a, const float* b) const -> float {
            calls->fetch_add(1, std::memory_order_relaxed);
            return typename ComputerTraits<Base, Metric, 16>::dist_func_t{}(a, b);
        }
    };
};
template <DistanceMetricsT Metric = DistanceMetricsT::EUCLIDEAN>
using RT = RouterTraits<CountingComputer<Metric>, IT>;

auto config(float rho = 1) -> Config { return Config(2, 0, 0, rho, 4, 4, 4, 4, 4); }
const Pruning pruning(2, 0); // theta=1 gives gamma=4, T_1=8, T_2=16.

auto vectors(std::initializer_list<float> coordinates) -> Vectors {
    Vectors result(static_cast<uint32_t>(coordinates.size()), 16);
    uint32_t id = 0;
    for (float x : coordinates) {
        std::fill_n(result.get(id), 16, 0.0f);
        result.get(id++)[0] = x;
    }
    return result;
}

// L2: singleton entry 0; L1: chain 0 -> 1 -> 2 -> 3; L0: complete.
// This exposes an observable upper-layer stopping point while allowing L0
// to recover the nearest neighbor from either stopping point.
auto graph(uint32_t top = 2) -> Graph {
    std::vector<std::size_t> counts(top + 1, 0);
    counts[top] = 1;
    counts[top == 0 ? 0 : 1] += 3;
    Graph result(top, 4, 4, 4, counts);
    for (uint32_t id = 0; id < 4; ++id) {
        const uint32_t highest = id == 0 ? top : std::min(top, 1u);
        auto& bucket = result.get_vids_by_highest_level_mut()[highest];
        const auto local = static_cast<uint32_t>(bucket.size());
        bucket.push_back(id);
        result.get_vertex_info_table_mut()[id] = {highest, local};
        for (uint32_t h = 0; h <= highest; ++h) {
            result.get_nbr_offsets_mut(highest)[local * (highest + 1) + highest - h] =
                h == 0 ? 3 : (h == 1 && id < 3 ? 1 : 0);
        }
    }
    result.allocate_neighbors_from_row_counts();
    for (uint32_t id = 0; id < 4; ++id) {
        auto row = result.fetch_level_nbrs_mut(id, 0);
        unsigned pos = 0;
        for (uint32_t other = 0; other < 4; ++other) if (id != other) row[pos++] = other;
        if (top > 0 && id < 3) result.fetch_level_nbrs_mut(id, 1)[0] = id + 1;
    }
    result.set_entry_point_vid(0);
    return result;
}

TEST(RouterEarlyStop, GreedyDefaultDisabledAndInclusiveThreshold) {
    auto data = vectors({12, 8, 4, 0});
    auto g = graph();
    std::atomic<unsigned> calls{0};
    RT<>::dist_func_t distance{&calls};
    RT<>::single_layer_router_t router(data, distance);
    RT<>::visited_table_t visited(4);
    const auto range = detail::make_layer_range(g, 1u);
    EXPECT_EQ(router.greedy_search(data.get(3), range, 0, 12, visited, 100).first, 3u);
    EXPECT_EQ(calls.load(), 3u);
    visited.clear(); calls = 0;
    EXPECT_EQ(router.greedy_search<true>(data.get(3), range, 0, 12, visited, 8).first, 1u);
    EXPECT_EQ(calls.load(), 1u);
    visited.clear(); calls = 0;
    EXPECT_EQ(router.greedy_search<true>(data.get(3), range, 0, 12, visited, 12).first, 0u);
    EXPECT_EQ(calls.load(), 0u);
    visited.clear(); calls = 0;
    EXPECT_EQ(router.greedy_search<true>(data.get(3), range, 0, 12, visited,
              std::nextafter(8.0f, 0.0f)).first, 2u);
    EXPECT_EQ(calls.load(), 2u);
}

template <DistanceMetricsT Metric>
auto verify_hierarchy() -> void {
    auto data = vectors({12, 8, 4, 0});
    auto g = graph();
    std::atomic<unsigned> calls{0};
    typename RT<Metric>::dist_func_t distance{&calls};
    typename RT<Metric>::hierarchical_graph_router_t router(data, distance, 1, 4);
    EXPECT_EQ(router.query(data.get(3), g).front().get_vid(), 3u);
    EXPECT_EQ(calls.load(), 7u);
    calls = 0;
    const auto early = router.template query<false, true>(data.get(3), g, config(), pruning);
    EXPECT_EQ(early.front().get_vid(), 3u);
    EXPECT_FLOAT_EQ(early.front().get_distance(), 0);
    EXPECT_EQ(calls.load(), 5u);
    const auto random = router.template query<true, true>(data.get(3), g, config(), pruning);
    EXPECT_EQ(random.front().get_vid(), 3u);
}

TEST(RouterEarlyStop, HierarchyUsesL2ThresholdOnlyAboveL0) {
    verify_hierarchy<DistanceMetricsT::EUCLIDEAN>();
}
TEST(RouterEarlyStop, HierarchyConvertsThresholdForSquaredL2) {
    verify_hierarchy<DistanceMetricsT::EUCLIDEAN_SQR>();
}

TEST(RouterEarlyStop, DisabledModeDoesNotAccessConfigs) {
    auto data = vectors({12, 8, 4, 0});
    auto g = graph();
    std::atomic<unsigned> calls{0};
    RT<>::dist_func_t distance{&calls};
    RT<>::hierarchical_graph_router_t router(data, distance, 1, 4);
    struct NoConfigMethods {};
    EXPECT_EQ(router.query(data.get(3), g, NoConfigMethods{}, NoConfigMethods{}).front().get_vid(), 3u);
    EXPECT_EQ(calls.load(), 7u);
    const auto batch = router.batch_query(data, g, NoConfigMethods{}, NoConfigMethods{});
    for (uint32_t id = 0; id < 4; ++id) EXPECT_EQ(batch[id].get_vid(), id);
}

TEST(RouterEarlyStop, BatchComputesThresholdsOnceAndMatchesIndividualQueries) {
    auto data = vectors({12, 8, 4, 0});
    auto g = graph();
    std::atomic<unsigned> calls{0};
    RT<>::dist_func_t distance{&calls};
    RT<>::hierarchical_graph_router_t router(data, distance, 1, 4);
    struct CountedConfig {
        mutable unsigned evaluations = 0;
        auto early_stop_threshold(uint32_t h, const Pruning& p) const -> float {
            ++evaluations;
            return config().early_stop_threshold(h, p);
        }
    } cfg;
    for (auto threads : {1u, 4u}) {
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
        cfg.evaluations = 0;
        const auto batch = router.batch_query<false, true>(data, g, cfg, pruning);
        EXPECT_EQ(cfg.evaluations, 2u);
        for (uint32_t id = 0; id < 4; ++id) {
            const auto single = router.query<false, true>(data.get(id), g, config(), pruning);
            EXPECT_EQ(batch[id].get_vid(), single.front().get_vid());
            EXPECT_EQ(batch[id].get_vid(), id);
        }
    }
}

TEST(RouterEarlyStop, EmptyAndBottomOnlyGraphsDoNotNeedValidUpperLayerThresholds) {
    auto data = vectors({12, 8, 4, 0});
    std::atomic<unsigned> calls{0};
    RT<>::dist_func_t distance{&calls};
    RT<>::hierarchical_graph_router_t router(data, distance, 1, 4);
    const Pruning invalid(1, 0);
    Graph empty(0, 4, 4, 0, {0});
    const auto none = router.query<false, true>(data.get(0), empty, config(), invalid);
    EXPECT_TRUE(none.empty());
    const auto batch = router.batch_query<false, true>(data, empty, config(), invalid);
    for (const auto& entry : batch) EXPECT_TRUE(entry.is_invalid());
    EXPECT_EQ(calls.load(), 0u);
    auto bottom = graph(0);
    const auto result = router.query<false, true>(data.get(3), bottom, config(), invalid);
    EXPECT_EQ(result.front().get_vid(), 3u);
}

TEST(RouterEarlyStop, ReadsConfigsFromIndexAndSupportsDynamicGraph) {
    auto data = vectors({12, 8, 4, 0});
    exact_artea::IndexStructure<IT> index(data, config(), pruning);
    index.add_vertices(4);
    for (uint32_t id = 0; id < 4; ++id) {
        index.assign_layer(id, id == 0 ? 1 : 0);
        auto row = index.fetch_level_nbrs(id, 0);
        unsigned pos = 0;
        for (uint32_t other = 0; other < 4; ++other) {
            if (id != other) row[pos++] = Base::nbr_t(other, 0);
        }
    }
    std::atomic<unsigned> calls{0};
    RT<>::dist_func_t distance{&calls};
    RT<>::hierarchical_graph_router_t router(data, distance, 1, 4);
    const auto single = router.query<false, true>(data.get(3), index);
    EXPECT_EQ(single.front().get_vid(), 3u);
    const auto batch = router.batch_query<false, true>(data, index);
    for (uint32_t id = 0; id < 4; ++id) EXPECT_EQ(batch[id].get_vid(), id);
    const auto bare = router.query<false, true>(
        data.get(3), index.get_hierarchical_graph(), config(), pruning);
    EXPECT_EQ(bare.front().get_vid(), 3u);
}

TEST(RouterEarlyStop, RejectsUnrepresentableSquaredThresholds) {
    auto data = vectors({12, 8, 4, 0});
    auto g = graph();
    std::atomic<unsigned> calls{0};
    RT<DistanceMetricsT::EUCLIDEAN_SQR>::dist_func_t distance{&calls};
    RT<DistanceMetricsT::EUCLIDEAN_SQR>::hierarchical_graph_router_t router(data, distance, 1, 4);
    EXPECT_THROW((router.query<false, true>(data.get(3), g, config(1e20f), pruning)),
                 std::overflow_error);
    EXPECT_THROW((router.query<false, true>(data.get(3), g, config(1e-30f), pruning)),
                 std::overflow_error);
    EXPECT_EQ(calls.load(), 0u);
}
} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
