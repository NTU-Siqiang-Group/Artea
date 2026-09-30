// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <thread>
#include <optional>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include <artea/cpu/framework/type_traits/base_traits.hpp>
#include <artea/cpu/framework/type_traits/computer_traits.hpp>
#include <artea/cpu/framework/type_traits/buffer_traits.hpp>
#include <artea/cpu/framework/type_traits/index_traits.hpp>
#include <artea/cpu/framework/type_traits/router_traits.hpp>
#include <artea/cpu/framework/type_traits/refiner_traits.hpp>
#include <artea/cpu/framework/type_traits/graph_factory_traits.hpp>
#include <artea/cpu/containers/vector_dataset.hpp>
#include <artea/cpu/containers/locked_buffer.hpp>
#include <artea/cpu/index/layer_config.hpp>
#include <artea/cpu/index/dynamic_structure/hierarchical_graph.hpp>
#include <artea/cpu/index/compact_structure/hierarchical_graph.hpp>
#include <artea/cpu/index/exact_artea/index_factory.hpp>
#include <artea/cpu/index/exact_artea/rnets_factory.hpp>
#include <artea/cpu/utils/simd_distance.hpp>
#include <artea/cpu/refiner/updaters/neighbor_updater.hpp>
#include <artea/cpu/refiner/nbr_log_table.hpp>
#include <artea/cpu/index/dynamic_structure/refining_graph.hpp>

namespace {
using namespace artea::cpu;
using Base = BaseTraits<uint32_t, float>;
using IT = IndexTraits<Base>;
using SIMD = ComputerTraits<Base, DistanceMetricsT::EUCLIDEAN, 16>;
struct Computer : SIMD {
    struct dist_func_t {
        std::atomic<std::size_t>* calls = nullptr;
        std::optional<float> forced_distance;
        std::optional<std::thread::id> caller_thread;
        auto operator()(const float* a, const float* b) const -> float {
            if (caller_thread && std::this_thread::get_id() != *caller_thread) {
                throw std::logic_error("ARC distance evaluation left the calling thread");
            }
            if (calls) calls->fetch_add(1, std::memory_order_relaxed);
            if (forced_distance) return *forced_distance;
            return SIMD::dist_func_t{}(a, b);
        }
    };
};
using Buffer = BufferTraits<Base, BufferPolicyT::LOCKED_BUFFER_WITH_MUTEX, 32>;
using Router = RouterTraits<Computer, IT>;
using Refiner = RefinerTraits<Computer, Buffer, IT, Router>;
using Traits = GraphFactoryTraits<Refiner, Router>;
using Factory = Traits::exact_artea::factory_t;
using RNets = Traits::exact_artea::rnets_factory_t;
using Index = Traits::exact_artea::index_t;
using Config = Traits::exact_artea::rgraph_config_t;
using Pruning = Traits::exact_artea::pruning_config_t;
using Vectors = Base::vector_array_t;
using Neighbor = Base::nbr_t;
const Computer::dist_func_t distance;

// alpha=2, theta=1 => gamma=4. With tau_k=tau=0 and beta=2,
// Delta_0=9 and Delta_h=18 for h>=1. Capacities deliberately stay at 1.
auto config(float rho = 2) -> Config { return Config(2, 0, 0, rho, 4, 4, 4, 1, 1); }
const Pruning pruning(2, 0, 2);

auto line(std::initializer_list<float> coordinates) -> Vectors {
    Vectors data(static_cast<uint32_t>(coordinates.size()), 16);
    uint32_t id = 0;
    for (float x : coordinates) {
        std::fill_n(data.get(id), 16, 0.0f);
        data.get(id++)[0] = x;
    }
    return data;
}

auto all_vids(const Index& index) -> std::vector<uint32_t> {
    std::vector<uint32_t> result(index.get_base_vecs().get_num_vecs());
    std::iota(result.begin(), result.end(), 0u);
    return result;
}

auto ids(const std::vector<Neighbor>& candidates) -> std::vector<uint32_t> {
    std::vector<uint32_t> result;
    for (const auto& candidate : candidates) result.push_back(candidate.get_vid());
    return result;
}

auto check_equal(const std::vector<Neighbor>& actual, const std::vector<Neighbor>& expected) -> void {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i].get_vid(), expected[i].get_vid());
        EXPECT_FLOAT_EQ(actual[i].get_distance(), expected[i].get_distance());
    }
}

TEST(ExactArteaArcCandidates, IncludesBoundaryExcludesSelfAndKeepsDuplicateCoordinates) {
    auto data = line({0, 36, std::nextafter(36.0f, 100.0f), -36, 1, 0, 80});
    Index index(data, config(), pruning);
    const auto result = Factory::get_arc_candidates(index, 0, 1, all_vids(index), distance);
    EXPECT_EQ(ids(result), (std::vector<uint32_t>{5, 4, 1, 3}));
    check_equal(result, {{5, 0}, {4, 1}, {1, 36}, {3, 36}});
}

TEST(ExactArteaArcCandidates, ScansSuppliedLayerBeforeAndAfterLayerAssignmentWithoutCapacityLimit) {
    auto data = line({0, 1, 2, 3, 10, 20, 40});
    Index index(data, config(), pruning);
    ASSERT_EQ(index.get_num_vertices(), 0u);
    const std::vector<uint32_t> layer_vids{6, 4, 0, 2};
    const auto before = Factory::get_arc_candidates(index, 0, 1, layer_vids, distance);
    EXPECT_EQ(ids(before), (std::vector<uint32_t>{2, 4}));
    RNets::fps_generator(index, distance);
    ASSERT_EQ(index.get_highest_level_id(1), 0u);
    const auto after = Factory::get_arc_candidates(index, 0, 1, layer_vids, distance);
    check_equal(after, before);
    EXPECT_GT(after.size(), index.rgraph_config().ul_max_nbr_size());
    for (uint32_t id = 0; id < data.get_num_vecs(); ++id) {
        for (uint32_t h = 0; h <= index.get_highest_level_id(id); ++h) {
            EXPECT_TRUE(index.fetch_level_nbrs(id, h).empty());
        }
    }
}

TEST(ExactArteaArcCandidates, LayerAndPruningParametersControlAbsoluteCutoff) {
    auto data = line({0, 18, 19, 36, 37, 72, 73});
    Index index(data, config(), pruning);
    EXPECT_EQ(ids(Factory::get_arc_candidates(index, 0, 0, all_vids(index), distance)), (std::vector<uint32_t>{1}));
    EXPECT_EQ(ids(Factory::get_arc_candidates(index, 0, 1, all_vids(index), distance)), (std::vector<uint32_t>{1, 2, 3}));
    EXPECT_EQ(ids(Factory::get_arc_candidates(index, 0, 2, all_vids(index), distance)), (std::vector<uint32_t>{1, 2, 3, 4, 5}));
    Index shifted(data, config(), Pruning(2, 1, 2)); // L1 cutoff = 46.
    EXPECT_EQ(ids(Factory::get_arc_candidates(shifted, 0, 1, all_vids(shifted), distance)), (std::vector<uint32_t>{1, 2, 3, 4}));
    Index scaled(data, config(), Pruning(3, 0, 2)); // gamma=3; L1 cutoff = 28.
    EXPECT_EQ(ids(Factory::get_arc_candidates(scaled, 0, 1, all_vids(scaled), distance)), (std::vector<uint32_t>{1, 2}));
}

TEST(ExactArteaArcCandidates, TenThousandPointsMatchIndependentScalarScanWithDifferentTbbLimits) {
    Vectors data(10000, 16);
    std::mt19937 random(721);
    for (uint32_t id = 0; id < data.get_num_vecs(); ++id) {
        for (unsigned d = 0; d < 16; ++d) data.get(id)[d] = static_cast<int>(random() % 201) - 100;
    }
    Index index(data, config(20), Pruning(2, 0, 20));
    for (uint32_t center : {0u, 512u, 9999u}) {
        for (uint32_t h : {0u, 1u, 2u}) {
            SCOPED_TRACE(center);
            SCOPED_TRACE(h);
            // Independent closed-form cutoffs for this config: 180, 360, 720.
            const double cutoff = h == 0 ? 180 : (h == 1 ? 360 : 720);
            std::vector<uint32_t> layer_vids;
            for (uint32_t id = 0; id < data.get_num_vecs(); id += h + 1) layer_vids.push_back(id);
            std::reverse(layer_vids.begin(), layer_vids.end());
            std::vector<Neighbor> expected;
            for (uint32_t id : layer_vids) {
                if (id == center) continue;
                double squared = 0;
                for (unsigned d = 0; d < 16; ++d) {
                    const double diff = double(data.get(id)[d]) - data.get(center)[d];
                    squared += diff * diff;
                }
                const float dist = static_cast<float>(std::sqrt(squared));
                if (dist <= cutoff) expected.emplace_back(id, dist);
            }
            std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b) {
                return a.get_distance() < b.get_distance() ||
                       (a.get_distance() == b.get_distance() && a.get_vid() < b.get_vid());
            });
            for (std::size_t threads : {1u, 8u}) {
                tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
                std::atomic<std::size_t> calls{0};
                const Computer::dist_func_t counted{&calls, std::nullopt, std::this_thread::get_id()};
                const auto candidates = Factory::get_arc_candidates(index, center, h, layer_vids, counted);
                check_equal(candidates, expected);
                for (const auto& candidate : candidates) EXPECT_TRUE(candidate.is_new());
                EXPECT_EQ(calls.load(), layer_vids.size() - (center % (h + 1) == 0));
            }
        }
    }
}

TEST(ExactArteaArcCandidates, EmptySingletonAndNoCandidates) {
    Vectors empty(0, 16);
    Index empty_index(empty, config(), pruning);
    EXPECT_THROW(Factory::get_arc_candidates(empty_index, 0, 0, all_vids(empty_index), distance), std::out_of_range);
    auto data = line({0});
    Index single(data, config(), pruning);
    std::atomic<std::size_t> calls{0};
    const Computer::dist_func_t counted{&calls, std::nullopt};
    EXPECT_TRUE(Factory::get_arc_candidates(single, 0, 0, all_vids(single), counted).empty());
    EXPECT_EQ(calls.load(), 0u);
    auto isolated_data = line({0, 100, 200});
    Index isolated(isolated_data, config(), pruning);
    EXPECT_TRUE(Factory::get_arc_candidates(isolated, 1, 1, all_vids(isolated), distance).empty());
}

TEST(ExactArteaArcCandidates, RejectsInvalidVertexLayerAndPruningBeforeScanning) {
    auto data = line({0, 1});
    Index index(data, config(), pruning);
    std::atomic<std::size_t> calls{0};
    const Computer::dist_func_t counted{&calls, std::nullopt};
    EXPECT_THROW(Factory::get_arc_candidates(index, 2, 0, all_vids(index), counted), std::out_of_range);
    EXPECT_THROW(Factory::get_arc_candidates(index, Base::invalid_vertex_id, 0, all_vids(index), counted), std::out_of_range);
    EXPECT_THROW(Factory::get_arc_candidates(index, 0, std::numeric_limits<uint32_t>::max(), all_vids(index), counted), std::out_of_range);
    Index invalid_pruning(data, config(), Pruning(1, 0, 2));
    EXPECT_THROW(Factory::get_arc_candidates(invalid_pruning, 0, 1, all_vids(invalid_pruning), counted), std::invalid_argument);
    EXPECT_EQ(calls.load(), 0u);
}

TEST(ExactArteaArcCandidates, RejectsUnrepresentableCutoffBeforeScanning) {
    auto data = line({0, 1});
    Index index(data, config(std::numeric_limits<float>::max() / 4), pruning);
    std::atomic<std::size_t> calls{0};
    const Computer::dist_func_t counted{&calls, std::nullopt};
    EXPECT_THROW(Factory::get_arc_candidates(index, 0, 1, all_vids(index), counted), std::overflow_error);
    EXPECT_EQ(calls.load(), 0u);
}

TEST(ExactArteaArcCandidates, RejectsInvalidDistancesWithoutChangingGraph) {
    auto data = line({0, 1});
    Index index(data, config(), pruning);
    EXPECT_FALSE(index.has_layers());
    EXPECT_FALSE(index.has_hierarchical_graph());
    for (float invalid : {-1.0f, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        const Computer::dist_func_t bad_distance{nullptr, invalid};
        EXPECT_THROW(Factory::get_arc_candidates(index, 0, 1, all_vids(index), bad_distance), std::domain_error);
        EXPECT_FALSE(index.has_layers());
        EXPECT_FALSE(index.has_hierarchical_graph());
        EXPECT_EQ(index.get_num_vertices(), 0u);
    }
}

TEST(ExactArteaArcCandidates, EmptyLayerAndInvalidGlobalLayerIds) {
    auto data = line({0, 1, 2});
    Index index(data, config(), pruning);
    std::atomic<std::size_t> calls{0};
    const Computer::dist_func_t counted{&calls, std::nullopt};
    EXPECT_TRUE(Factory::get_arc_candidates(index, 0, 1, {}, counted).empty());
    EXPECT_EQ(calls.load(), 0u);
    const std::vector<uint32_t> invalid{3, 0};
    EXPECT_THROW(Factory::get_arc_candidates(index, 0, 1, invalid, counted), std::out_of_range);
    EXPECT_EQ(calls.load(), 0u);
}

TEST(ExactArteaPruning, UsesDifferentUpperAndBottomRulesWithoutDegreeLimit) {
    auto data = line({0, 3.25f, 4});
    Index index(data, config(1), Pruning(2, 1, 1));
    const std::vector<uint32_t> layer{2, 0, 1};
    auto candidates = Factory::get_arc_candidates(index, 0, 0, layer, distance);
    EXPECT_EQ(ids(candidates), (std::vector<uint32_t>{1, 2}));
    for (const auto& candidate : candidates) EXPECT_TRUE(candidate.is_new());
    const auto updaters = Factory::make_pruning_updaters(index, distance);
    auto upper = candidates;
    auto bottom = candidates;
    updaters.upper(upper);
    updaters.bottom(bottom);
    EXPECT_EQ(ids(upper), (std::vector<uint32_t>{1}));
    EXPECT_EQ(ids(bottom), (std::vector<uint32_t>{1, 2}));
    EXPECT_GT(bottom.size(), index.rgraph_config().bl_max_nbr_size());
    for (const auto& neighbor : bottom) EXPECT_TRUE(neighbor.is_old());
    check_equal(Factory::prune_arc_candidates(index, 0, 0, layer, distance), bottom);
    check_equal(Factory::prune_arc_candidates(index, 0, 1, layer, distance), upper);
}

TEST(ExactArteaPruning, FiltersLayerBeforePruningAndChecksOldCandidatesToo) {
    auto data = line({0, 1, 1.25f});
    Index index(data, config(), pruning);
    const std::vector<uint32_t> upper_layer{2, 0};
    // Vertex 1 is not in this layer and must not prune vertex 2.
    EXPECT_EQ(ids(Factory::prune_arc_candidates(index, 0, 1, upper_layer, distance)),
              (std::vector<uint32_t>{2}));
    const auto updaters = Factory::make_pruning_updaters(index, distance);
    auto old = Factory::get_arc_candidates(index, 0, 1, all_vids(index), distance);
    for (auto& candidate : old) candidate.mark_as_old();
    updaters.upper(old);
    EXPECT_EQ(ids(old), (std::vector<uint32_t>{1}));
}

TEST(ExactArteaPruning, EmptySingletonStrictBoundaryAndNonpositiveBottomRadius) {
    auto data = line({0, 1, 2});
    Index index(data, config(1), Pruning(2, 1, 1));
    const auto updaters = Factory::make_pruning_updaters(index, distance);
    std::vector<Neighbor> empty;
    updaters.upper(empty);
    updaters.bottom(empty);
    EXPECT_TRUE(empty.empty());
    std::vector<Neighbor> single{{2, 2, true}};
    updaters.bottom(single);
    ASSERT_EQ(single.size(), 1u);
    EXPECT_TRUE(single.front().is_old());
    // Upper conflict distance = threshold = 1; the open ball excludes the boundary.
    auto upper = Factory::prune_arc_candidates(index, 0, 1, all_vids(index), distance);
    EXPECT_EQ(ids(upper), (std::vector<uint32_t>{1, 2}));
    // Both bottom thresholds are negative, so neither candidate conflicts.
    auto bottom = Factory::prune_arc_candidates(index, 0, 0, all_vids(index), distance);
    EXPECT_EQ(ids(bottom), (std::vector<uint32_t>{1, 2}));
    EXPECT_TRUE(Factory::prune_arc_candidates(index, 0, 1, {}, distance).empty());
}

auto scalar_distance(const Vectors& data, uint32_t a, uint32_t b) -> float {
    double squared = 0;
    for (unsigned d = 0; d < 16; ++d) {
        const double diff = double(data.get(a)[d]) - data.get(b)[d];
        squared += diff * diff;
    }
    return static_cast<float>(std::sqrt(squared));
}

TEST(ExactArteaPruning, MatchesIndependentPaperAlgorithmForBothLayers) {
    Vectors data(128, 16);
    std::mt19937 random(927);
    for (uint32_t id = 0; id < 128; ++id) {
        for (unsigned d = 0; d < 16; ++d) data.get(id)[d] = int(random() % 11) - 5;
    }
    for (float tau : {0.0f, 1.0f}) {
        Index index(data, config(), Pruning(2, tau, 2));
        for (uint32_t center : {0u, 17u, 64u, 127u}) {
            for (uint32_t h : {0u, 1u, 2u}) {
                SCOPED_TRACE(tau);
                SCOPED_TRACE(center);
                SCOPED_TRACE(h);
                std::vector<uint32_t> layer;
                for (uint32_t id = 0; id < 128; id += h + 1) layer.push_back(id);
                std::reverse(layer.begin(), layer.end());
                // Independent closed form for alpha=2, theta=1, beta=2, rho=2.
                const float cutoff = (h == 0 ? 18 : (h == 1 ? 36 : 72)) + 10 * tau;
                std::vector<Neighbor> scanned;
                for (uint32_t id : layer) {
                    const float d = scalar_distance(data, center, id);
                    if (id != center && d <= cutoff) scanned.emplace_back(id, d);
                }
                std::sort(scanned.begin(), scanned.end(), [](const auto& a, const auto& b) {
                    return a.get_distance() < b.get_distance() ||
                           (a.get_distance() == b.get_distance() && a.get_vid() < b.get_vid());
                });
                std::vector<Neighbor> expected;
                for (const auto& candidate : scanned) {
                    const float conflict_radius =
                        (candidate.get_distance() - (h == 0 ? 3 * tau * 2 : 0)) / 2;
                    const bool conflict = std::any_of(expected.begin(), expected.end(), [&](const auto& retained) {
                        return scalar_distance(data, candidate.get_vid(), retained.get_vid()) < conflict_radius;
                    });
                    if (!conflict) expected.push_back(candidate);
                }
                const auto actual = Factory::prune_arc_candidates(index, center, h, layer, distance);
                check_equal(actual, expected);
                EXPECT_GT(actual.size(), 1u); // Configured degree is 1, deliberately ignored.
            }
        }
    }
}

TEST(ExactArteaPruning, RejectsInvalidConfigShiftOverflowAndPairDistances) {
    auto data = line({0, 1, 1.25f});
    Index bad_alpha(data, config(), Pruning(1, 0, 2));
    EXPECT_THROW(Factory::make_pruning_updaters(bad_alpha, distance), std::invalid_argument);
    Index overflow(data, config(), Pruning(2, std::numeric_limits<float>::max(), 2));
    EXPECT_THROW(Factory::make_pruning_updaters(overflow, distance), std::overflow_error);
    Index index(data, config(), pruning);
    const auto original = Factory::get_arc_candidates(index, 0, 0, all_vids(index), distance);
    for (float invalid : {-1.0f, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        const Computer::dist_func_t bad_distance{nullptr, invalid};
        const auto updaters = Factory::make_pruning_updaters(index, bad_distance);
        auto candidates = original;
        EXPECT_THROW(updaters.upper(candidates), std::domain_error);
        check_equal(candidates, original);
        for (const auto& candidate : candidates) EXPECT_TRUE(candidate.is_new());
    }
}

TEST(ExactArteaPruning, SharedKernelPreservesLegacyIncrementalAndConditionDispatch) {
    auto data = line({0, 1, 1.25f});
    IT::dynamic::refining_graph_t graph(data, IT::layer_config_t(1));
    Buffer::log_table_t logs;
    Refiner::pruning_updater_t legacy(distance, data, logs, graph, 2, 1.5f);
    std::vector<Neighbor> old{{1, 1}, {2, 1.25f}};
    legacy(0, old);
    EXPECT_EQ(ids(old), (std::vector<uint32_t>{1, 2}));
    std::vector<Neighbor> fresh{{1, 1, true}, {2, 1.25f, true}};
    legacy(0, fresh); // Ordinary refiner calls still use scaled_ineq.
    EXPECT_EQ(ids(fresh), (std::vector<uint32_t>{1}));
    fresh = {{1, 1, true}, {2, 1.25f, true}};
    legacy.template update_impl<PruningConditionT::scaled_shifted_ineq>(0, fresh);
    EXPECT_EQ(ids(fresh), (std::vector<uint32_t>{1, 2}));
    std::vector<Neighbor> empty;
    legacy(0, empty);
    EXPECT_TRUE(empty.empty());
}

TEST(ExactArteaPruning, ExistingCompactBuilderStoresVariableRowsBeyondCapacityMetadata) {
    auto data = line({0, 3.25f, 4});
    Index index(data, config(1), Pruning(2, 1, 1));
    std::vector<std::vector<Neighbor>> bottom(3);
    for (uint32_t id = 0; id < 3; ++id) {
        bottom[id] = Factory::prune_arc_candidates(index, id, 0, all_vids(index), distance);
    }
    // Vertex 0 is the singleton top; all three bottom rows retain two neighbors.
    IT::compact::hierarchical_graph_t graph(1, 1, 1, 3, {2, 1});
    graph.get_vids_by_highest_level_mut() = {{1, 2}, {0}};
    graph.get_vertex_info_table_mut() = {{1, 0}, {0, 0}, {0, 1}};
    auto base_counts = graph.get_nbr_offsets_mut(0);
    base_counts[0] = bottom[1].size();
    base_counts[1] = bottom[2].size();
    auto top_counts = graph.get_nbr_offsets_mut(1);
    top_counts[0] = 0;
    top_counts[1] = bottom[0].size();
    graph.allocate_neighbors_from_row_counts();
    graph.set_entry_point_vid(0);
    for (uint32_t id = 0; id < 3; ++id) {
        auto row = graph.fetch_level_nbrs_mut(id, 0);
        ASSERT_EQ(row.size(), bottom[id].size());
        for (std::size_t i = 0; i < row.size(); ++i) row[i] = bottom[id][i].get_vid();
        const auto stored = graph.fetch_level_nbrs(id, 0);
        EXPECT_EQ((std::vector<uint32_t>(stored.begin(), stored.end())), ids(bottom[id]));
        EXPECT_GT(graph.num_valid_nbrs(id, 0), graph.max_nbr_size(0));
    }
    EXPECT_TRUE(graph.fetch_level_nbrs(0, 1).empty());
    EXPECT_EQ(graph.neighbor_ids().size(), 6u);
}
} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
