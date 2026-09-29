// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <tbb/global_control.h>

// Exercise the production traits, SIMD distance, dataset and graph storage
// without pulling in unrelated refiners and their MKL dependency.
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
#include <artea/cpu/index/exact_artea/rnets_factory.hpp>
#include <artea/cpu/utils/simd_distance.hpp>

namespace {
using namespace artea::cpu;
using Base = BaseTraits<uint32_t, float>;
using IndexTraitsT = IndexTraits<Base>;
using Computer = ComputerTraits<Base, DistanceMetricsT::EUCLIDEAN, 16>;
using Buffer = BufferTraits<Base, BufferPolicyT::LOCKED_BUFFER_WITH_MUTEX, 32>;
using Router = RouterTraits<Computer, IndexTraitsT>;
using Refiner = RefinerTraits<Computer, Buffer, IndexTraitsT, Router>;
using FactoryTraits = GraphFactoryTraits<Refiner, Router>;
using RNetsFactoryT = FactoryTraits::exact_artea::rnets_factory_t;
using Index = FactoryTraits::exact_artea::index_t;
using Config = FactoryTraits::exact_artea::rgraph_config_t;
using Pruning = FactoryTraits::exact_artea::pruning_config_t;
using Vectors = Base::vector_array_t;
using Layers = std::vector<std::vector<uint32_t>>;
const Computer::dist_func_t distance;

auto config(float rho = 1, float beta = 2, float tau_k = 0) -> Config {
    return Config(beta, tau_k, 0, rho, 10, 10, 10, 3, 5);
}

auto line(std::initializer_list<float> coordinates) -> Vectors {
    Vectors vectors(static_cast<uint32_t>(coordinates.size()), 16);
    uint32_t id = 0;
    for (float coordinate : coordinates) {
        std::fill_n(vectors.get(id), 16, 0.0f);
        vectors.get(id++)[0] = coordinate;
    }
    return vectors;
}

auto layer(const Index& index, uint32_t h) -> std::vector<uint32_t> {
    std::vector<uint32_t> ids;
    for (uint32_t id = 0; id < index.get_num_vertices(); ++id) {
        if (index.get_highest_level_id(id) >= h) ids.push_back(id);
    }
    return ids;
}

// Independent reference: recompute every minimum from all selected centers,
// without the incremental cache, covered-point skip, or parallel reduction.
auto reference(const Vectors& vectors, const Config& cfg) -> Layers {
    Layers result(1);
    result.front().resize(vectors.get_num_vecs());
    std::iota(result.front().begin(), result.front().end(), uint32_t(0));
    while (result.back().size() > 1) {
        const auto& previous = result.back();
        const auto radius = cfg.radius_at(static_cast<uint32_t>(result.size()));
        std::vector<uint32_t> centers{previous.front()};
        for (;;) {
            uint32_t farthest = Base::invalid_vertex_id;
            float farthest_distance = radius;
            for (auto point : previous) {
                float nearest = std::numeric_limits<float>::max();
                for (auto center : centers) {
                    nearest = std::min(nearest, distance(vectors.get(point), vectors.get(center)));
                }
                if (nearest > farthest_distance ||
                    (nearest > radius && nearest == farthest_distance && point < farthest)) {
                    farthest = point;
                    farthest_distance = nearest;
                }
            }
            if (farthest == Base::invalid_vertex_id) break;
            centers.push_back(farthest);
        }
        std::sort(centers.begin(), centers.end());
        result.push_back(std::move(centers));
    }
    return result;
}

auto verify(const Index& index, const Layers& expected) -> void {
    const auto& vectors = index.get_base_vecs();
    ASSERT_EQ(index.get_num_vertices(), vectors.get_num_vecs());
    ASSERT_EQ(index.max_allowed_level_id(), expected.size() - 1);
    ASSERT_EQ(index.get_hierarchical_graph().max_allowed_level_id(), expected.size() - 1);
    if (vectors.get_num_vecs() == 0) {
        EXPECT_EQ(index.top_occupied_level_id(), Index::invalid_level_id);
        return;
    }
    ASSERT_EQ(index.top_occupied_level_id(), expected.size() - 1);
    for (uint32_t h = 0; h < expected.size(); ++h) {
        SCOPED_TRACE(h);
        const auto current = layer(index, h);
        ASSERT_EQ(current, expected[h]);
        for (auto id : current) {
            const auto neighbors = index.fetch_level_nbrs(id, h);
            EXPECT_EQ(neighbors.size(), h == 0 ? 5 : 3);
            for (const auto& neighbor : neighbors) EXPECT_TRUE(neighbor.is_invalid());
        }
        if (h == 0) continue;
        const auto radius = index.radius_at(h);
        for (std::size_t i = 0; i < current.size(); ++i) {
            for (std::size_t j = i + 1; j < current.size(); ++j) {
                EXPECT_GT(distance(vectors.get(current[i]), vectors.get(current[j])), radius);
            }
        }
        for (auto point : expected[h - 1]) {
            float nearest = std::numeric_limits<float>::max();
            for (auto center : current) {
                nearest = std::min(nearest, distance(vectors.get(point), vectors.get(center)));
            }
            EXPECT_LE(nearest, radius);
        }
    }
    EXPECT_EQ(expected.back().size(), 1);
}

TEST(ExactArteaRnets, EmptyAndSingleton) {
    for (auto n : {0u, 1u}) {
        Vectors vectors(n, 16);
        if (n) std::fill_n(vectors.get(0), 16, 0.0f);
        Index index(vectors, config());
        RNetsFactoryT::fps_generator(index, distance);
        verify(index, reference(vectors, config()));
    }
}

TEST(ExactArteaRnets, DuplicatePointsRemainOnBaseLayer) {
    auto vectors = line({7, 7, 7, 7});
    Index index(vectors, config());
    RNetsFactoryT::fps_generator(index, distance);
    verify(index, Layers{{0, 1, 2, 3}, {0}});
}

TEST(ExactArteaRnets, RadiusEqualityIsCovered) {
    auto vectors = line({0, 1, 2});
    Index index(vectors, config());
    RNetsFactoryT::fps_generator(index, distance);
    verify(index, Layers{{0, 1, 2}, {0, 2}, {0}});
}

TEST(ExactArteaRnets, NextFloatAboveRadiusIsNotCovered) {
    auto vectors = line({0, 1, std::nextafter(1.0f, 2.0f)});
    Index index(vectors, config());
    RNetsFactoryT::fps_generator(index, distance);
    verify(index, Layers{{0, 1, 2}, {0, 2}, {0}});
}

TEST(ExactArteaRnets, EqualFarthestDistancesChooseSmallestId) {
    auto vectors = line({0, 3, 4});
    vectors.get(1)[1] = 4;
    vectors.get(2)[1] = 3;
    Index index(vectors, config(2));
    RNetsFactoryT::fps_generator(index, distance);
    verify(index, Layers{{0, 1, 2}, {0, 1}, {0, 1}, {0}});
}

TEST(ExactArteaRnets, UsesTauKInLayerRadius) {
    auto vectors = line({0, 2, 3, 5});
    Index index(vectors, config(1, 2, 2));
    RNetsFactoryT::fps_generator(index, distance);
    verify(index, Layers{{0, 1, 2, 3}, {0, 3}, {0}});
}

TEST(ExactArteaRnets, ActualHeightCanExceedInsertionHeuristic) {
    auto vectors = line({0, 1024});
    Index index(vectors, config());
    EXPECT_EQ(index.max_allowed_level_id(), 1);
    RNetsFactoryT::fps_generator(index, distance);
    EXPECT_EQ(index.top_occupied_level_id(), 11);
    verify(index, reference(vectors, config()));
}

TEST(ExactArteaRnets, MatchesBruteForceWithOneAndEightThreads) {
    for (uint32_t seed : {17u, 93u, 721u}) {
        SCOPED_TRACE(seed);
        Vectors vectors(1025, 16);
        std::mt19937 random(seed);
        for (uint32_t i = 0; i < vectors.get_num_vecs(); ++i) {
            std::fill_n(vectors.get(i), 16, 0.0f);
            vectors.get(i)[0] = static_cast<float>(random() % 21);
            vectors.get(i)[1] = static_cast<float>(random() % 21);
        }
        const auto cfg = config(3, 1.75f, 0.5f);
        const auto expected = reference(vectors, cfg);
        for (std::size_t threads : {1u, 8u}) {
            tbb::global_control limit(tbb::global_control::max_allowed_parallelism, threads);
            Index index(vectors, cfg, Pruning(1.7f, 0.25f, 3));
            RNetsFactoryT::fps_generator(index, distance);
            verify(index, expected);
            EXPECT_EQ(&index.get_base_vecs(), &vectors);
            EXPECT_FLOAT_EQ(index.pruning_config().scale_coeffs(), 1.7f);
            EXPECT_FLOAT_EQ(index.pruning_config().shifted_coeffs(), 0.25f);
            EXPECT_FLOAT_EQ(index.pruning_config().l0_min_distance(), 3);
        }
    }
}

TEST(ExactArteaRnets, RejectsNonemptyGraphAndSupportsExplicitRebuild) {
    auto vectors = line({0, 2, 5});
    Index index(vectors, config());
    RNetsFactoryT::fps_generator(index, distance);
    const auto* graph = &index.get_hierarchical_graph();
    EXPECT_THROW(RNetsFactoryT::fps_generator(index, distance), std::logic_error);
    EXPECT_EQ(&index.get_hierarchical_graph(), graph);
    index.prepare_build(config());
    RNetsFactoryT::fps_generator(index, distance);
    verify(index, reference(vectors, config()));
}

TEST(ExactArteaRnets, RejectsNonfiniteDistancesWithoutPublishing) {
    for (float value : {std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        auto vectors = line({0, value});
        Index index(vectors, config());
        const auto* graph = &index.get_hierarchical_graph();
        EXPECT_THROW(RNetsFactoryT::fps_generator(index, distance), std::domain_error);
        EXPECT_EQ(&index.get_hierarchical_graph(), graph);
        EXPECT_EQ(index.get_num_vertices(), 0);
    }
}

TEST(ExactArteaRnets, RejectsRadiusOverflowWithoutPublishing) {
    auto vectors = line({0, 4});
    Index index(vectors, config(2, std::numeric_limits<float>::max()));
    const auto* graph = &index.get_hierarchical_graph();
    EXPECT_THROW(RNetsFactoryT::fps_generator(index, distance), std::overflow_error);
    EXPECT_EQ(&index.get_hierarchical_graph(), graph);
    EXPECT_EQ(index.max_allowed_level_id(), 1);
    EXPECT_EQ(index.get_num_vertices(), 0);
}

TEST(ExactArteaRnets, OwnedDatasetMustBePrepared) {
    auto dataset = std::make_unique<Base::vector_dataset_t>();
    dataset->get_base_vecs() = line({0, 3, 7});
    Index index(std::move(dataset));
    EXPECT_THROW(RNetsFactoryT::fps_generator(index, distance), std::logic_error);
    index.prepare_build(config());
    RNetsFactoryT::fps_generator(index, distance);
    verify(index, reference(index.get_base_vecs(), config()));
}
} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
