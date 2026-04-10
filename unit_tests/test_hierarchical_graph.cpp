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
 * @FilePath: /Artea/unit_tests/test_hierarchical_graph.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Tests for HierarchicalGraph: layer container, atomic
 *               lnbr_t entry point, and concurrent vertex insertion across
 *               its InternalGraph layers.
 */

#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

class HierarchicalGraphTest : public ::testing::Test {
protected:
    /**
     * @brief Build a HierarchicalGraph with @p num_layers committed layers,
     *        each layer pre-allocating @p max_num_vertices_per_layer vertices
     *        with neighbor capacity @p max_nbr_size.
     *
     * Uses @c grow_layers so both the slot and the visible count are
     * committed in one step. The container is reserved with capacity
     * @c num_layers (the minimum that fits all requested layers).
     */
    static auto make_hg(
        const layer_num_t num_layers,
        const vertex_num_t max_num_vertices_per_layer,
        const vertex_num_t max_nbr_size
    ) -> std::unique_ptr<dynamic::hierarchical_graph_t> {
        auto hg = std::make_unique<dynamic::hierarchical_graph_t>(num_layers);
        hg->grow_layers(num_layers, [&](layer_id_t) {
            return std::make_unique<dynamic::internal_graph_t>(
                max_num_vertices_per_layer, max_nbr_size
            );
        });
        return hg;
    }
};

// ============================================================
//  Construction & layer container basics
// ============================================================

TEST_F(HierarchicalGraphTest, ConstructionEmpty) {
    dynamic::hierarchical_graph_t hg(/*max_layers=*/0);
    EXPECT_EQ(hg.get_num_layers(), 0u);
    EXPECT_EQ(hg.max_layers(), 0u);
}

TEST_F(HierarchicalGraphTest, ConstructionReservesCapacity) {
    dynamic::hierarchical_graph_t hg(/*max_layers=*/5);
    // Visible count starts at 0 because no layers have been committed yet.
    EXPECT_EQ(hg.get_num_layers(), 0u);
    EXPECT_EQ(hg.max_layers(), 5u);
    // All reserved slots are default-constructed unique_ptrs (nullptr).
    for (layer_id_t l = 0; l < 5; ++l) {
        EXPECT_EQ(hg.get_layer_graphs()[l].get(), nullptr);
    }
}

TEST_F(HierarchicalGraphTest, GrowLayersCommitsProgressively) {
    dynamic::hierarchical_graph_t hg(/*max_layers=*/4);
    std::atomic<uint32_t> factory_calls{0};
    auto factory = [&](layer_id_t) {
        factory_calls.fetch_add(1);
        return std::make_unique<dynamic::internal_graph_t>(/*max=*/128, /*nbr=*/8);
    };

    EXPECT_EQ(hg.get_num_layers(), 0u);
    hg.grow_layers(2, factory);
    EXPECT_EQ(hg.get_num_layers(), 2u);
    EXPECT_EQ(factory_calls.load(), 2u);

    // Idempotent: calling with the same target doesn't invoke factory again.
    hg.grow_layers(2, factory);
    EXPECT_EQ(hg.get_num_layers(), 2u);
    EXPECT_EQ(factory_calls.load(), 2u);

    // Incremental growth.
    hg.grow_layers(4, factory);
    EXPECT_EQ(hg.get_num_layers(), 4u);
    EXPECT_EQ(factory_calls.load(), 4u);
}

TEST_F(HierarchicalGraphTest, GrowLayersConcurrentCallersNoDoubleFactory) {
    // Multiple threads all asking for the same growth target. The factory
    // must run exactly N times in total (not N × num_threads).
    dynamic::hierarchical_graph_t hg(/*max_layers=*/8);
    constexpr layer_num_t target = 5;
    std::atomic<uint32_t> factory_calls{0};
    auto factory = [&](layer_id_t) {
        factory_calls.fetch_add(1);
        return std::make_unique<dynamic::internal_graph_t>(/*max=*/64, /*nbr=*/8);
    };

    constexpr uint32_t num_threads = 16;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (uint32_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&] { hg.grow_layers(target, factory); });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(hg.get_num_layers(), target);
    EXPECT_EQ(factory_calls.load(), target);
    for (layer_id_t l = 0; l < target; ++l) {
        EXPECT_NE(hg.get_layer_graphs()[l].get(), nullptr);
    }
}

TEST_F(HierarchicalGraphTest, CommitLayerAfterSetLayerGraph) {
    // Legacy two-step flow: set_layer_graph then commit_layer.
    dynamic::hierarchical_graph_t hg(/*max_layers=*/3);
    EXPECT_EQ(hg.get_num_layers(), 0u);

    hg.set_layer_graph(0, std::make_unique<dynamic::internal_graph_t>(/*max=*/256, /*nbr=*/16));
    // Not yet visible.
    EXPECT_EQ(hg.get_num_layers(), 0u);
    hg.commit_layer(0);
    EXPECT_EQ(hg.get_num_layers(), 1u);

    hg.set_layer_graph(1, std::make_unique<dynamic::internal_graph_t>(/*max=*/256, /*nbr=*/16));
    hg.commit_layer(1);
    EXPECT_EQ(hg.get_num_layers(), 2u);
}

TEST_F(HierarchicalGraphTest, SetGetLayerGraphRoundTrip) {
    auto hg = make_hg(/*num_layers=*/3,
                      /*max_num_vertices_per_layer=*/1024,
                      /*max_nbr_size=*/32);

    EXPECT_EQ(hg->get_num_layers(), 3u);

    for (layer_id_t l = 0; l < 3; ++l) {
        auto& layer = hg->get_layer_graph(l);
        EXPECT_EQ(layer.get_max_num_vertices(), 1024u);
        EXPECT_EQ(layer.max_nbr_size(), 32u);
        EXPECT_EQ(layer.get_num_vertices(), 0u);
    }
}

TEST_F(HierarchicalGraphTest, GetLayerGraphsConstAndMutable) {
    auto hg = make_hg(/*num_layers=*/2, /*max=*/256, /*nbr=*/16);

    // mutable accessor
    auto& mut = hg->get_layer_graphs();
    EXPECT_EQ(mut.size(), 2u);

    // const accessor
    const auto& chg = *hg;
    const auto& cref = chg.get_layer_graphs();
    EXPECT_EQ(cref.size(), 2u);

    // const layer access
    const auto& clayer = chg.get_layer_graph(1);
    EXPECT_EQ(clayer.max_nbr_size(), 16u);
}

// ============================================================
//  Per-layer InternalGraph operations through HierarchicalGraph
// ============================================================

TEST_F(HierarchicalGraphTest, PerLayerSerialVertexInsertion) {
    auto hg = make_hg(/*num_layers=*/3,
                      /*max=*/1024,
                      /*nbr=*/16);

    // Insert deterministic vertices into each layer.
    for (layer_id_t l = 0; l < 3; ++l) {
        auto& layer = hg->get_layer_graph(l);
        for (vertex_num_t i = 0; i < 100; ++i) {
            const vertex_id_t v = layer.add_vertex(/*lower_layer_vid=*/i + 1);
            EXPECT_EQ(v, i);
            EXPECT_EQ(layer.num_valid_nbrs(v), 0u);
        }
        EXPECT_EQ(layer.get_num_vertices(), 100u);
    }

    // Inter-layer links survived.
    for (layer_id_t l = 0; l < 3; ++l) {
        const auto& layer = hg->get_layer_graph(l);
        for (vertex_num_t i = 0; i < 100; ++i) {
            EXPECT_EQ(layer.get_inter_layer_link(i), i + 1);
        }
    }
}

TEST_F(HierarchicalGraphTest, ConcurrentVertexInsertionWithinLayer) {
    constexpr vertex_num_t per_layer = 100'000;
    auto hg = make_hg(/*num_layers=*/2,
                      /*max=*/per_layer,
                      /*nbr=*/16);

    // Concurrently insert into layer 0.
    auto& layer0 = hg->get_layer_graph(0);
    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, per_layer),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                layer0.add_vertex(i);  // lower_layer_vid = i
            }
        }
    );

    EXPECT_EQ(layer0.get_num_vertices(), per_layer);

    // Each lower_layer_vid in [0, per_layer) must appear exactly once.
    std::vector<uint8_t> seen(per_layer, 0);
    for (vertex_num_t v = 0; v < per_layer; ++v) {
        const vertex_id_t lower = layer0.get_inter_layer_link(v);
        ASSERT_LT(lower, per_layer);
        seen[lower]++;
    }
    for (vertex_num_t i = 0; i < per_layer; ++i) {
        EXPECT_EQ(seen[i], 1) << "lower_layer_vid " << i << " seen " << int(seen[i]) << " times";
    }
}

TEST_F(HierarchicalGraphTest, IndependentLayersInParallel) {
    // Insert into multiple layers in parallel; verify each layer is
    // independent (no cross-talk through the hierarchy).
    constexpr layer_num_t num_layers = 4;
    constexpr vertex_num_t per_layer = 20'000;
    auto hg = make_hg(num_layers, per_layer, /*nbr=*/16);

    std::vector<std::thread> workers;
    workers.reserve(num_layers);
    for (layer_id_t l = 0; l < num_layers; ++l) {
        workers.emplace_back([l, &hg] {
            auto& layer = hg->get_layer_graph(l);
            for (vertex_num_t i = 0; i < per_layer; ++i) {
                layer.add_vertex(i + l * 1000u);
            }
        });
    }
    for (auto& w : workers) w.join();

    for (layer_id_t l = 0; l < num_layers; ++l) {
        const auto& layer = hg->get_layer_graph(l);
        EXPECT_EQ(layer.get_num_vertices(), per_layer);
        for (vertex_num_t i = 0; i < per_layer; ++i) {
            EXPECT_EQ(layer.get_inter_layer_link(i), i + l * 1000u);
        }
    }
}

TEST_F(HierarchicalGraphTest, AddNbrUnderConcurrentInsertion) {
    // Insert vertices, then concurrently add neighbors. The per-vertex
    // spinlock in InternalGraph should prevent corruption.
    constexpr vertex_num_t num_v = 5'000;
    constexpr vertex_num_t max_nbr = 32;
    auto hg = make_hg(/*num_layers=*/2,
                      /*max=*/num_v,
                      /*nbr=*/max_nbr);

    auto& layer = hg->get_layer_graph(0);
    for (vertex_num_t i = 0; i < num_v; ++i) {
        layer.add_vertex(i);
    }

    // Each thread adds neighbors targeting the same set of "hot" vertices
    // to maximize lock contention. Use a simple FIFO pruner.
    auto fifo_prune = [](lnbr_t* slots, lnbr_t new_nbr) -> uint64_t {
        for (vertex_num_t i = 1; i < max_nbr; ++i) slots[i - 1] = slots[i];
        slots[max_nbr - 1] = new_nbr;
        return max_nbr;
    };

    constexpr vertex_id_t hot_src = 0;
    constexpr uint32_t num_threads = 8;
    constexpr uint32_t per_thread_inserts = 5000;

    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, &layer, &fifo_prune] {
            for (uint32_t i = 0; i < per_thread_inserts; ++i) {
                const vertex_id_t target = (t * per_thread_inserts + i) % num_v;
                layer.add_nbr(hot_src, lnbr_t(target, target), fifo_prune);
            }
        });
    }
    for (auto& th : threads) th.join();

    // After all the inserts, the hot vertex's array must hold exactly
    // max_nbr valid neighbors with no sentinel slot in [0, max_nbr).
    EXPECT_EQ(layer.num_valid_nbrs(hot_src), max_nbr);
    auto block = layer.fetch_nbrs(hot_src);
    for (vertex_num_t i = 1; i <= max_nbr; ++i) {
        EXPECT_NE(block[i], base_traits_t::invalid_lnbr)
            << "Slot " << i << " unexpectedly sentinel after contended insertion";
    }
}

// ============================================================
//  Larger scale stress: many layers, many vertices
// ============================================================

TEST_F(HierarchicalGraphTest, LargeScaleManyLayersManyVertices) {
    constexpr layer_num_t num_layers = 8;
    constexpr vertex_num_t per_layer = 100'000;
    constexpr vertex_num_t max_nbr = 16;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto hg = make_hg(num_layers, per_layer, max_nbr);

    // Concurrently populate every layer in parallel.
    tbb::parallel_for(
        tbb::blocked_range<layer_id_t>(0, num_layers),
        [&](const tbb::blocked_range<layer_id_t>& lr) {
            for (layer_id_t l = lr.begin(); l < lr.end(); ++l) {
                auto& layer = hg->get_layer_graph(l);
                tbb::parallel_for(
                    tbb::blocked_range<vertex_num_t>(0, per_layer),
                    [&](const tbb::blocked_range<vertex_num_t>& vr) {
                        for (vertex_num_t i = vr.begin(); i < vr.end(); ++i) {
                            layer.add_vertex(i);
                        }
                    }
                );
            }
        }
    );
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    ARTEA_INFO(fmt::format(
        "HierarchicalGraph: {} layers x {} vertices populated in {} ms",
        num_layers, per_layer, ms));

    // Each layer should have exactly per_layer vertices and the union of
    // their inter_layer_links should be a permutation of [0, per_layer).
    for (layer_id_t l = 0; l < num_layers; ++l) {
        const auto& layer = hg->get_layer_graph(l);
        EXPECT_EQ(layer.get_num_vertices(), per_layer);

        std::vector<uint8_t> seen(per_layer, 0);
        for (vertex_num_t v = 0; v < per_layer; ++v) {
            const vertex_id_t lower = layer.get_inter_layer_link(v);
            ASSERT_LT(lower, per_layer);
            seen[lower]++;
        }
        for (vertex_num_t i = 0; i < per_layer; ++i) {
            ASSERT_EQ(seen[i], 1) << "layer " << l << " lower " << i;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
