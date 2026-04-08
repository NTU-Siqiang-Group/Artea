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
 * @FilePath: /Artea/unit_tests/test_hierarchical_graph_v2.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Tests for HierarchicalGraphV2: layer container, atomic
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

class HierarchicalGraphV2Test : public ::testing::Test {
protected:
    /**
     * @brief Build a HierarchicalGraphV2 with @p num_layers layers, each
     *        layer pre-allocating @p max_num_vertices_per_layer vertices
     *        with neighbor capacity @p max_nbr_size.
     */
    static auto make_hg(
        const layer_num_t num_layers,
        const vertex_num_t max_num_vertices_per_layer,
        const vertex_num_t max_nbr_size
    ) -> std::unique_ptr<hierarchical_graph_v2_t> {
        auto hg = std::make_unique<hierarchical_graph_v2_t>(num_layers);
        for (layer_id_t l = 0; l < num_layers; ++l) {
            hg->set_layer_graph(
                l,
                std::make_unique<internal_graph_t>(
                    max_num_vertices_per_layer, max_nbr_size
                )
            );
        }
        return hg;
    }
};

// ============================================================
//  Construction & layer container basics
// ============================================================

TEST_F(HierarchicalGraphV2Test, ConstructionEmpty) {
    hierarchical_graph_v2_t hg(/*num_layers=*/0);
    EXPECT_EQ(hg.get_num_layers(), 0u);
    EXPECT_EQ(hg.get_entry_point(), base_traits_t::invalid_lnbr);
}

TEST_F(HierarchicalGraphV2Test, ConstructionPreallocatesLayerSlots) {
    hierarchical_graph_v2_t hg(/*num_layers=*/5);
    EXPECT_EQ(hg.get_num_layers(), 5u);
    EXPECT_EQ(hg.get_entry_point(), base_traits_t::invalid_lnbr);
    // All layer slots are default-constructed unique_ptrs (nullptr).
    for (layer_id_t l = 0; l < 5; ++l) {
        EXPECT_EQ(hg.get_layer_graphs()[l].get(), nullptr);
    }
}

TEST_F(HierarchicalGraphV2Test, SetGetLayerGraphRoundTrip) {
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

TEST_F(HierarchicalGraphV2Test, GetLayerGraphsConstAndMutable) {
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
//  Atomic entry point
// ============================================================

TEST_F(HierarchicalGraphV2Test, EntryPointInitiallyInvalid) {
    hierarchical_graph_v2_t hg(/*num_layers=*/1);
    const lnbr_t ep = hg.get_entry_point();
    EXPECT_EQ(ep, base_traits_t::invalid_lnbr);
    EXPECT_EQ(ep.base_vid, base_traits_t::invalid_vertex_id);
    EXPECT_EQ(ep.layer_vid, base_traits_t::invalid_vertex_id);
}

TEST_F(HierarchicalGraphV2Test, UpdateEntryPointSerial) {
    hierarchical_graph_v2_t hg(/*num_layers=*/1);

    hg.update_entry_point(lnbr_t(42, 7));
    {
        auto ep = hg.get_entry_point();
        EXPECT_EQ(ep.base_vid, 42u);
        EXPECT_EQ(ep.layer_vid, 7u);
    }

    hg.update_entry_point(lnbr_t(99, 100));
    {
        auto ep = hg.get_entry_point();
        EXPECT_EQ(ep.base_vid, 99u);
        EXPECT_EQ(ep.layer_vid, 100u);
    }
}

TEST_F(HierarchicalGraphV2Test, UpdateEntryPointConcurrent) {
    // Many threads racing on update_entry_point — final value must be one
    // of the values that was actually written, never a torn read.
    hierarchical_graph_v2_t hg(/*num_layers=*/1);

    constexpr uint32_t num_threads = 16;
    constexpr uint32_t writes_per_thread = 10'000;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (uint32_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, &hg] {
            for (uint32_t i = 0; i < writes_per_thread; ++i) {
                // Encode (thread, iter) into the lnbr_t fields so we can
                // verify the final value below.
                hg.update_entry_point(lnbr_t(t, i));
            }
        });
    }
    for (auto& th : threads) th.join();

    const lnbr_t final_ep = hg.get_entry_point();
    EXPECT_LT(final_ep.base_vid, num_threads);
    EXPECT_LT(final_ep.layer_vid, writes_per_thread);
}

TEST_F(HierarchicalGraphV2Test, ConcurrentReadWriteEntryPointNoTear) {
    // Reader threads watch the entry point while writer threads update it.
    // Each writer always uses (base_vid == layer_vid) so any torn read
    // (different halves from different writes) would be detectable.
    hierarchical_graph_v2_t hg(/*num_layers=*/1);
    hg.update_entry_point(lnbr_t(0, 0));

    constexpr uint32_t num_writers = 4;
    constexpr uint32_t num_readers = 4;
    constexpr uint32_t writes_per_writer = 50'000;
    constexpr uint32_t reads_per_reader  = 200'000;

    std::atomic<bool> stop{false};
    std::atomic<uint32_t> tear_count{0};

    std::vector<std::thread> writers;
    for (uint32_t w = 0; w < num_writers; ++w) {
        writers.emplace_back([w, &hg] {
            for (uint32_t i = 0; i < writes_per_writer; ++i) {
                const vertex_id_t v = w * writes_per_writer + i;
                hg.update_entry_point(lnbr_t(v, v));
            }
        });
    }

    std::vector<std::thread> readers;
    for (uint32_t r = 0; r < num_readers; ++r) {
        readers.emplace_back([&hg, &stop, &tear_count] {
            for (uint32_t i = 0; i < reads_per_reader; ++i) {
                if (stop.load(std::memory_order_relaxed)) break;
                const lnbr_t ep = hg.get_entry_point();
                if (ep.base_vid != ep.layer_vid) {
                    tear_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& th : writers) th.join();
    stop.store(true);
    for (auto& th : readers) th.join();

    EXPECT_EQ(tear_count.load(), 0u);
}

// ============================================================
//  Per-layer InternalGraph operations through HierarchicalGraphV2
// ============================================================

TEST_F(HierarchicalGraphV2Test, PerLayerSerialVertexInsertion) {
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

TEST_F(HierarchicalGraphV2Test, ConcurrentVertexInsertionWithinLayer) {
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

TEST_F(HierarchicalGraphV2Test, IndependentLayersInParallel) {
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

TEST_F(HierarchicalGraphV2Test, AddNbrUnderConcurrentInsertion) {
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
//  Combined: layer mutation + concurrent entry point updates
// ============================================================

TEST_F(HierarchicalGraphV2Test, ConcurrentEntryPointUpdateDuringInsertion) {
    // Producer threads insert into layer 0; in parallel a separate writer
    // bumps the entry point to point at the most recently inserted vertex.
    constexpr vertex_num_t per_layer = 50'000;
    auto hg = make_hg(/*num_layers=*/3,
                      /*max=*/per_layer,
                      /*nbr=*/16);

    auto& layer0 = hg->get_layer_graph(0);
    std::atomic<bool> done{false};

    std::thread ep_writer([&] {
        std::mt19937 rng(2026);
        while (!done.load(std::memory_order_relaxed)) {
            const vertex_num_t cur = layer0.get_num_vertices();
            if (cur > 0) {
                const vertex_id_t v = rng() % cur;
                hg->update_entry_point(lnbr_t(v, v));
            }
        }
    });

    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, per_layer),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                layer0.add_vertex(i);
            }
        }
    );

    done.store(true);
    ep_writer.join();

    EXPECT_EQ(layer0.get_num_vertices(), per_layer);
    const lnbr_t ep = hg->get_entry_point();
    // Either still at the initial invalid value (if writer never observed
    // any insertion) or a valid (v, v) tuple. Verify no torn read.
    if (ep != base_traits_t::invalid_lnbr) {
        EXPECT_EQ(ep.base_vid, ep.layer_vid);
        EXPECT_LT(ep.base_vid, per_layer);
    }
}

// ============================================================
//  Larger scale stress: many layers, many vertices
// ============================================================

TEST_F(HierarchicalGraphV2Test, LargeScaleManyLayersManyVertices) {
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
        "HierarchicalGraphV2: {} layers x {} vertices populated in {} ms",
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

    // Set entry point pointing into the top layer.
    const layer_id_t top = num_layers - 1;
    hg->update_entry_point(lnbr_t(top, per_layer / 2));
    EXPECT_EQ(hg->get_entry_point(), lnbr_t(top, per_layer / 2));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
