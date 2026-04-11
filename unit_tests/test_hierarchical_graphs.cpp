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
 * @FilePath: /Artea/unit_tests/test_hierarchical_graphs.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Tests for dynamic::HierarchicalGraph, compact::HierarchicalGraph,
 *               and HierarchicalGraphCompactor.
 */

#include <atomic>
#include <memory>
#include <vector>
#include <gtest/gtest.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

// ============================================================
//  Helpers
// ============================================================

static auto make_dynamic_hg(
    const layer_num_t num_layers,
    const vertex_num_t vertices_per_layer = 128,
    const vertex_num_t max_nbr_size = 8
) -> std::unique_ptr<dynamic::hierarchical_graph_t> {
    auto hg = std::make_unique<dynamic::hierarchical_graph_t>(num_layers);
    hg->grow_layers(num_layers, [&](const layer_id_t) {
        return std::make_unique<dynamic::internal_graph_t>(
            vertices_per_layer, max_nbr_size);
    });
    return hg;
}

// ============================================================
//  dynamic::HierarchicalGraph
// ============================================================

class DynamicHierarchicalGraphTest : public ::testing::Test {};

TEST_F(DynamicHierarchicalGraphTest, ConstructionEmpty) {
    dynamic::hierarchical_graph_t hg(/*max_layers=*/0);
    EXPECT_EQ(hg.get_num_layers(), 0u);
    EXPECT_EQ(hg.max_layers(), 0u);
}

TEST_F(DynamicHierarchicalGraphTest, ConstructionReservesCapacity) {
    dynamic::hierarchical_graph_t hg(/*max_layers=*/5);
    EXPECT_EQ(hg.get_num_layers(), 0u);
    EXPECT_EQ(hg.max_layers(), 5u);
}

TEST_F(DynamicHierarchicalGraphTest, GrowLayersProgressively) {
    dynamic::hierarchical_graph_t hg(/*max_layers=*/4);
    auto factory = [](const layer_id_t) {
        return std::make_unique<dynamic::internal_graph_t>(/*max=*/128, /*nbr=*/8);
    };

    EXPECT_EQ(hg.grow_layers(2, factory), 2u);
    EXPECT_EQ(hg.get_num_layers(), 2u);

    // Idempotent: requesting ≤ current count is a no-op.
    EXPECT_EQ(hg.grow_layers(2, factory), 2u);

    // Grow further.
    EXPECT_EQ(hg.grow_layers(4, factory), 4u);
    EXPECT_EQ(hg.get_num_layers(), 4u);
}

TEST_F(DynamicHierarchicalGraphTest, GrowLayersConcurrent) {
    dynamic::hierarchical_graph_t hg(/*max_layers=*/8);
    std::atomic<uint32_t> factory_calls{0};

    auto factory = [&](const layer_id_t) {
        factory_calls.fetch_add(1, std::memory_order_relaxed);
        return std::make_unique<dynamic::internal_graph_t>(/*max=*/64, /*nbr=*/8);
    };

    const layer_num_t target = 8;
    tbb::parallel_for(
        tbb::blocked_range<int>(0, 16),
        [&](const tbb::blocked_range<int>&) {
            hg.grow_layers(target, factory);
        }
    );

    EXPECT_EQ(hg.get_num_layers(), target);
    // Factory must be called exactly `target` times (no double-allocation).
    EXPECT_EQ(factory_calls.load(), target);
}

TEST_F(DynamicHierarchicalGraphTest, CommitLayerAfterSetLayerGraph) {
    dynamic::hierarchical_graph_t hg(/*max_layers=*/3);

    hg.set_layer_graph(0, std::make_unique<dynamic::internal_graph_t>(/*max=*/256, /*nbr=*/16));
    EXPECT_EQ(hg.get_num_layers(), 0u);  // not yet committed

    hg.commit_layer(0);
    EXPECT_EQ(hg.get_num_layers(), 1u);

    hg.set_layer_graph(1, std::make_unique<dynamic::internal_graph_t>(/*max=*/256, /*nbr=*/16));
    hg.commit_layer(1);
    EXPECT_EQ(hg.get_num_layers(), 2u);
}

TEST_F(DynamicHierarchicalGraphTest, GetLayerGraphRoundTrip) {
    auto hg = make_dynamic_hg(3, 128, 8);

    for (layer_id_t l = 0; l < 3; ++l) {
        auto& layer = hg->get_layer_graph(l);
        EXPECT_EQ(layer.get_num_vertices(), 0u);
        EXPECT_EQ(layer.max_nbr_size(), 8u);
    }
}

TEST_F(DynamicHierarchicalGraphTest, TrimEmptyLayers) {
    auto hg = make_dynamic_hg(5, 128, 8);
    EXPECT_EQ(hg->get_num_layers(), 5u);

    // Insert vertices only into layers 0 and 1.
    hg->get_layer_graph(0).add_vertex(0);
    hg->get_layer_graph(1).add_vertex(1);

    hg->trim_empty_layers();
    EXPECT_EQ(hg->get_num_layers(), 2u);
}

TEST_F(DynamicHierarchicalGraphTest, TrimEmptyLayersAllEmpty) {
    auto hg = make_dynamic_hg(3, 128, 8);
    hg->trim_empty_layers();
    EXPECT_EQ(hg->get_num_layers(), 0u);
}

TEST_F(DynamicHierarchicalGraphTest, TrimEmptyLayersNoneEmpty) {
    auto hg = make_dynamic_hg(3, 128, 8);
    for (layer_id_t l = 0; l < 3; ++l) {
        hg->get_layer_graph(l).add_vertex(l);
    }
    hg->trim_empty_layers();
    EXPECT_EQ(hg->get_num_layers(), 3u);
}

TEST_F(DynamicHierarchicalGraphTest, PerLayerVertexInsertion) {
    auto hg = make_dynamic_hg(3, 1024, 8);

    for (layer_id_t l = 0; l < 3; ++l) {
        auto& layer = hg->get_layer_graph(l);
        for (vertex_num_t i = 0; i < 100; ++i) {
            layer.add_vertex(l * 1000 + i);
        }
        EXPECT_EQ(layer.get_num_vertices(), 100u);
    }
}

TEST_F(DynamicHierarchicalGraphTest, ConcurrentVertexInsertionWithinLayer) {
    auto hg = make_dynamic_hg(1, 200'000, 8);
    auto& layer = hg->get_layer_graph(0);

    const vertex_num_t count = 100'000;
    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, count),
        [&](const tbb::blocked_range<vertex_num_t>& r) {
            for (vertex_num_t i = r.begin(); i < r.end(); ++i) {
                layer.add_vertex(i);
            }
        }
    );

    EXPECT_EQ(layer.get_num_vertices(), count);
}

TEST_F(DynamicHierarchicalGraphTest, IndependentLayersInParallel) {
    const layer_num_t num_layers = 4;
    const vertex_num_t per_layer = 20'000;
    auto hg = make_dynamic_hg(num_layers, per_layer * 2, 8);

    tbb::parallel_for(
        tbb::blocked_range<layer_id_t>(0, num_layers),
        [&](const tbb::blocked_range<layer_id_t>& r) {
            for (layer_id_t l = r.begin(); l < r.end(); ++l) {
                auto& layer = hg->get_layer_graph(l);
                for (vertex_num_t i = 0; i < per_layer; ++i) {
                    layer.add_vertex(l * per_layer + i);
                }
            }
        }
    );

    for (layer_id_t l = 0; l < num_layers; ++l) {
        EXPECT_EQ(hg->get_layer_graph(l).get_num_vertices(), per_layer);
    }
}

TEST_F(DynamicHierarchicalGraphTest, AddNbrUnderConcurrentInsertion) {
    auto hg = make_dynamic_hg(1, 10'000, 8);
    auto& layer = hg->get_layer_graph(0);

    const vertex_num_t num_verts = 5000;
    for (vertex_num_t i = 0; i < num_verts; ++i) {
        layer.add_vertex(i);
    }

    // Pick a hot vertex and have 8 threads contend on it.
    const vertex_id_t hot = 0;
    auto prune_fn = [](lnbr_t* slots, lnbr_t new_nbr) -> uint64_t {
        slots[7] = new_nbr;
        return 8;
    };

    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, 10'000),
        [&](const tbb::blocked_range<vertex_num_t>& r) {
            for (vertex_num_t i = r.begin(); i < r.end(); ++i) {
                const vertex_id_t nbr_bv = (i % (num_verts - 1)) + 1;
                layer.add_nbr(hot, lnbr_t(nbr_bv, nbr_bv), prune_fn);
            }
        }
    );

    const uint64_t count = layer.num_valid_nbrs(hot);
    EXPECT_GE(count, 1u);
    EXPECT_LE(count, 8u);
}

TEST_F(DynamicHierarchicalGraphTest, LargeScaleStress) {
    const layer_num_t num_layers = 4;
    const vertex_num_t per_layer = 50'000;
    auto hg = make_dynamic_hg(num_layers, per_layer * 2, 16);

    tbb::parallel_for(
        tbb::blocked_range<layer_id_t>(0, num_layers),
        [&](const tbb::blocked_range<layer_id_t>& r) {
            for (layer_id_t l = r.begin(); l < r.end(); ++l) {
                auto& layer = hg->get_layer_graph(l);
                for (vertex_num_t i = 0; i < per_layer; ++i) {
                    layer.add_vertex(l * per_layer + i);
                }
            }
        }
    );

    for (layer_id_t l = 0; l < num_layers; ++l) {
        EXPECT_EQ(hg->get_layer_graph(l).get_num_vertices(), per_layer);
    }
}

// ============================================================
//  compact::HierarchicalGraph
// ============================================================

class CompactHierarchicalGraphTest : public ::testing::Test {};

TEST_F(CompactHierarchicalGraphTest, Construction) {
    compact::hierarchical_graph_t hg(3);
    EXPECT_EQ(hg.get_num_layers(), 3u);
}

TEST_F(CompactHierarchicalGraphTest, SetAndGetLayerGraph) {
    compact::hierarchical_graph_t hg(2);

    hg.set_layer_graph(0, std::make_unique<compact::internal_graph_t>(100, 8));
    hg.set_layer_graph(1, std::make_unique<compact::internal_graph_t>(50, 8));

    EXPECT_EQ(hg.get_layer_graph(0).get_num_vertices(), 100u);
    EXPECT_EQ(hg.get_layer_graph(1).get_num_vertices(), 50u);
}

TEST_F(CompactHierarchicalGraphTest, MoveSemantics) {
    compact::hierarchical_graph_t hg(2);
    hg.set_layer_graph(0, std::make_unique<compact::internal_graph_t>(100, 8));
    hg.set_layer_graph(1, std::make_unique<compact::internal_graph_t>(50, 8));

    compact::hierarchical_graph_t moved = std::move(hg);
    EXPECT_EQ(moved.get_num_layers(), 2u);
    EXPECT_EQ(moved.get_layer_graph(0).get_num_vertices(), 100u);
    EXPECT_EQ(moved.get_layer_graph(1).get_num_vertices(), 50u);
}

TEST_F(CompactHierarchicalGraphTest, NeighborReadWrite) {
    compact::hierarchical_graph_t hg(1);
    hg.set_layer_graph(0, std::make_unique<compact::internal_graph_t>(10, 4));

    auto& layer = hg.get_layer_graph(0);
    auto nbrs = layer.fetch_nbrs(0);
    nbrs[0] = lnbr_t(100, 200);
    nbrs[1] = lnbr_t(101, 201);

    auto nbrs_read = layer.fetch_nbrs(0);
    EXPECT_EQ(nbrs_read[0].base_vid, 100u);
    EXPECT_EQ(nbrs_read[1].base_vid, 101u);
}

// ============================================================
//  HierarchicalGraphCompactor
//  (dynamic::HierarchicalGraph -> compact::HierarchicalGraph)
// ============================================================

class HierarchicalGraphCompactorTest : public ::testing::Test {
protected:
    static constexpr layer_num_t  num_layers     = 3;
    static constexpr vertex_num_t max_per_layer  = 1000;
    static constexpr vertex_num_t max_nbr_size   = 16;
    static constexpr vertex_num_t extracted_size  = 8;

    void SetUp() override {
        src_ = make_dynamic_hg(num_layers, max_per_layer, max_nbr_size);

        // Populate each layer with deterministic vertices and neighbors.
        for (layer_id_t l = 0; l < num_layers; ++l) {
            auto& layer = src_->get_layer_graph(l);
            const vertex_num_t n = max_per_layer / (l + 1);  // fewer at top
            for (vertex_num_t i = 0; i < n; ++i) {
                const vertex_id_t v = layer.add_vertex(l * 10000 + i);

                // Write a few neighbors.
                const vertex_num_t nbr_count = std::min<vertex_num_t>(i % (max_nbr_size + 1), max_nbr_size);
                auto block = layer.fetch_nbrs(v);
                for (vertex_num_t j = 0; j < nbr_count; ++j) {
                    block[1 + j] = lnbr_t(
                        l * 10000 + static_cast<vertex_id_t>(j),
                        static_cast<vertex_id_t>(j));
                }
                layer.num_valid_nbrs(v, nbr_count);

                // Set inter-layer link.
                layer.set_inter_layer_link(v, i + 7);
            }
        }
    }

    std::unique_ptr<dynamic::hierarchical_graph_t> src_;
};

TEST_F(HierarchicalGraphCompactorTest, BasicCompaction) {
    auto compact = hierarchical_graph_compactor_t::compact_graph(*src_, extracted_size);

    EXPECT_EQ(compact.get_num_layers(), num_layers);

    for (layer_id_t l = 0; l < num_layers; ++l) {
        const auto& src_layer = src_->get_layer_graph(l);
        const auto& compact_layer = compact.get_layer_graph(l);

        EXPECT_EQ(compact_layer.get_num_vertices(), src_layer.get_num_vertices());
        EXPECT_EQ(compact_layer.max_nbr_size(), extracted_size);
    }
}

TEST_F(HierarchicalGraphCompactorTest, NeighborValuesPreserved) {
    auto compact = hierarchical_graph_compactor_t::compact_graph(*src_, extracted_size);

    for (layer_id_t l = 0; l < num_layers; ++l) {
        const auto& src_layer = src_->get_layer_graph(l);
        const auto& compact_layer = compact.get_layer_graph(l);
        const vertex_num_t n = src_layer.get_num_vertices();

        for (vertex_id_t v = 0; v < n; ++v) {
            const uint64_t valid_count = src_layer.num_valid_nbrs(v);
            const uint64_t expected_copied = std::min(
                static_cast<uint64_t>(extracted_size), valid_count);

            auto src_block = src_layer.fetch_nbrs(v);
            auto compact_nbrs = compact_layer.fetch_nbrs(v);

            for (uint64_t j = 0; j < expected_copied; ++j) {
                EXPECT_EQ(compact_nbrs[j].base_vid, src_block[1 + j].base_vid);
                EXPECT_EQ(compact_nbrs[j].layer_vid, src_block[1 + j].layer_vid);
            }
            for (uint64_t j = expected_copied; j < extracted_size; ++j) {
                EXPECT_EQ(compact_nbrs[j], base_traits_t::invalid_lnbr);
            }
        }
    }
}

TEST_F(HierarchicalGraphCompactorTest, InterLayerLinksPreserved) {
    auto compact = hierarchical_graph_compactor_t::compact_graph(*src_, extracted_size);

    for (layer_id_t l = 0; l < num_layers; ++l) {
        const auto& src_layer = src_->get_layer_graph(l);
        const auto& compact_layer = compact.get_layer_graph(l);
        const vertex_num_t n = src_layer.get_num_vertices();

        for (vertex_id_t v = 0; v < n; ++v) {
            EXPECT_EQ(compact_layer.get_inter_layer_link(v),
                      src_layer.get_inter_layer_link(v));
        }
    }
}

TEST_F(HierarchicalGraphCompactorTest, EmptyLayersHandled) {
    // Create a hierarchical graph with an empty top layer.
    auto hg = make_dynamic_hg(3, 1000, 8);
    hg->get_layer_graph(0).add_vertex(0);
    hg->get_layer_graph(1).add_vertex(1);
    // Layer 2 is empty.

    auto compact = hierarchical_graph_compactor_t::compact_graph(*hg, 4);
    EXPECT_EQ(compact.get_num_layers(), 3u);
    EXPECT_EQ(compact.get_layer_graph(0).get_num_vertices(), 1u);
    EXPECT_EQ(compact.get_layer_graph(1).get_num_vertices(), 1u);
    EXPECT_EQ(compact.get_layer_graph(2).get_num_vertices(), 0u);
}

TEST_F(HierarchicalGraphCompactorTest, CompactIsMovable) {
    auto compact = hierarchical_graph_compactor_t::compact_graph(*src_, extracted_size);
    const vertex_num_t n0 = compact.get_layer_graph(0).get_num_vertices();

    compact::hierarchical_graph_t moved = std::move(compact);
    EXPECT_EQ(moved.get_num_layers(), num_layers);
    EXPECT_EQ(moved.get_layer_graph(0).get_num_vertices(), n0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
