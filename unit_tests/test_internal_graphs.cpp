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
 * @FilePath: /Artea/unit_tests/test_internal_graphs.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Tests for CompactInternalGraph and InternalGraph.
 */

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <gtest/gtest.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

// ============================================================
//  CompactInternalGraph Tests
// ============================================================

class CompactInternalGraphTest : public ::testing::Test {
protected:
    static constexpr vertex_num_t num_vertices = 1'000'000;
    static constexpr vertex_num_t max_nbr_size = 32;

    void SetUp() override {
        graph_ = std::make_unique<compact::internal_graph_t>(num_vertices, max_nbr_size);
    }

    std::unique_ptr<compact::internal_graph_t> graph_;
};

TEST_F(CompactInternalGraphTest, Construction) {
    EXPECT_EQ(graph_->get_num_vertices(), num_vertices);
    EXPECT_EQ(graph_->max_nbr_size(), max_nbr_size);
}

TEST_F(CompactInternalGraphTest, InitializedToInvalid) {
    std::atomic<uint32_t> errors{0};
    tbb::parallel_for(
        tbb::blocked_range<vertex_id_t>(0, num_vertices),
        [&](const tbb::blocked_range<vertex_id_t>& range) {
            for (vertex_id_t v = range.begin(); v < range.end(); ++v) {
                auto nbrs = graph_->fetch_nbrs(v);
                if (nbrs.size() != max_nbr_size) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                for (vertex_num_t i = 0; i < max_nbr_size; ++i) {
                    if (nbrs[i].base_vid != base_traits_t::invalid_vertex_id ||
                        nbrs[i].layer_vid != base_traits_t::invalid_vertex_id) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }
                }
            }
        }
    );
    EXPECT_EQ(errors.load(), 0u);
}

TEST_F(CompactInternalGraphTest, WriteAndReadNeighbors) {
    const vertex_id_t src = 42;
    auto nbrs = graph_->fetch_nbrs(src);

    // Write some neighbors
    nbrs[0] = lnbr_t(100, 200);
    nbrs[1] = lnbr_t(101, 201);
    nbrs[2] = lnbr_t(102, 202);

    // Read back
    auto nbrs_read = graph_->fetch_nbrs(src);
    EXPECT_EQ(nbrs_read[0].base_vid, 100u);
    EXPECT_EQ(nbrs_read[0].layer_vid, 200u);
    EXPECT_EQ(nbrs_read[1].base_vid, 101u);
    EXPECT_EQ(nbrs_read[1].layer_vid, 201u);
    EXPECT_EQ(nbrs_read[2].base_vid, 102u);
    EXPECT_EQ(nbrs_read[2].layer_vid, 202u);

    // Remaining should still be invalid
    for (vertex_num_t i = 3; i < max_nbr_size; ++i) {
        EXPECT_EQ(nbrs_read[i], base_traits_t::invalid_lnbr);
    }
}

TEST_F(CompactInternalGraphTest, InterLayerLinks) {
    // Initially all invalid (parallel check)
    std::atomic<uint32_t> errors{0};
    tbb::parallel_for(
        tbb::blocked_range<vertex_id_t>(0, num_vertices),
        [&](const tbb::blocked_range<vertex_id_t>& range) {
            for (vertex_id_t v = range.begin(); v < range.end(); ++v) {
                if (graph_->get_inter_layer_link(v) != base_traits_t::invalid_vertex_id) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    );
    EXPECT_EQ(errors.load(), 0u);

    // Set some links
    graph_->set_inter_layer_link(0, 10);
    graph_->set_inter_layer_link(50, 500);
    graph_->set_inter_layer_link(num_vertices - 1, 0);

    EXPECT_EQ(graph_->get_inter_layer_link(0), 10u);
    EXPECT_EQ(graph_->get_inter_layer_link(50), 500u);
    EXPECT_EQ(graph_->get_inter_layer_link(num_vertices - 1), 0u);
    EXPECT_EQ(graph_->get_inter_layer_link(1), base_traits_t::invalid_vertex_id);
}

TEST_F(CompactInternalGraphTest, MaxNbrSizeGetterSetter) {
    EXPECT_EQ(graph_->max_nbr_size(), max_nbr_size);
    graph_->max_nbr_size(16);
    EXPECT_EQ(graph_->max_nbr_size(), 16u);
}

TEST_F(CompactInternalGraphTest, CsrNbrsAccessor) {
    auto& csr = graph_->get_csr_nbrs();
    EXPECT_EQ(csr.size(), static_cast<size_t>(num_vertices) * max_nbr_size);

    const auto& const_csr = std::as_const(*graph_).get_csr_nbrs();
    EXPECT_EQ(const_csr.size(), static_cast<size_t>(num_vertices) * max_nbr_size);
}

TEST_F(CompactInternalGraphTest, InterLayerLinksAccessor) {
    auto& links = graph_->get_inter_layer_links();
    EXPECT_EQ(links.size(), num_vertices);

    const auto& const_links = std::as_const(*graph_).get_inter_layer_links();
    EXPECT_EQ(const_links.size(), num_vertices);
}

TEST_F(CompactInternalGraphTest, VertexIsolation) {
    // Write to vertex 500000, verify vertex 500001 is unaffected
    const vertex_id_t target = num_vertices / 2;
    auto nbrs_target = graph_->fetch_nbrs(target);
    for (vertex_num_t i = 0; i < max_nbr_size; ++i) {
        nbrs_target[i] = lnbr_t(i, i + 1000);
    }

    auto nbrs_next = graph_->fetch_nbrs(target + 1);
    for (vertex_num_t i = 0; i < max_nbr_size; ++i) {
        EXPECT_EQ(nbrs_next[i], base_traits_t::invalid_lnbr);
    }
}

TEST_F(CompactInternalGraphTest, SentinelBasedTraversal) {
    const vertex_id_t src = 7;
    auto nbrs = graph_->fetch_nbrs(src);

    // Write 5 valid neighbors
    const vertex_num_t valid_count = 5;
    for (vertex_num_t i = 0; i < valid_count; ++i) {
        nbrs[i] = lnbr_t(i * 10, i);
    }
    // The rest remain as invalid sentinels

    // Sentinel-based traversal
    vertex_num_t count = 0;
    auto nbrs_read = graph_->fetch_nbrs(src);
    for (vertex_num_t i = 0; i < max_nbr_size; ++i) {
        if (nbrs_read[i].base_vid == base_traits_t::invalid_vertex_id) break;
        ++count;
    }
    EXPECT_EQ(count, valid_count);
}

TEST_F(CompactInternalGraphTest, ParallelRead) {
    // Populate all vertices with deterministic data
    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        auto nbrs = graph_->fetch_nbrs(v);
        for (vertex_num_t i = 0; i < max_nbr_size; ++i) {
            nbrs[i] = lnbr_t(v + i, v * 2 + i);
        }
        graph_->set_inter_layer_link(v, v + 1);
    }

    // Parallel read verification
    std::atomic<uint32_t> errors{0};

    tbb::parallel_for(
        tbb::blocked_range<vertex_id_t>(0, num_vertices),
        [&](const tbb::blocked_range<vertex_id_t>& range) {
            for (vertex_id_t v = range.begin(); v < range.end(); ++v) {
                auto nbrs = graph_->fetch_nbrs(v);
                for (vertex_num_t i = 0; i < max_nbr_size; ++i) {
                    if (nbrs[i].base_vid != v + i ||
                        nbrs[i].layer_vid != v * 2 + i) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                if (graph_->get_inter_layer_link(v) != v + 1) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    );

    EXPECT_EQ(errors.load(), 0u);
}

// ============================================================
//  InternalGraph Tests
// ============================================================

class InternalGraphTest : public ::testing::Test {
protected:
    static constexpr vertex_num_t max_num_vertices = 1'000'000;
    static constexpr vertex_num_t max_nbr_size = 63;

    void SetUp() override {
        graph_ = std::make_unique<dynamic::internal_graph_t>(max_num_vertices, max_nbr_size);
    }

    std::unique_ptr<dynamic::internal_graph_t> graph_;
};

TEST_F(InternalGraphTest, Construction) {
    EXPECT_EQ(graph_->get_num_vertices(), 0u);
    EXPECT_EQ(graph_->get_max_num_vertices(), max_num_vertices);
    EXPECT_EQ(graph_->max_nbr_size(), max_nbr_size);
}

TEST_F(InternalGraphTest, AddVertex) {
    vertex_id_t layer_vid = graph_->add_vertex(100);
    EXPECT_EQ(layer_vid, 0u);
    EXPECT_EQ(graph_->get_num_vertices(), 1u);
    EXPECT_EQ(graph_->get_inter_layer_link(layer_vid), 100u);
}

TEST_F(InternalGraphTest, AddVertexInitializesNeighbors) {
    vertex_id_t layer_vid = graph_->add_vertex(0);

    auto nbrs = graph_->fetch_nbrs(layer_vid);
    // nbrs[0] is the atomic header, skip it
    for (vertex_num_t i = 1; i <= max_nbr_size; ++i) {
        EXPECT_EQ(nbrs[i], base_traits_t::invalid_lnbr);
    }
}

TEST_F(InternalGraphTest, NumValidNbrs) {
    vertex_id_t v = graph_->add_vertex(0);

    EXPECT_EQ(graph_->num_valid_nbrs(v), 0u);

    graph_->num_valid_nbrs(v, 5);
    EXPECT_EQ(graph_->num_valid_nbrs(v), 5u);

    graph_->num_valid_nbrs(v, 63);
    EXPECT_EQ(graph_->num_valid_nbrs(v), 63u);
}

TEST_F(InternalGraphTest, WriteAndReadNeighbors) {
    vertex_id_t v = graph_->add_vertex(42);

    auto nbrs = graph_->fetch_nbrs(v);
    // Write neighbors starting at index 1 (index 0 is header)
    nbrs[1] = lnbr_t(100, 200);
    nbrs[2] = lnbr_t(101, 201);
    nbrs[3] = lnbr_t(102, 202);
    graph_->num_valid_nbrs(v, 3);

    // Read back
    auto nbrs_read = graph_->fetch_nbrs(v);
    EXPECT_EQ(nbrs_read[1].base_vid, 100u);
    EXPECT_EQ(nbrs_read[1].layer_vid, 200u);
    EXPECT_EQ(nbrs_read[2].base_vid, 101u);
    EXPECT_EQ(nbrs_read[2].layer_vid, 201u);
    EXPECT_EQ(nbrs_read[3].base_vid, 102u);
    EXPECT_EQ(nbrs_read[3].layer_vid, 202u);
    EXPECT_EQ(graph_->num_valid_nbrs(v), 3u);
}

TEST_F(InternalGraphTest, SequentialAddMultipleVertices) {
    const vertex_num_t count = 1000;
    for (vertex_num_t i = 0; i < count; ++i) {
        vertex_id_t v = graph_->add_vertex(i * 10);
        EXPECT_EQ(v, i);
    }

    EXPECT_EQ(graph_->get_num_vertices(), count);

    for (vertex_id_t v = 0; v < count; ++v) {
        EXPECT_EQ(graph_->get_inter_layer_link(v), v * 10);
    }
}

TEST_F(InternalGraphTest, InterLayerLinkSetGet) {
    vertex_id_t v0 = graph_->add_vertex(10);
    vertex_id_t v1 = graph_->add_vertex(20);

    EXPECT_EQ(graph_->get_inter_layer_link(v0), 10u);
    EXPECT_EQ(graph_->get_inter_layer_link(v1), 20u);

    // Overwrite
    graph_->set_inter_layer_link(v0, 999);
    EXPECT_EQ(graph_->get_inter_layer_link(v0), 999u);
}

TEST_F(InternalGraphTest, MaxNbrSizeGetterSetter) {
    EXPECT_EQ(graph_->max_nbr_size(), max_nbr_size);
    graph_->max_nbr_size(16);
    EXPECT_EQ(graph_->max_nbr_size(), 16u);
}

TEST_F(InternalGraphTest, FetchNbrsSpanSize) {
    vertex_id_t v = graph_->add_vertex(0);

    auto nbrs = graph_->fetch_nbrs(v);
    // 1 header + max_nbr_size neighbor slots
    EXPECT_EQ(nbrs.size(), static_cast<size_t>(max_nbr_size) + 1);

    const auto& const_graph = *graph_;
    auto const_nbrs = const_graph.fetch_nbrs(v);
    EXPECT_EQ(const_nbrs.size(), static_cast<size_t>(max_nbr_size) + 1);
}

TEST_F(InternalGraphTest, VertexIsolation) {
    vertex_id_t v0 = graph_->add_vertex(0);
    vertex_id_t v1 = graph_->add_vertex(1);

    // Write to v0
    auto nbrs_0 = graph_->fetch_nbrs(v0);
    for (vertex_num_t i = 1; i <= max_nbr_size; ++i) {
        nbrs_0[i] = lnbr_t(i, i + 100);
    }
    graph_->num_valid_nbrs(v0, max_nbr_size);

    // v1 should still be all invalid
    auto nbrs_1 = graph_->fetch_nbrs(v1);
    for (vertex_num_t i = 1; i <= max_nbr_size; ++i) {
        EXPECT_EQ(nbrs_1[i], base_traits_t::invalid_lnbr);
    }
    EXPECT_EQ(graph_->num_valid_nbrs(v1), 0u);
}

TEST_F(InternalGraphTest, CsrNbrsAccessor) {
    auto& csr = graph_->get_csr_nbrs();
    EXPECT_EQ(csr.size(), static_cast<size_t>(max_num_vertices) * (max_nbr_size + 1));
}

TEST_F(InternalGraphTest, VertexInfoAccessor) {
    graph_->add_vertex(10);
    graph_->add_vertex(20);

    auto& info = graph_->get_vertex_info();
    EXPECT_EQ(info.size(), 2u);

    const auto& const_info = std::as_const(*graph_).get_vertex_info();
    EXPECT_EQ(const_info.size(), 2u);
}

// --- Parallel Tests for InternalGraph ---

TEST_F(InternalGraphTest, ParallelAddVertex) {
    const vertex_num_t count = max_num_vertices;

    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, count),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                vertex_id_t v = graph_->add_vertex(i);
                // Each thread writes its own neighbors
                auto nbrs = graph_->fetch_nbrs(v);
                for (vertex_num_t j = 1; j <= 5; ++j) {
                    nbrs[j] = lnbr_t(v + j, j);
                }
                graph_->num_valid_nbrs(v, 5);
            }
        }
    );

    EXPECT_EQ(graph_->get_num_vertices(), count);

    // Verify each vertex has correct num_valid_nbrs
    for (vertex_id_t v = 0; v < count; ++v) {
        EXPECT_EQ(graph_->num_valid_nbrs(v), 5u);
    }
}

TEST_F(InternalGraphTest, ParallelAddVertexInterLayerLinks) {
    const vertex_num_t count = max_num_vertices;

    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, count),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                graph_->add_vertex(i * 3);
            }
        }
    );

    EXPECT_EQ(graph_->get_num_vertices(), count);

    // Verify all inter-layer links are present (values i*3, order non-deterministic)
    std::vector<vertex_id_t> links;
    links.reserve(count);
    for (vertex_id_t v = 0; v < count; ++v) {
        links.push_back(graph_->get_inter_layer_link(v));
    }
    std::sort(links.begin(), links.end());

    std::vector<vertex_id_t> expected;
    expected.reserve(count);
    for (vertex_num_t i = 0; i < count; ++i) {
        expected.push_back(i * 3);
    }
    std::sort(expected.begin(), expected.end());

    EXPECT_EQ(links, expected);
}

TEST_F(InternalGraphTest, ParallelWriteThenParallelRead) {
    const vertex_num_t count = max_num_vertices;
    const vertex_num_t nbrs_per_vertex = 10;

    // Phase 1: Parallel write
    // Each thread adds a vertex and writes neighbors deterministically
    // Store (thread_index -> layer_vid) mapping for later verification
    std::vector<vertex_id_t> layer_vids(count);

    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, count),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                vertex_id_t v = graph_->add_vertex(i);
                layer_vids[i] = v;

                auto nbrs = graph_->fetch_nbrs(v);
                for (vertex_num_t j = 1; j <= nbrs_per_vertex; ++j) {
                    nbrs[j] = lnbr_t(i + j * 100, j);
                }
                graph_->num_valid_nbrs(v, nbrs_per_vertex);
            }
        }
    );

    EXPECT_EQ(graph_->get_num_vertices(), count);

    // Phase 2: Parallel read and verify
    std::atomic<uint32_t> errors{0};

    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, count),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                vertex_id_t v = layer_vids[i];

                if (graph_->num_valid_nbrs(v) != nbrs_per_vertex) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                if (graph_->get_inter_layer_link(v) != i) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }

                auto nbrs = graph_->fetch_nbrs(v);
                for (vertex_num_t j = 1; j <= nbrs_per_vertex; ++j) {
                    if (nbrs[j].base_vid != i + j * 100 ||
                        nbrs[j].layer_vid != j) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                // Remaining slots should be invalid
                for (vertex_num_t j = nbrs_per_vertex + 1;
                     j <= max_nbr_size; ++j) {
                    if (nbrs[j] != base_traits_t::invalid_lnbr) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    );

    EXPECT_EQ(errors.load(), 0u);
}

TEST_F(InternalGraphTest, ParallelAddVertexFullCapacityThenParallelRead) {
    // Fill to max capacity
    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, max_num_vertices),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                vertex_id_t v = graph_->add_vertex(i);
                auto nbrs = graph_->fetch_nbrs(v);
                for (vertex_num_t j = 1; j <= max_nbr_size; ++j) {
                    nbrs[j] = lnbr_t(i + j, j);
                }
                graph_->num_valid_nbrs(v, max_nbr_size);
            }
        }
    );

    EXPECT_EQ(graph_->get_num_vertices(), max_num_vertices);

    // Parallel read: verify every vertex has full neighbors and correct count
    std::atomic<uint32_t> errors{0};
    tbb::parallel_for(
        tbb::blocked_range<vertex_id_t>(0, max_num_vertices),
        [&](const tbb::blocked_range<vertex_id_t>& range) {
            for (vertex_id_t v = range.begin(); v < range.end(); ++v) {
                if (graph_->num_valid_nbrs(v) != max_nbr_size) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    );
    EXPECT_EQ(errors.load(), 0u);
}

// --- add_nbr Tests ---

// Simple pruning functor: drops the last neighbor, replaces with new_nbr.
auto simple_prune_fn = [](lnbr_t* slots, lnbr_t new_nbr) -> uint64_t {
    // Replace the last slot with new_nbr (count stays at max_nbr_size)
    // We don't know max_nbr_size here, but we can scan backward for simplicity.
    // In tests, the array is always full (63 entries), so slot[62] = new_nbr.
    slots[62] = new_nbr;
    return 63;
};

// Pruning functor that halves the array and appends the new neighbor.
auto halving_prune_fn = [](lnbr_t* slots, lnbr_t new_nbr) -> uint64_t {
    constexpr uint64_t max_nbr = 63;
    const uint64_t keep = max_nbr / 2;
    slots[keep] = new_nbr;
    return keep + 1;
};

TEST_F(InternalGraphTest, AddNbrSerial) {
    vertex_id_t v = graph_->add_vertex(0);

    for (vertex_num_t i = 0; i < max_nbr_size; ++i) {
        auto result = graph_->add_nbr(v, lnbr_t(i, i + 100), simple_prune_fn);
        EXPECT_EQ(result, dynamic::internal_graph_t::AddNbrResult::APPENDED);
    }

    EXPECT_EQ(graph_->num_valid_nbrs(v), max_nbr_size);

    // Verify neighbors
    auto nbrs = graph_->fetch_nbrs(v);
    for (vertex_num_t i = 0; i < max_nbr_size; ++i) {
        EXPECT_EQ(nbrs[i + 1].base_vid, i);
        EXPECT_EQ(nbrs[i + 1].layer_vid, i + 100);
    }
}

TEST_F(InternalGraphTest, AddNbrTriggersPruning) {
    vertex_id_t v = graph_->add_vertex(0);

    // Fill the array
    for (vertex_num_t i = 0; i < max_nbr_size; ++i) {
        graph_->add_nbr(v, lnbr_t(i, i), simple_prune_fn);
    }
    EXPECT_EQ(graph_->num_valid_nbrs(v), max_nbr_size);

    // Next add triggers pruning (halving_prune_fn keeps half + the new one)
    auto result = graph_->add_nbr(v, lnbr_t(999, 999), halving_prune_fn);
    EXPECT_EQ(result, dynamic::internal_graph_t::AddNbrResult::PRUNED);

    const uint64_t expected_count = max_nbr_size / 2 + 1;
    EXPECT_EQ(graph_->num_valid_nbrs(v), expected_count);

    // Verify the new neighbor is present
    auto nbrs = graph_->fetch_nbrs(v);
    EXPECT_EQ(nbrs[expected_count].base_vid, 999u);
    EXPECT_EQ(nbrs[expected_count].layer_vid, 999u);

    // Trailing slots should be invalid
    for (uint64_t i = expected_count + 1; i <= max_nbr_size; ++i) {
        EXPECT_EQ(nbrs[i], base_traits_t::invalid_lnbr);
    }
}

TEST_F(InternalGraphTest, ParallelAddNbrSameVertex) {
    vertex_id_t v = graph_->add_vertex(0);
    const vertex_num_t total_adds = 1'000'000;

    std::atomic<uint32_t> prune_count{0};
    auto counting_prune_fn = [&](lnbr_t* slots, lnbr_t new_nbr) -> uint64_t {
        prune_count.fetch_add(1, std::memory_order_relaxed);
        constexpr uint64_t max_nbr = 63;
        const uint64_t keep = max_nbr / 2;
        slots[keep] = new_nbr;
        return keep + 1;
    };

    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, total_adds),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                graph_->add_nbr(v, lnbr_t(i, i), counting_prune_fn);
            }
        }
    );

    // Count must be valid (in range [1, max_nbr_size])
    const uint64_t final_count = graph_->num_valid_nbrs(v);
    EXPECT_GE(final_count, 1u);
    EXPECT_LE(final_count, max_nbr_size);

    // Pruning must have been triggered many times
    EXPECT_GT(prune_count.load(), 0u);
}

TEST_F(InternalGraphTest, ParallelAddNbrMultipleVertices) {
    const vertex_num_t num_vertices = 10000;
    const vertex_num_t adds_per_vertex = 200;

    // Create vertices
    std::vector<vertex_id_t> vids(num_vertices);
    for (vertex_num_t i = 0; i < num_vertices; ++i) {
        vids[i] = graph_->add_vertex(i);
    }

    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, num_vertices),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                for (vertex_num_t j = 0; j < adds_per_vertex; ++j) {
                    graph_->add_nbr(vids[i], lnbr_t(j, j), halving_prune_fn);
                }
            }
        }
    );

    // Every vertex should have a valid count
    std::atomic<uint32_t> errors{0};
    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, num_vertices),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                const uint64_t count = graph_->num_valid_nbrs(vids[i]);
                if (count < 1 || count > max_nbr_size) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    );
    EXPECT_EQ(errors.load(), 0u);
}

TEST_F(InternalGraphTest, ParallelAddNbrSameVertexHighContention) {
    // Stress test: many threads hammer the SAME vertex
    vertex_id_t v = graph_->add_vertex(0);
    const vertex_num_t total_adds = 5'000'000;

    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, total_adds),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                graph_->add_nbr(v, lnbr_t(i % 10000, i % 10000), halving_prune_fn);
            }
        }
    );

    const uint64_t final_count = graph_->num_valid_nbrs(v);
    EXPECT_GE(final_count, 1u);
    EXPECT_LE(final_count, max_nbr_size);

    // All non-trailing slots should have valid (non-sentinel) data
    auto nbrs = graph_->fetch_nbrs(v);
    for (uint64_t i = 1; i <= final_count; ++i) {
        EXPECT_NE(nbrs[i].base_vid, base_traits_t::invalid_vertex_id);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
