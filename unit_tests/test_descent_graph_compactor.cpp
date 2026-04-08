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
 * @FilePath: /Artea/unit_tests/test_descent_graph_compactor.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Tests for DescentGraphCompactor (single-layer and hierarchical compaction).
 */

#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

// ============================================================
//  Single-layer (DescentGraph -> CompactDescentGraph) tests
// ============================================================

class DescentGraphCompactorTest : public ::testing::Test {
protected:
    static constexpr vertex_num_t num_vertices = 50'000;
    static constexpr vec_dim_t vec_dim = 32;
    static constexpr vertex_num_t max_nbr_size = 64;
    static constexpr vertex_num_t reserved_nbr_size = 96;

    void SetUp() override {
        vecs_ = std::make_unique<vector_array_t>(vec_dim);
        vecs_->reserve(num_vertices);

        std::mt19937 rng(2026);
        std::uniform_real_distribution<vec_ele_t> dist(0.0f, 1.0f);
        std::vector<vec_ele_t> buf(vec_dim);

        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            for (vec_dim_t d = 0; d < vec_dim; ++d) {
                buf[d] = dist(rng);
            }
            vecs_->append_vec(buf.data());
        }

        descent_graph_ = std::make_unique<conv_graph::index_t>(
            *vecs_,
            layer_config_t(max_nbr_size, reserved_nbr_size),
            conv_graph::pruning_config_t(1.0f, 0.0f),
            conv_graph::propagate_config_t(4, 14)
        );
    }

    /**
     * @brief Populate the descent graph deterministically: vertex i gets
     *        @c (i % (max_nbr_size + 1)) neighbors. Neighbor j has id
     *        @c ((i * 31 + j * 17) % num_vertices) and distance @c (i + j).
     */
    void populate_deterministic_neighbors() {
        auto& nbrs_arr = descent_graph_->get_nbrs_arr();
        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& range) {
                for (vertex_id_t v = range.begin(); v < range.end(); ++v) {
                    auto& nbrs = nbrs_arr[v];
                    nbrs.clear();
                    const vertex_num_t k = v % (max_nbr_size + 1);
                    for (vertex_num_t j = 0; j < k; ++j) {
                        const vertex_id_t nid = (v * 31u + j * 17u) % num_vertices;
                        nbrs.push_back(dnbr_t(nid, static_cast<distance_t>(v + j), true));
                    }
                }
            }
        );
    }

    std::unique_ptr<vector_array_t> vecs_;
    std::unique_ptr<conv_graph::index_t> descent_graph_;
};

TEST_F(DescentGraphCompactorTest, EmptyGraphAllSentinel) {
    // No neighbors populated; every vertex's compact row should be all sentinel.
    const vertex_num_t extracted = 32;
    auto compact = descent_graph_compactor_t::from_descent_graph(*descent_graph_, extracted);

    EXPECT_EQ(compact.get_num_vertices(), num_vertices);
    EXPECT_EQ(compact.get_extracted_nbr_size(), extracted);

    std::atomic<uint32_t> errors{0};
    tbb::parallel_for(
        tbb::blocked_range<vertex_id_t>(0, num_vertices),
        [&](const tbb::blocked_range<vertex_id_t>& range) {
            for (vertex_id_t v = range.begin(); v < range.end(); ++v) {
                auto row = compact.fetch_nbrs(v);
                for (vertex_num_t i = 0; i < extracted; ++i) {
                    if (row[i] != base_traits_t::invalid_vertex_id) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    );
    EXPECT_EQ(errors.load(), 0u);
}

TEST_F(DescentGraphCompactorTest, BasicConversionEqualSize) {
    populate_deterministic_neighbors();

    const vertex_num_t extracted = max_nbr_size;
    auto compact = descent_graph_compactor_t::from_descent_graph(*descent_graph_, extracted);

    EXPECT_EQ(compact.get_num_vertices(), num_vertices);
    EXPECT_EQ(compact.get_extracted_nbr_size(), extracted);

    const auto& nbrs_arr = descent_graph_->get_nbrs_arr();

    std::atomic<uint32_t> errors{0};
    tbb::parallel_for(
        tbb::blocked_range<vertex_id_t>(0, num_vertices),
        [&](const tbb::blocked_range<vertex_id_t>& range) {
            for (vertex_id_t v = range.begin(); v < range.end(); ++v) {
                const auto& src_nbrs = nbrs_arr[v];
                auto row = compact.fetch_nbrs(v);

                const vertex_num_t copy_count = std::min(
                    static_cast<vertex_num_t>(src_nbrs.size()), extracted
                );
                for (vertex_num_t i = 0; i < copy_count; ++i) {
                    if (row[i] != src_nbrs[i].get_id()) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                for (vertex_num_t i = copy_count; i < extracted; ++i) {
                    if (row[i] != base_traits_t::invalid_vertex_id) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    );
    EXPECT_EQ(errors.load(), 0u);
}

TEST_F(DescentGraphCompactorTest, ExtractedNbrSizeSmaller) {
    populate_deterministic_neighbors();

    const vertex_num_t extracted = 16;
    auto compact = descent_graph_compactor_t::from_descent_graph(*descent_graph_, extracted);

    const auto& nbrs_arr = descent_graph_->get_nbrs_arr();
    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        const auto& src_nbrs = nbrs_arr[v];
        auto row = compact.fetch_nbrs(v);
        EXPECT_EQ(row.size(), extracted);

        const vertex_num_t copy_count = std::min(
            static_cast<vertex_num_t>(src_nbrs.size()), extracted
        );
        for (vertex_num_t i = 0; i < copy_count; ++i) {
            EXPECT_EQ(row[i], src_nbrs[i].get_id());
        }
        for (vertex_num_t i = copy_count; i < extracted; ++i) {
            EXPECT_EQ(row[i], base_traits_t::invalid_vertex_id);
        }
    }
}

TEST_F(DescentGraphCompactorTest, ExtractedNbrSizeLargerThrows) {
    EXPECT_THROW({
        descent_graph_compactor_t::from_descent_graph(*descent_graph_, max_nbr_size + 1);
    }, std::runtime_error);
}

TEST_F(DescentGraphCompactorTest, NeighborOrderPreserved) {
    populate_deterministic_neighbors();

    const vertex_num_t extracted = 32;
    auto compact = descent_graph_compactor_t::from_descent_graph(*descent_graph_, extracted);

    // Sample a handful of vertices and verify slot-by-slot order matches.
    const auto& nbrs_arr = descent_graph_->get_nbrs_arr();
    const std::vector<vertex_id_t> probes = {0, 1, 7, 31, 1023, 9999, 49999};
    for (const auto v : probes) {
        const auto& src_nbrs = nbrs_arr[v];
        auto row = compact.fetch_nbrs(v);
        const vertex_num_t copy_count = std::min(
            static_cast<vertex_num_t>(src_nbrs.size()), extracted
        );
        for (vertex_num_t i = 0; i < copy_count; ++i) {
            EXPECT_EQ(row[i], src_nbrs[i].get_id())
                << "Order mismatch at vertex " << v << " slot " << i;
        }
    }
}

TEST_F(DescentGraphCompactorTest, SourceUnchangedAfterCompaction) {
    populate_deterministic_neighbors();

    auto compact = descent_graph_compactor_t::from_descent_graph(*descent_graph_, max_nbr_size);

    EXPECT_EQ(descent_graph_->get_num_vertices(), num_vertices);
    const auto& nbrs_arr = descent_graph_->get_nbrs_arr();
    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        const vertex_num_t expected_count = v % (max_nbr_size + 1);
        EXPECT_EQ(nbrs_arr[v].size(), expected_count);
    }
}

TEST_F(DescentGraphCompactorTest, LargeScaleParallelCorrectness) {
    populate_deterministic_neighbors();

    const vertex_num_t extracted = 32;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto compact = descent_graph_compactor_t::from_descent_graph(*descent_graph_, extracted);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    ARTEA_INFO(fmt::format(
        "DescentGraphCompactor: {} vertices, extracted_nbr_size={} took {} ms",
        num_vertices, extracted, ms));

    const auto& nbrs_arr = descent_graph_->get_nbrs_arr();
    std::atomic<uint32_t> errors{0};
    tbb::parallel_for(
        tbb::blocked_range<vertex_id_t>(0, num_vertices),
        [&](const tbb::blocked_range<vertex_id_t>& range) {
            for (vertex_id_t v = range.begin(); v < range.end(); ++v) {
                const auto& src_nbrs = nbrs_arr[v];
                auto row = compact.fetch_nbrs(v);
                const vertex_num_t copy_count = std::min(
                    static_cast<vertex_num_t>(src_nbrs.size()), extracted
                );
                for (vertex_num_t i = 0; i < copy_count; ++i) {
                    if (row[i] != src_nbrs[i].get_id()) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                for (vertex_num_t i = copy_count; i < extracted; ++i) {
                    if (row[i] != base_traits_t::invalid_vertex_id) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    );
    EXPECT_EQ(errors.load(), 0u);
}

// ============================================================
//  Hierarchical (HierarchicalGraph -> HierarchicalSearchGraph) tests
// ============================================================

class HierarchicalDescentGraphCompactorTest : public ::testing::Test {
protected:
    static constexpr vec_dim_t vec_dim = 16;
    static constexpr vertex_num_t bl_max_nbr_size = 48;
    static constexpr vertex_num_t bl_reserved_nbr_size = 64;
    static constexpr vertex_num_t ul_max_nbr_size = 24;
    static constexpr vertex_num_t ul_reserved_nbr_size = 32;

    /**
     * @brief Build a synthetic vector_array_t with @p n random vectors.
     */
    static auto make_random_vecs(const vertex_num_t n, const uint32_t seed)
        -> std::unique_ptr<vector_array_t>
    {
        auto vecs = std::make_unique<vector_array_t>(vec_dim);
        vecs->reserve(n);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<vec_ele_t> dist(0.0f, 1.0f);
        std::vector<vec_ele_t> buf(vec_dim);
        for (vertex_num_t i = 0; i < n; ++i) {
            for (vec_dim_t d = 0; d < vec_dim; ++d) buf[d] = dist(rng);
            vecs->append_vec(buf.data());
        }
        return vecs;
    }

    /**
     * @brief Populate a layer's neighbor array deterministically.
     *        Vertex v gets (v % (max_nbr_size + 1)) neighbors.
     */
    static void populate_layer(conv_graph::index_t& layer_graph, const vertex_num_t cap) {
        const vertex_num_t n = layer_graph.get_num_vertices();
        auto& nbrs_arr = layer_graph.get_nbrs_arr();
        for (vertex_id_t v = 0; v < n; ++v) {
            auto& nbrs = nbrs_arr[v];
            nbrs.clear();
            const vertex_num_t k = v % (cap + 1);
            for (vertex_num_t j = 0; j < k; ++j) {
                const vertex_id_t nid = (v * 13u + j * 7u) % n;
                nbrs.push_back(dnbr_t(nid, static_cast<distance_t>(v + j), true));
            }
        }
    }

    /**
     * @brief Build a synthetic multi-layer hierarchical graph.
     *
     * Layout (bottom-up shrinking sizes): layer_sizes[0] = bottom layer.
     * @return A fully populated artea_graph::index_t (not via the factory).
     */
    auto build_hierarchy(const std::vector<vertex_num_t>& layer_sizes)
        -> std::unique_ptr<artea_graph::index_t>
    {
        const layer_num_t num_layers = static_cast<layer_num_t>(layer_sizes.size());
        EXPECT_GE(num_layers, 1u);

        // Build bottom (base) vecs and store on the test fixture so the
        // reference held by hg outlives the test scope.
        base_vecs_ = make_random_vecs(layer_sizes[0], /*seed=*/12345);

        layer_config_t bl_cfg(bl_max_nbr_size, bl_reserved_nbr_size);
        layer_config_t ul_cfg(ul_max_nbr_size, ul_reserved_nbr_size);
        artea_graph::pruning_config_t prune_cfg(1.0f, 0.0f);
        artea_graph::propagate_config_t prop_cfg(4, 14);
        artea_graph::rnet_config_t rnet_cfg;

        auto hg = std::make_unique<artea_graph::index_t>(
            *base_vecs_, bl_cfg, ul_cfg, prune_cfg, prune_cfg, prop_cfg, rnet_cfg
        );

        // Step 1: append upper layer vecs to the manager (must complete before
        // we capture references to them; references to elements of the
        // _upper_layer_vecs std::vector would be invalidated by reallocation).
        auto& mgr = hg->get_hierarchy_manager();
        mgr.get_upper_layer_vecs().reserve(num_layers - 1);
        for (layer_id_t l = 1; l < num_layers; ++l) {
            auto upper = make_random_vecs(layer_sizes[l],
                                          /*seed=*/12345u + 100u * l);
            mgr.bottom_up_append(std::move(*upper));
        }

        // Step 2: resize the layer_graphs vector and populate each layer.
        hg->resize(num_layers);

        // Bottom layer.
        {
            auto bottom = std::make_unique<conv_graph::index_t>(
                *base_vecs_, bl_cfg,
                conv_graph::pruning_config_t(1.0f, 0.0f),
                conv_graph::propagate_config_t(4, 14)
            );
            populate_layer(*bottom, bl_max_nbr_size);
            hg->set_layer_graph(0, std::move(bottom));
        }

        // Upper layers.
        for (layer_id_t l = 1; l < num_layers; ++l) {
            const auto& upper_vecs_ref = mgr.get_layer_vecs(l);
            auto upper = std::make_unique<conv_graph::index_t>(
                upper_vecs_ref, ul_cfg,
                conv_graph::pruning_config_t(1.0f, 0.0f),
                conv_graph::propagate_config_t(4, 14)
            );
            populate_layer(*upper, ul_max_nbr_size);
            hg->set_layer_graph(l, std::move(upper));
        }

        // Step 3: inter-layer links: layer l (l>=1) maps each of its vertices
        // to a vertex in layer l-1. Use a deterministic dense mapping so we
        // can verify after compaction.
        for (layer_id_t l = 1; l < num_layers; ++l) {
            const vertex_num_t n_l = layer_sizes[l];
            std::vector<vertex_id_t> links(n_l);
            for (vertex_num_t i = 0; i < n_l; ++i) {
                // simple deterministic injection into parent layer
                links[i] = (i * 7u + l) % layer_sizes[l - 1];
            }
            hg->get_inter_layer_links().bottom_up_append(std::move(links));
        }

        // Step 4: entry point — pick a vertex in the top layer.
        const vertex_id_t top_layer_size = layer_sizes.back();
        hg->set_entry_point(top_layer_size > 0 ? (top_layer_size / 2) : 0);

        return hg;
    }

    std::unique_ptr<vector_array_t> base_vecs_;
};

TEST_F(HierarchicalDescentGraphCompactorTest, SingleLayerHierarchy) {
    auto hg = build_hierarchy({8000});
    ASSERT_EQ(hg->get_num_layers(), 1u);

    auto search_graph = descent_graph_compactor_t::from_hierarchical_graph(
        *hg, /*bl=*/bl_max_nbr_size, /*ul=*/ul_max_nbr_size);

    EXPECT_EQ(search_graph.get_num_layers(), 1u);
    EXPECT_EQ(search_graph.get_bl_extracted_nbr_size(), bl_max_nbr_size);
    EXPECT_EQ(search_graph.get_ul_extracted_nbr_size(), ul_max_nbr_size);
    EXPECT_EQ(search_graph.get_entry_point(), hg->get_entry_point());

    const auto& bottom = search_graph.get_layer_graph(0);
    EXPECT_EQ(bottom.get_num_vertices(), 8000u);
    EXPECT_EQ(bottom.get_extracted_nbr_size(), bl_max_nbr_size);
}

TEST_F(HierarchicalDescentGraphCompactorTest, MultiLayerStructurePreserved) {
    const std::vector<vertex_num_t> layer_sizes = {10000, 2000, 400};
    auto hg = build_hierarchy(layer_sizes);
    ASSERT_EQ(hg->get_num_layers(), layer_sizes.size());

    const vertex_num_t bl_extracted = bl_max_nbr_size;
    const vertex_num_t ul_extracted = ul_max_nbr_size;
    auto search_graph = descent_graph_compactor_t::from_hierarchical_graph(
        *hg, bl_extracted, ul_extracted);

    // Layer count + per-layer sizes
    ASSERT_EQ(search_graph.get_num_layers(), layer_sizes.size());
    for (layer_id_t l = 0; l < layer_sizes.size(); ++l) {
        const auto& layer = search_graph.get_layer_graph(l);
        EXPECT_EQ(layer.get_num_vertices(), layer_sizes[l]);
        const vertex_num_t expected_extracted =
            (l == 0) ? bl_extracted : ul_extracted;
        EXPECT_EQ(layer.get_extracted_nbr_size(), expected_extracted);
    }

    // Per-layer neighbor preservation
    for (layer_id_t l = 0; l < layer_sizes.size(); ++l) {
        const auto& src_layer = hg->get_layer_graph(l);
        const auto& dst_layer = search_graph.get_layer_graph(l);
        const vertex_num_t cap = (l == 0) ? bl_max_nbr_size : ul_max_nbr_size;
        const vertex_num_t extracted = dst_layer.get_extracted_nbr_size();

        const auto& src_nbrs_arr = src_layer.get_nbrs_arr();
        for (vertex_id_t v = 0; v < src_layer.get_num_vertices(); ++v) {
            const auto& src_nbrs = src_nbrs_arr[v];
            auto row = dst_layer.fetch_nbrs(v);
            const vertex_num_t copy_count = std::min(
                static_cast<vertex_num_t>(src_nbrs.size()), extracted
            );
            for (vertex_num_t i = 0; i < copy_count; ++i) {
                EXPECT_EQ(row[i], src_nbrs[i].get_id())
                    << "Layer " << l << " vertex " << v << " slot " << i;
            }
            for (vertex_num_t i = copy_count; i < extracted; ++i) {
                EXPECT_EQ(row[i], base_traits_t::invalid_vertex_id)
                    << "Layer " << l << " vertex " << v << " sentinel slot " << i;
            }
            (void)cap;
        }
    }

    // Entry point preserved.
    EXPECT_EQ(search_graph.get_entry_point(), hg->get_entry_point());
}

TEST_F(HierarchicalDescentGraphCompactorTest, InterLayerLinksPreserved) {
    const std::vector<vertex_num_t> layer_sizes = {6000, 1500, 300, 60};
    auto hg = build_hierarchy(layer_sizes);

    auto search_graph = descent_graph_compactor_t::from_hierarchical_graph(
        *hg, bl_max_nbr_size, ul_max_nbr_size);

    const auto& src_links = hg->get_inter_layer_links();
    const auto& dst_links = search_graph.get_inter_layer_links();
    for (layer_id_t l = 1; l < layer_sizes.size(); ++l) {
        auto src_span = src_links.get_layer_links(l);
        auto dst_span = dst_links.get_layer_links(l);
        ASSERT_EQ(src_span.size(), dst_span.size())
            << "Layer " << l << " inter-layer link size mismatch";
        for (size_t i = 0; i < src_span.size(); ++i) {
            EXPECT_EQ(src_span[i], dst_span[i])
                << "Layer " << l << " link " << i;
        }
    }
}

TEST_F(HierarchicalDescentGraphCompactorTest, ExtractedSmallerThanMax) {
    const std::vector<vertex_num_t> layer_sizes = {5000, 800};
    auto hg = build_hierarchy(layer_sizes);

    const vertex_num_t bl_extracted = 16;
    const vertex_num_t ul_extracted = 8;
    auto search_graph = descent_graph_compactor_t::from_hierarchical_graph(
        *hg, bl_extracted, ul_extracted);

    EXPECT_EQ(search_graph.get_bl_extracted_nbr_size(), bl_extracted);
    EXPECT_EQ(search_graph.get_ul_extracted_nbr_size(), ul_extracted);
    EXPECT_EQ(search_graph.get_layer_graph(0).get_extracted_nbr_size(), bl_extracted);
    EXPECT_EQ(search_graph.get_layer_graph(1).get_extracted_nbr_size(), ul_extracted);

    // Verify that downsampled layers preserve the prefix correctly.
    for (layer_id_t l = 0; l < layer_sizes.size(); ++l) {
        const auto& src_layer = hg->get_layer_graph(l);
        const auto& dst_layer = search_graph.get_layer_graph(l);
        const vertex_num_t extracted =
            (l == 0) ? bl_extracted : ul_extracted;

        const auto& src_nbrs_arr = src_layer.get_nbrs_arr();
        for (vertex_id_t v = 0; v < src_layer.get_num_vertices(); ++v) {
            const auto& src_nbrs = src_nbrs_arr[v];
            auto row = dst_layer.fetch_nbrs(v);
            const vertex_num_t copy_count = std::min(
                static_cast<vertex_num_t>(src_nbrs.size()), extracted
            );
            for (vertex_num_t i = 0; i < copy_count; ++i) {
                EXPECT_EQ(row[i], src_nbrs[i].get_id());
            }
            for (vertex_num_t i = copy_count; i < extracted; ++i) {
                EXPECT_EQ(row[i], base_traits_t::invalid_vertex_id);
            }
        }
    }
}

TEST_F(HierarchicalDescentGraphCompactorTest, ExtractedLargerThanMaxThrows) {
    auto hg = build_hierarchy({4000, 600});
    EXPECT_THROW({
        descent_graph_compactor_t::from_hierarchical_graph(
            *hg, bl_max_nbr_size + 1, ul_max_nbr_size);
    }, std::runtime_error);
    EXPECT_THROW({
        descent_graph_compactor_t::from_hierarchical_graph(
            *hg, bl_max_nbr_size, ul_max_nbr_size + 1);
    }, std::runtime_error);
}

TEST_F(HierarchicalDescentGraphCompactorTest, LargeMultiLayer) {
    // ~50k bottom + tapered upper layers, exercises both parallel
    // bottom-layer compaction and per-layer wiring.
    const std::vector<vertex_num_t> layer_sizes = {50'000, 12'000, 3'000, 600, 100};
    auto hg = build_hierarchy(layer_sizes);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto search_graph = descent_graph_compactor_t::from_hierarchical_graph(
        *hg, bl_max_nbr_size, ul_max_nbr_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    ARTEA_INFO(fmt::format(
        "from_hierarchical_graph: {} layers, bottom={} took {} ms",
        layer_sizes.size(), layer_sizes[0], ms));

    ASSERT_EQ(search_graph.get_num_layers(), layer_sizes.size());
    for (layer_id_t l = 0; l < layer_sizes.size(); ++l) {
        EXPECT_EQ(search_graph.get_layer_graph(l).get_num_vertices(), layer_sizes[l]);
    }
    EXPECT_EQ(search_graph.get_entry_point(), hg->get_entry_point());

    // Verify inter-layer links of every layer (source must remain intact).
    const auto& src_links = hg->get_inter_layer_links();
    const auto& dst_links = search_graph.get_inter_layer_links();
    for (layer_id_t l = 1; l < layer_sizes.size(); ++l) {
        auto src_span = src_links.get_layer_links(l);
        auto dst_span = dst_links.get_layer_links(l);
        ASSERT_EQ(src_span.size(), dst_span.size());
        for (size_t i = 0; i < src_span.size(); ++i) {
            ASSERT_EQ(src_span[i], dst_span[i]);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
