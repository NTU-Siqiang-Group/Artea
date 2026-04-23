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
 * @FilePath: /Artea/unit_tests/test_refining_graphs.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Tests for dynamic::RefiningGraph, compact::RefiningGraph,
 *               and RefiningGraphCompactor.
 */

#include <algorithm>
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
//  Shared fixture: builds a conv_graph::index_t with random vectors
//  and populates its neighbor arrays with random neighbors.
// ============================================================

class RefiningGraphTest : public ::testing::Test {
protected:
    static constexpr vertex_num_t num_vertices = 10'000;
    static constexpr vec_dim_t    vec_dim      = 128;

    void SetUp() override {
        vecs_ = std::make_unique<vector_array_t>(vec_dim);
        vecs_->reserve(num_vertices);

        std::mt19937 rng(42);
        std::uniform_real_distribution<vec_ele_t> dist(0.0f, 1.0f);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            std::vector<vec_ele_t> v(vec_dim);
            for (vec_dim_t d = 0; d < vec_dim; ++d) v[d] = dist(rng);
            vecs_->append_vec(v.data());
        }

        dist_func_ = std::make_unique<dist_func_t>(vec_dim);

        layer_config_ = layer_config_t(64);
        graph_index_ = std::make_unique<conv_graph::index_t>(
            *vecs_, layer_config_,
            conv_graph::pruning_config_t(1.0, 0.0),
            conv_graph::propagate_config_t(4, 14));
    }

    void populate_random_neighbors(const vertex_num_t max_nbrs_per_vertex) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<vertex_id_t> id_dist(0, num_vertices - 1);
        auto& nbrs_arr = graph_index_->get_nbrs_arr();

        for (vertex_id_t u = 0; u < num_vertices; ++u) {
            auto& nbrs = nbrs_arr[u];
            nbrs.clear();
            const vertex_num_t count = 1 + (rng() % max_nbrs_per_vertex);
            for (vertex_num_t i = 0; i < count; ++i) {
                vertex_id_t v = id_dist(rng);
                if (v == u) v = (v + 1) % num_vertices;
                const distance_t d = (*dist_func_)(vecs_->get(u), vecs_->get(v));
                nbrs.push_back(nbr_t(v, d, true));
            }
            std::sort(nbrs.begin(), nbrs.end(),
                [](const nbr_t& a, const nbr_t& b) {
                    return a.get_distance() < b.get_distance();
                });
        }
    }

    layer_config_t layer_config_{64};
    std::unique_ptr<vector_array_t> vecs_;
    std::unique_ptr<dist_func_t>    dist_func_;
    std::unique_ptr<conv_graph::index_t> graph_index_;
};

// ============================================================
//  dynamic::RefiningGraph (via conv_graph::index_t wrapper)
// ============================================================

TEST_F(RefiningGraphTest, Construction) {
    EXPECT_EQ(graph_index_->get_num_vertices(), num_vertices);
    EXPECT_EQ(graph_index_->layer_config().max_nbr_size(), 64u);
}

TEST_F(RefiningGraphTest, InitialNeighborsEmpty) {
    const auto& nbrs_arr = graph_index_->get_nbrs_arr();
    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        EXPECT_TRUE(nbrs_arr[v].empty());
    }
}

TEST_F(RefiningGraphTest, PopulateAndReadNeighbors) {
    populate_random_neighbors(50);
    const auto& nbrs_arr = graph_index_->get_nbrs_arr();

    uint64_t total_edges = 0;
    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        EXPECT_GE(nbrs_arr[v].size(), 1u);
        EXPECT_LE(nbrs_arr[v].size(), 50u);
        total_edges += nbrs_arr[v].size();
    }
    EXPECT_GT(total_edges, 0u);
}

TEST_F(RefiningGraphTest, NeighborsSortedByDistance) {
    populate_random_neighbors(50);
    const auto& nbrs_arr = graph_index_->get_nbrs_arr();

    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        const auto& nbrs = nbrs_arr[v];
        for (size_t i = 1; i < nbrs.size(); ++i) {
            EXPECT_LE(nbrs[i - 1].get_distance(), nbrs[i].get_distance());
        }
    }
}

TEST_F(RefiningGraphTest, VecsDataAccessor) {
    EXPECT_EQ(graph_index_->get_vecs_data().get_num_vecs(), num_vertices);
    EXPECT_EQ(graph_index_->get_vecs_data().get_vec_dim(), vec_dim);
}

TEST_F(RefiningGraphTest, MoveSemantics) {
    populate_random_neighbors(30);
    const auto& nbrs_before = graph_index_->get_nbrs_arr();
    const size_t edges_v0 = nbrs_before[0].size();

    conv_graph::index_t moved = std::move(*graph_index_);
    EXPECT_EQ(moved.get_num_vertices(), num_vertices);
    EXPECT_EQ(moved.get_nbrs_arr()[0].size(), edges_v0);
}

// ============================================================
//  compact::RefiningGraph (via RefiningGraphCompactor)
// ============================================================

TEST_F(RefiningGraphTest, CompactorBasicConversion) {
    populate_random_neighbors(50);
    const vertex_num_t extracted = 32;

    auto compact = refining_graph_compactor_t::compact_graph(*graph_index_, extracted);
    EXPECT_EQ(compact.get_num_vertices(), num_vertices);
    EXPECT_EQ(compact.get_extracted_nbr_size(), extracted);

    const auto& src_nbrs_arr = graph_index_->get_nbrs_arr();
    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        const auto& src_nbrs = src_nbrs_arr[v];
        auto compact_nbrs = compact.fetch_nbrs(v);
        const vertex_num_t expected_count =
            std::min(static_cast<vertex_num_t>(src_nbrs.size()), extracted);

        for (vertex_num_t i = 0; i < expected_count; ++i) {
            EXPECT_EQ(compact_nbrs[i], src_nbrs[i].get_vid());
        }
        for (vertex_num_t i = expected_count; i < extracted; ++i) {
            EXPECT_EQ(compact_nbrs[i], base_traits_t::invalid_vertex_id);
        }
    }
}

TEST_F(RefiningGraphTest, CompactorEmptyGraph) {
    // Neighbors are empty (default after construction).
    const vertex_num_t extracted = 16;
    auto compact = refining_graph_compactor_t::compact_graph(*graph_index_, extracted);
    EXPECT_EQ(compact.get_num_vertices(), num_vertices);

    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        auto nbrs = compact.fetch_nbrs(v);
        for (vertex_num_t i = 0; i < extracted; ++i) {
            EXPECT_EQ(nbrs[i], base_traits_t::invalid_vertex_id);
        }
    }
}

TEST_F(RefiningGraphTest, CompactorExtractedSmallerThanActual) {
    populate_random_neighbors(50);
    const vertex_num_t extracted = 8;

    auto compact = refining_graph_compactor_t::compact_graph(*graph_index_, extracted);
    const auto& src_nbrs_arr = graph_index_->get_nbrs_arr();

    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        auto compact_nbrs = compact.fetch_nbrs(v);
        const auto& src_nbrs = src_nbrs_arr[v];
        const vertex_num_t copy_count =
            std::min(static_cast<vertex_num_t>(src_nbrs.size()), extracted);
        for (vertex_num_t i = 0; i < copy_count; ++i) {
            EXPECT_EQ(compact_nbrs[i], src_nbrs[i].get_vid());
        }
    }
}

TEST_F(RefiningGraphTest, CompactorNeighborOrderPreserved) {
    populate_random_neighbors(50);
    const vertex_num_t extracted = 32;

    auto compact = refining_graph_compactor_t::compact_graph(*graph_index_, extracted);

    for (vertex_id_t v = 0; v < std::min<vertex_id_t>(100, num_vertices); ++v) {
        auto nbrs = compact.fetch_nbrs(v);
        distance_t prev_dist = 0.0f;
        for (vertex_num_t i = 0; i < extracted; ++i) {
            if (nbrs[i] == base_traits_t::invalid_vertex_id) break;
            const distance_t d = (*dist_func_)(vecs_->get(v), vecs_->get(nbrs[i]));
            EXPECT_GE(d, prev_dist);
            prev_dist = d;
        }
    }
}

TEST_F(RefiningGraphTest, CompactorExtractedLargerThanMaxThrows) {
    populate_random_neighbors(10);
    EXPECT_THROW({
        refining_graph_compactor_t::compact_graph(
            *graph_index_,
            graph_index_->layer_config().max_nbr_size() + 1);
    }, std::runtime_error);
}

TEST_F(RefiningGraphTest, CompactReadWriteRoundTrip) {
    populate_random_neighbors(50);
    const vertex_num_t extracted = 32;

    auto compact = refining_graph_compactor_t::compact_graph(*graph_index_, extracted);

    // Verify get_neighbors (raw pointer) matches fetch_nbrs (span)
    for (vertex_id_t v = 0; v < std::min<vertex_id_t>(100, num_vertices); ++v) {
        const vertex_id_t* raw = compact.get_neighbors(v);
        auto span = compact.fetch_nbrs(v);
        for (vertex_num_t i = 0; i < extracted; ++i) {
            EXPECT_EQ(raw[i], span[i]);
        }
    }
}

TEST_F(RefiningGraphTest, CompactVecsDataAccessor) {
    const vertex_num_t extracted = 16;
    auto compact = refining_graph_compactor_t::compact_graph(*graph_index_, extracted);
    EXPECT_EQ(compact.get_vecs_data().get_num_vecs(), num_vertices);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
