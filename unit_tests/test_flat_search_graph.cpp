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
 * @FilePath: /Artea/tests/test_flat_search_graph.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Test for FlatSearchGraph and FlatSearchGraphFactory with synthetic data
 */

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <chrono>
#include <gtest/gtest.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;
using namespace artea::cpu::default_context;

class FlatSearchGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create synthetic large-scale graph
        num_vertices_ = 10000;
        vec_dim_ = 128;
        layer_config_ = layer_config_t(64, 128);
        edges_builder_config_ = edges_builder_config_t(1.0, 0.0, 4, 14);

        // Create VectorArray and populate with random vectors
        vecs_ = std::make_unique<vector_array_t>(vec_dim_);
        vecs_->reserve(num_vertices_);

        std::mt19937 rng(42);
        std::uniform_real_distribution<vec_ele_t> dist(0.0f, 1.0f);

        for (vertex_num_t i = 0; i < num_vertices_; ++i) {
            std::vector<vec_ele_t> v(vec_dim_);
            for (vec_dim_t d = 0; d < vec_dim_; ++d) {
                v[d] = dist(rng);
            }
            vecs_->append_vec(v.data());
        }

        dist_func_ = std::make_unique<dist_func_t>(vec_dim_);

        // Initialize flat graph
        flat_graph_ = std::make_unique<flat_graph_t>(
            *vecs_,
            layer_config_,
            edges_builder_config_
        );
    }

    // Helper function to compute distance
    distance_t compute_distance(vertex_id_t v1, vertex_id_t v2) {
        const vec_ele_t* vec1 = vecs_->get(v1);
        const vec_ele_t* vec2 = vecs_->get(v2);
        return (*dist_func_)(vec1, vec2);
    }

    // Helper to populate flat graph with random neighbors
    void populate_random_neighbors(vertex_num_t max_neighbors_per_vertex) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<vertex_id_t> dist(0, num_vertices_ - 1);

        auto& nbrs_arr = flat_graph_->get_nbrs_arr();

        for (vertex_id_t u = 0; u < num_vertices_; ++u) {
            nbr_arr_t& nbrs = nbrs_arr[u];
            nbrs.clear();

            vertex_num_t num_nbrs = 1 + (rng() % max_neighbors_per_vertex);

            for (vertex_num_t i = 0; i < num_nbrs; ++i) {
                vertex_id_t v = dist(rng);
                if (v == u) {
                    v = (v + 1) % num_vertices_;
                }

                distance_t d = compute_distance(u, v);
                nbrs.push_back(nbr_t(v, d, true));
            }

            std::sort(nbrs.begin(), nbrs.end(),
                [](const nbr_t& a, const nbr_t& b) {
                    return a.get_distance() < b.get_distance();
                });
        }
    }

    vertex_num_t num_vertices_;
    vec_dim_t vec_dim_;
    layer_config_t layer_config_{64, 128};
    edges_builder_config_t edges_builder_config_{1.0, 0.0, 4, 14};
    std::unique_ptr<vector_array_t> vecs_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<flat_graph_t> flat_graph_;
};

TEST_F(FlatSearchGraphTest, BasicConversion) {
    const vertex_num_t max_neighbors = 50;
    populate_random_neighbors(max_neighbors);

    const auto& flat_nbrs_arr = flat_graph_->get_nbrs_arr();

    const vertex_num_t extracted_nbr_size = 32;
    auto flat_search_graph = search_graph_converter_t::from_flat_graph(*flat_graph_, extracted_nbr_size);

    EXPECT_EQ(flat_search_graph.get_num_vertices(), num_vertices_);
    EXPECT_EQ(flat_search_graph.get_extracted_nbr_size(), extracted_nbr_size);

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const auto& flat_nbrs = flat_nbrs_arr[u];
        auto search_nbrs = flat_search_graph.fetch_nbrs(u);

        EXPECT_EQ(search_nbrs.size(), extracted_nbr_size);

        const vertex_num_t copy_count = std::min(
            static_cast<vertex_num_t>(flat_nbrs.size()),
            extracted_nbr_size
        );

        for (vertex_num_t i = 0; i < copy_count; ++i) {
            EXPECT_EQ(search_nbrs[i], flat_nbrs[i].get_id());
        }

        for (vertex_num_t i = copy_count; i < extracted_nbr_size; ++i) {
            EXPECT_EQ(search_nbrs[i], base_traits_t::invalid_vertex_id);
        }
    }
}

TEST_F(FlatSearchGraphTest, EmptyFlatGraph) {
    const vertex_num_t extracted_nbr_size = 32;
    auto flat_search_graph = search_graph_converter_t::from_flat_graph(*flat_graph_, extracted_nbr_size);

    EXPECT_EQ(flat_search_graph.get_num_vertices(), num_vertices_);
    EXPECT_EQ(flat_search_graph.get_extracted_nbr_size(), extracted_nbr_size);

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        auto search_nbrs = flat_search_graph.fetch_nbrs(u);
        EXPECT_EQ(search_nbrs.size(), extracted_nbr_size);

        for (vertex_num_t i = 0; i < extracted_nbr_size; ++i) {
            EXPECT_EQ(search_nbrs[i], base_traits_t::invalid_vertex_id);
        }
    }
}

TEST_F(FlatSearchGraphTest, FixNbrSizeLargerThanFlatNbrs) {
    const vertex_num_t max_neighbors = 10;
    populate_random_neighbors(max_neighbors);

    const auto& flat_nbrs_arr = flat_graph_->get_nbrs_arr();
    const vertex_num_t extracted_nbr_size = 64;
    auto flat_search_graph = search_graph_converter_t::from_flat_graph(*flat_graph_, extracted_nbr_size);

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const auto& flat_nbrs = flat_nbrs_arr[u];
        auto search_nbrs = flat_search_graph.fetch_nbrs(u);

        EXPECT_EQ(search_nbrs.size(), extracted_nbr_size);

        for (size_t i = 0; i < flat_nbrs.size(); ++i) {
            EXPECT_EQ(search_nbrs[i], flat_nbrs[i].get_id());
        }

        for (vertex_num_t i = flat_nbrs.size(); i < extracted_nbr_size; ++i) {
            EXPECT_EQ(search_nbrs[i], base_traits_t::invalid_vertex_id);
        }
    }
}

TEST_F(FlatSearchGraphTest, FixNbrSizeSmallerThanFlatNbrs) {
    const vertex_num_t max_neighbors = 50;
    populate_random_neighbors(max_neighbors);

    const auto& flat_nbrs_arr = flat_graph_->get_nbrs_arr();
    const vertex_num_t extracted_nbr_size = 16;
    auto flat_search_graph = search_graph_converter_t::from_flat_graph(*flat_graph_, extracted_nbr_size);

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const auto& flat_nbrs = flat_nbrs_arr[u];
        auto search_nbrs = flat_search_graph.fetch_nbrs(u);

        EXPECT_EQ(search_nbrs.size(), extracted_nbr_size);

        // When extracted_nbr_size < flat_nbrs.size(), only the first extracted_nbr_size neighbors are copied
        const vertex_num_t copy_count = std::min(
            static_cast<vertex_num_t>(flat_nbrs.size()),
            extracted_nbr_size
        );

        for (vertex_num_t i = 0; i < copy_count; ++i) {
            EXPECT_EQ(search_nbrs[i], flat_nbrs[i].get_id());
        }
    }
}

TEST_F(FlatSearchGraphTest, SingleNeighborPerVertex) {
    const vertex_num_t max_neighbors = 1;
    populate_random_neighbors(max_neighbors);

    const auto& flat_nbrs_arr = flat_graph_->get_nbrs_arr();
    const vertex_num_t extracted_nbr_size = 32;
    auto flat_search_graph = search_graph_converter_t::from_flat_graph(*flat_graph_, extracted_nbr_size);

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const auto& flat_nbrs = flat_nbrs_arr[u];
        auto search_nbrs = flat_search_graph.fetch_nbrs(u);

        EXPECT_EQ(flat_nbrs.size(), 1);
        EXPECT_EQ(search_nbrs[0], flat_nbrs[0].get_id());

        for (vertex_num_t i = 1; i < extracted_nbr_size; ++i) {
            EXPECT_EQ(search_nbrs[i], base_traits_t::invalid_vertex_id);
        }
    }
}

TEST_F(FlatSearchGraphTest, NeighborOrderPreservation) {
    const vertex_num_t max_neighbors = 40;
    populate_random_neighbors(max_neighbors);

    const auto& flat_nbrs_arr = flat_graph_->get_nbrs_arr();
    const vertex_num_t extracted_nbr_size = 32;
    auto flat_search_graph = search_graph_converter_t::from_flat_graph(*flat_graph_, extracted_nbr_size);

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const auto& flat_nbrs = flat_nbrs_arr[u];
        auto search_nbrs = flat_search_graph.fetch_nbrs(u);

        const vertex_num_t copy_count = std::min(
            static_cast<vertex_num_t>(flat_nbrs.size()),
            extracted_nbr_size
        );

        for (vertex_num_t i = 0; i < copy_count; ++i) {
            EXPECT_EQ(search_nbrs[i], flat_nbrs[i].get_id());
        }
    }
}

TEST_F(FlatSearchGraphTest, LargeFixNbrSize) {
    const vertex_num_t max_neighbors = 50;
    populate_random_neighbors(max_neighbors);

    const vertex_num_t extracted_nbr_size = 128;

    // extracted_nbr_size (128) exceeds max_nbr_size (64), should throw an exception
    EXPECT_THROW({
        auto flat_search_graph = search_graph_converter_t::from_flat_graph(*flat_graph_, extracted_nbr_size);
    }, std::runtime_error);
}

TEST_F(FlatSearchGraphTest, SmallFixNbrSize) {
    const vertex_num_t max_neighbors = 50;
    populate_random_neighbors(max_neighbors);

    const vertex_num_t extracted_nbr_size = 4;
    auto flat_search_graph = search_graph_converter_t::from_flat_graph(*flat_graph_, extracted_nbr_size);

    EXPECT_EQ(flat_search_graph.get_num_vertices(), num_vertices_);
    EXPECT_EQ(flat_search_graph.get_extracted_nbr_size(), extracted_nbr_size);
}

TEST_F(FlatSearchGraphTest, PerformanceTest) {
    const vertex_num_t max_neighbors = 64;
    populate_random_neighbors(max_neighbors);

    const vertex_num_t extracted_nbr_size = 32;

    auto start = std::chrono::high_resolution_clock::now();
    auto flat_search_graph = search_graph_converter_t::from_flat_graph(*flat_graph_, extracted_nbr_size);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    logger.info(fmt::format("Conversion of {} vertices with extracted_nbr_size={} took {} ms",
        num_vertices_, extracted_nbr_size, duration.count()));

    EXPECT_EQ(flat_search_graph.get_num_vertices(), num_vertices_);
    EXPECT_EQ(flat_search_graph.get_extracted_nbr_size(), extracted_nbr_size);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}