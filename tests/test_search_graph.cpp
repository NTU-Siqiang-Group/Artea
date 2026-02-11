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
 * @FilePath: /Artea/tests/test_search_graph.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Test for SearchGraph conversion from FlatGraph
 */

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <gtest/gtest.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>

using namespace artea;
using namespace artea::cpu;

// Type definitions using SIMPLE_EUCLIDEAN for low-dimensional vectors
using vec_num_t = uint32_t;
using vec_ele_t = float;
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::SIMPLE_EUCLIDEAN>;
using index_traits_t = IndexTraits<base_traits_t>;

using dist_func_t = typename computer_traits_t::dist_func_t;
using vector_array_t = typename computer_traits_t::vector_array_t;
using vertex_id_t = typename base_traits_t::vertex_id_t;
using distance_t = typename base_traits_t::distance_t;
using nbr_t = typename base_traits_t::nbr_t;
using nbr_arr_t = typename base_traits_t::nbr_arr_t;
using flat_graph_t = typename index_traits_t::flat_graph_t;
using search_graph_t = typename index_traits_t::search_graph_t;

class SearchGraphCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 2D graph with 10 vertices
        // Arranged in a 2x5 grid pattern for testing
        num_vertices_ = 10;
        vec_dim_ = 2;
        reserved_nbr_size_ = 8;

        // Create VectorArray and populate with vectors
        vecs_ = std::make_unique<vector_array_t>(vec_dim_);
        vecs_->reserve(num_vertices_);

        // Add vertices in a 2x5 grid
        for (int row = 0; row < 2; ++row) {
            for (int col = 0; col < 5; ++col) {
                std::vector<vec_ele_t> v = {
                    static_cast<vec_ele_t>(col),
                    static_cast<vec_ele_t>(row)
                };
                vecs_->append_vec(v.data());
            }
        }

        dist_func_ = std::make_unique<dist_func_t>(vec_dim_);

        // Initialize flat graph
        flat_graph_ = std::make_unique<flat_graph_t>(
            *vecs_,
            num_vertices_,
            reserved_nbr_size_,
            reserved_nbr_size_ * 2
        );
    }

    // Helper function to compute distance
    distance_t compute_distance(vertex_id_t v1, vertex_id_t v2) {
        const vec_ele_t* vec1 = vecs_->get(v1);
        const vec_ele_t* vec2 = vecs_->get(v2);
        return (*dist_func_)(vec1, vec2);
    }

    // Helper to populate flat graph with random neighbors
    void populate_random_neighbors(vec_num_t max_neighbors_per_vertex) {
        std::mt19937 rng(42);  // Fixed seed for reproducibility
        std::uniform_int_distribution<vertex_id_t> dist(0, num_vertices_ - 1);

        auto& nbrs_arr = flat_graph_->get_nbrs_arr();

        for (vertex_id_t u = 0; u < num_vertices_; ++u) {
            nbr_arr_t& nbrs = nbrs_arr[u];
            nbrs.clear();

            // Generate random number of neighbors (1 to max_neighbors_per_vertex)
            vec_num_t num_nbrs = 1 + (rng() % max_neighbors_per_vertex);

            for (vec_num_t i = 0; i < num_nbrs; ++i) {
                vertex_id_t v = dist(rng);
                // Avoid self-loops
                if (v == u) {
                    v = (v + 1) % num_vertices_;
                }

                distance_t d = compute_distance(u, v);
                nbrs.push_back(nbr_t(v, d, true));
            }

            // Sort by distance
            std::sort(nbrs.begin(), nbrs.end(),
                [](const nbr_t& a, const nbr_t& b) {
                    return a.get_distance() < b.get_distance();
                });
        }
    }

    vec_num_t num_vertices_;
    vec_num_t vec_dim_;
    vec_num_t reserved_nbr_size_;
    std::unique_ptr<vector_array_t> vecs_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<flat_graph_t> flat_graph_;
};

TEST_F(SearchGraphCorrectnessTest, BasicConversion) {
    // Populate flat graph with random neighbors
    const vec_num_t max_neighbors = 6;
    populate_random_neighbors(max_neighbors);

    const auto& flat_nbrs_arr = flat_graph_->get_nbrs_arr();

    logger.info("FlatGraph before conversion:");
    for (vertex_id_t u = 0; u < std::min(num_vertices_, static_cast<vec_num_t>(5)); ++u) {
        const auto& nbrs = flat_nbrs_arr[u];
        std::string nbr_list;
        for (size_t i = 0; i < nbrs.size(); ++i) {
            nbr_list += fmt::format("({}, {:.3f})", nbrs[i].get_id(), nbrs[i].get_distance());
            if (i < nbrs.size() - 1) nbr_list += ", ";
        }
        logger.info(fmt::format("  v{} ({} nbrs) -> [{}]", u, nbrs.size(), nbr_list));
    }

    // Convert to search graph with fix_nbr_size = 4
    const vec_num_t fix_nbr_size = 4;
    auto search_graph = search_graph_t::from_flat_graph(*flat_graph_, fix_nbr_size);

    logger.info(fmt::format("\nSearchGraph after conversion (fix_nbr_size={}):", fix_nbr_size));
    for (vertex_id_t u = 0; u < std::min(num_vertices_, static_cast<vec_num_t>(5)); ++u) {
        auto nbrs_span = search_graph.fetch_nbrs(u);
        std::string nbr_list;
        for (size_t i = 0; i < nbrs_span.size(); ++i) {
            vertex_id_t nbr_id = nbrs_span[i];
            if (nbr_id != base_traits_t::invalid_vertex_id) {
                nbr_list += fmt::format("{}", nbr_id);
            } else {
                nbr_list += "INV";
            }
            if (i < nbrs_span.size() - 1) nbr_list += ", ";
        }
        logger.info(fmt::format("  v{} -> [{}]", u, nbr_list));
    }

    // Verify basic properties
    EXPECT_EQ(search_graph.get_num_vertices(), num_vertices_)
        << "Number of vertices should match";
    EXPECT_EQ(search_graph.get_fix_nbr_size(), fix_nbr_size)
        << "Fixed neighbor size should match";

    // Verify neighbor conversion correctness
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const auto& flat_nbrs = flat_nbrs_arr[u];
        auto search_nbrs = search_graph.fetch_nbrs(u);

        EXPECT_EQ(search_nbrs.size(), fix_nbr_size)
            << fmt::format("Vertex {} should have exactly {} neighbors in search graph", u, fix_nbr_size);

        // Check that the first min(flat_nbrs.size(), fix_nbr_size) neighbors match
        const vec_num_t copy_count = std::min(
            static_cast<vec_num_t>(flat_nbrs.size()),
            fix_nbr_size
        );

        for (vec_num_t i = 0; i < copy_count; ++i) {
            EXPECT_EQ(search_nbrs[i], flat_nbrs[i].get_id())
                << fmt::format("Vertex {} neighbor {} should match: expected {}, got {}",
                              u, i, flat_nbrs[i].get_id(), search_nbrs[i]);
        }

        // Check that remaining slots are filled with invalid_vertex_id
        for (vec_num_t i = copy_count; i < fix_nbr_size; ++i) {
            EXPECT_EQ(search_nbrs[i], base_traits_t::invalid_vertex_id)
                << fmt::format("Vertex {} neighbor {} should be invalid (got {})",
                              u, i, search_nbrs[i]);
        }
    }
}

TEST_F(SearchGraphCorrectnessTest, ConversionWithFewerNeighbors) {
    // Test case where flat graph has fewer neighbors than fix_nbr_size
    auto& nbrs_arr = flat_graph_->get_nbrs_arr();

    // Manually create a graph where each vertex has only 2 neighbors
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        nbr_arr_t& nbrs = nbrs_arr[u];
        nbrs.clear();

        // Add 2 neighbors
        vertex_id_t v1 = (u + 1) % num_vertices_;
        vertex_id_t v2 = (u + 2) % num_vertices_;

        distance_t d1 = compute_distance(u, v1);
        distance_t d2 = compute_distance(u, v2);

        nbrs.push_back(nbr_t(v1, d1, true));
        nbrs.push_back(nbr_t(v2, d2, true));
    }

    logger.info("FlatGraph with 2 neighbors per vertex:");
    for (vertex_id_t u = 0; u < std::min(num_vertices_, static_cast<vec_num_t>(5)); ++u) {
        const auto& nbrs = nbrs_arr[u];
        logger.info(fmt::format("  v{} -> [{}, {}]", u, nbrs[0].get_id(), nbrs[1].get_id()));
    }

    // Convert with fix_nbr_size = 5 (more than available neighbors)
    const vec_num_t fix_nbr_size = 5;
    auto search_graph = search_graph_t::from_flat_graph(*flat_graph_, fix_nbr_size);

    logger.info(fmt::format("\nSearchGraph with fix_nbr_size={} (more than available):", fix_nbr_size));

    // Verify conversion
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const auto& flat_nbrs = nbrs_arr[u];
        auto search_nbrs = search_graph.fetch_nbrs(u);

        // First 2 neighbors should match
        EXPECT_EQ(search_nbrs[0], flat_nbrs[0].get_id());
        EXPECT_EQ(search_nbrs[1], flat_nbrs[1].get_id());

        // Remaining 3 slots should be invalid
        for (vec_num_t i = 2; i < fix_nbr_size; ++i) {
            EXPECT_EQ(search_nbrs[i], base_traits_t::invalid_vertex_id)
                << fmt::format("Vertex {} neighbor {} should be invalid", u, i);
        }
    }

    logger.info("All vertices correctly padded with invalid IDs");
}

TEST_F(SearchGraphCorrectnessTest, ConversionWithMoreNeighbors) {
    // Test case where flat graph has more neighbors than fix_nbr_size
    auto& nbrs_arr = flat_graph_->get_nbrs_arr();

    // Create a graph where each vertex has many neighbors
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        nbr_arr_t& nbrs = nbrs_arr[u];
        nbrs.clear();

        // Add all other vertices as neighbors
        for (vertex_id_t v = 0; v < num_vertices_; ++v) {
            if (v != u) {
                distance_t d = compute_distance(u, v);
                nbrs.push_back(nbr_t(v, d, true));
            }
        }

        // Sort by distance
        std::sort(nbrs.begin(), nbrs.end(),
            [](const nbr_t& a, const nbr_t& b) {
                return a.get_distance() < b.get_distance();
            });
    }

    logger.info(fmt::format("FlatGraph with {} neighbors per vertex (complete graph):", num_vertices_ - 1));

    // Convert with fix_nbr_size = 3 (less than available neighbors)
    const vec_num_t fix_nbr_size = 3;
    auto search_graph = search_graph_t::from_flat_graph(*flat_graph_, fix_nbr_size);

    logger.info(fmt::format("\nSearchGraph with fix_nbr_size={} (less than available):", fix_nbr_size));

    // Verify conversion - only first fix_nbr_size neighbors should be copied
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const auto& flat_nbrs = nbrs_arr[u];
        auto search_nbrs = search_graph.fetch_nbrs(u);

        EXPECT_EQ(search_nbrs.size(), fix_nbr_size);

        // All fix_nbr_size neighbors should match the first fix_nbr_size from flat graph
        for (vec_num_t i = 0; i < fix_nbr_size; ++i) {
            EXPECT_EQ(search_nbrs[i], flat_nbrs[i].get_id())
                << fmt::format("Vertex {} neighbor {} mismatch", u, i);
        }

        // No invalid IDs should be present (all slots filled)
        for (vec_num_t i = 0; i < fix_nbr_size; ++i) {
            EXPECT_NE(search_nbrs[i], base_traits_t::invalid_vertex_id)
                << fmt::format("Vertex {} neighbor {} should not be invalid", u, i);
        }
    }

    logger.info("All vertices correctly truncated to fix_nbr_size");
}

TEST_F(SearchGraphCorrectnessTest, EmptyFlatGraph) {
    // Test conversion of an empty flat graph (no neighbors)
    auto& nbrs_arr = flat_graph_->get_nbrs_arr();

    // Clear all neighbors
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        nbrs_arr[u].clear();
    }

    logger.info("FlatGraph with no neighbors (empty graph)");

    const vec_num_t fix_nbr_size = 4;
    auto search_graph = search_graph_t::from_flat_graph(*flat_graph_, fix_nbr_size);

    logger.info(fmt::format("SearchGraph with fix_nbr_size={} from empty graph:", fix_nbr_size));

    // Verify all neighbors are invalid
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        auto search_nbrs = search_graph.fetch_nbrs(u);

        for (vec_num_t i = 0; i < fix_nbr_size; ++i) {
            EXPECT_EQ(search_nbrs[i], base_traits_t::invalid_vertex_id)
                << fmt::format("Empty graph: vertex {} neighbor {} should be invalid", u, i);
        }
    }

    logger.info("All neighbors correctly set to invalid for empty graph");
}

TEST_F(SearchGraphCorrectnessTest, VectorDataReference) {
    // Test that search graph correctly references the same vector data
    populate_random_neighbors(5);

    const vec_num_t fix_nbr_size = 4;
    auto search_graph = search_graph_t::from_flat_graph(*flat_graph_, fix_nbr_size);

    // Verify that both graphs reference the same vector data
    const auto& flat_vecs = flat_graph_->get_vecs_data();
    const auto& search_vecs = search_graph.get_vecs_data();

    EXPECT_EQ(&flat_vecs, &search_vecs)
        << "SearchGraph should reference the same vector data as FlatGraph";

    // Verify vector data is accessible and correct
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const vec_ele_t* flat_vec = flat_vecs.get(u);
        const vec_ele_t* search_vec = search_vecs.get(u);

        EXPECT_EQ(flat_vec, search_vec)
            << fmt::format("Vector pointers for vertex {} should be identical", u);

        // Verify vector values match
        for (vec_num_t d = 0; d < vec_dim_; ++d) {
            EXPECT_EQ(flat_vec[d], search_vec[d])
                << fmt::format("Vector values for vertex {} dimension {} should match", u, d);
        }
    }

    logger.info("Vector data correctly shared between FlatGraph and SearchGraph");
}

TEST_F(SearchGraphCorrectnessTest, GetNeighborsPointer) {
    // Test the get_neighbors() pointer interface
    populate_random_neighbors(6);

    const vec_num_t fix_nbr_size = 4;
    auto search_graph = search_graph_t::from_flat_graph(*flat_graph_, fix_nbr_size);

    const auto& flat_nbrs_arr = flat_graph_->get_nbrs_arr();

    // Test both const and non-const versions
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const vertex_id_t* nbrs_ptr = search_graph.get_neighbors(u);
        const auto& flat_nbrs = flat_nbrs_arr[u];

        const vec_num_t copy_count = std::min(
            static_cast<vec_num_t>(flat_nbrs.size()),
            fix_nbr_size
        );

        // Verify pointer access matches span access
        auto nbrs_span = search_graph.fetch_nbrs(u);
        for (vec_num_t i = 0; i < fix_nbr_size; ++i) {
            EXPECT_EQ(nbrs_ptr[i], nbrs_span[i])
                << fmt::format("Pointer and span access should match for vertex {} neighbor {}", u, i);
        }

        // Verify correctness
        for (vec_num_t i = 0; i < copy_count; ++i) {
            EXPECT_EQ(nbrs_ptr[i], flat_nbrs[i].get_id());
        }

        for (vec_num_t i = copy_count; i < fix_nbr_size; ++i) {
            EXPECT_EQ(nbrs_ptr[i], base_traits_t::invalid_vertex_id);
        }
    }

    logger.info("get_neighbors() pointer interface works correctly");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
