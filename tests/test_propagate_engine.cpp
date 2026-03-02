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
 * @FilePath: /Artea/tests/test_propagate_engine.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Test for PropagateEngine with TriangleUpdater and ReverseUpdater
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <gtest/gtest.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/simple_tests_context.hpp>

// Type definitions using SIMPLE_EUCLIDEAN for low-dimensional vectors
using namespace artea;
using namespace artea::cpu::simple_tests_context;

class PropagateEngineCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 2D graph with 12 vertices
        // Arranged in a 3x4 grid pattern:
        //   v0: (0, 0)    v1: (1, 0)    v2: (2, 0)    v3: (3, 0)
        //   v4: (0, 1)    v5: (1, 1)    v6: (2, 1)    v7: (3, 1)
        //   v8: (0, 2)    v9: (1, 2)    v10: (2, 2)   v11: (3, 2)
        //
        // This creates a more realistic test case with various distance relationships

        num_vertices_ = 12;
        vec_dim_ = 2;
        max_nbr_size_ = 8;

        // Create VectorArray and populate with vectors
        vecs_ = std::make_unique<vector_array_t>(vec_dim_);
        vecs_->reserve(num_vertices_);

        // Add vertices in a 3x4 grid
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 4; ++col) {
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
            max_nbr_size_,
            max_nbr_size_ * 2
        );
    }

    // Helper function to compute distance
    distance_t compute_distance(vertex_id_t v1, vertex_id_t v2) {
        const vec_ele_t* vec1 = vecs_->get(v1);
        const vec_ele_t* vec2 = vecs_->get(v2);
        return (*dist_func_)(vec1, vec2);
    }

    // Helper to check if edge (u, v) exists in neighbor array
    bool has_edge(const nbr_arr_t& nbrs, vertex_id_t target_id, distance_t* out_dist = nullptr) {
        for (const auto& nbr : nbrs) {
            if (nbr.get_id() == target_id) {
                if (out_dist) *out_dist = nbr.get_distance();
                return true;
            }
        }
        return false;
    }

    // Verify RNG property manually
    bool verify_rng_property(
        const std::vector<nbr_arr_t>& nbrs_arr,
        ratio_t scale_coeffs = 1.0,
        ratio_t shifted_coeffs = 0.0
    ) {
        for (vertex_id_t u = 0; u < num_vertices_; ++u) {
            const auto& nbrs = nbrs_arr[u];

            for (size_t i = 0; i < nbrs.size(); ++i) {
                vertex_id_t v = nbrs[i].get_id();
                distance_t d_uv = nbrs[i].get_distance();
                distance_t threshold = (d_uv / scale_coeffs) - shifted_coeffs;

                // Check all other vertices
                for (vertex_id_t w = 0; w < num_vertices_; ++w) {
                    if (w == u || w == v) continue;

                    distance_t d_uw = compute_distance(u, w);
                    distance_t d_vw = compute_distance(v, w);

                    // If w violates RNG property for edge (u, v)
                    if (d_uw < threshold && d_vw < threshold) {
                        logger.error(fmt::format(
                            "RNG violation: edge ({}, {}) with d={:.3f}, "
                            "vertex {} creates shortcut with d({}, {})={:.3f}, d({}, {})={:.3f}, threshold={:.3f}",
                            u, v, d_uv, w, u, w, d_uw, v, w, d_vw, threshold
                        ));
                        return false;
                    }
                }
            }
        }
        return true;
    }

    vec_num_t num_vertices_;
    vec_num_t vec_dim_;
    vec_num_t max_nbr_size_;
    std::unique_ptr<vector_array_t> vecs_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<flat_graph_t> flat_graph_;
};

TEST_F(PropagateEngineCorrectnessTest, TrianglePruningWithPropagateEngine) {
    // Create initial complete graph
    auto& nbrs_arr = flat_graph_->get_nbrs_arr();

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        nbr_arr_t& nbrs = nbrs_arr[u];
        for (vertex_id_t v = 0; v < num_vertices_; ++v) {
            if (v != u) {
                distance_t dist = compute_distance(u, v);
                nbrs.push_back(nbr_t(v, dist, true));
            }
        }
        // Sort by distance
        std::sort(nbrs.begin(), nbrs.end(),
            [](const nbr_t& a, const nbr_t& b) {
                return a.get_distance() < b.get_distance();
            });
    }

    logger.info(fmt::format("Initial complete graph with {} vertices:", num_vertices_));
    for (vertex_id_t u = 0; u < std::min(num_vertices_, static_cast<vec_num_t>(5)); ++u) {
        const auto& nbrs = nbrs_arr[u];
        std::string nbr_list;
        for (size_t i = 0; i < std::min(nbrs.size(), static_cast<size_t>(8)); ++i) {
            nbr_list += fmt::format("({}, {:.3f})", nbrs[i].get_id(), nbrs[i].get_distance());
            if (i < std::min(nbrs.size(), static_cast<size_t>(8)) - 1) nbr_list += ", ";
        }
        if (nbrs.size() > 8) nbr_list += "...";
        logger.info(fmt::format("  v{} -> [{}]", u, nbr_list));
    }
    if (num_vertices_ > 5) {
        logger.info("  ... (showing first 5 vertices)");
    }

    const size_t initial_edges = num_vertices_ * (num_vertices_ - 1);

    // Create PropagateEngine and TriangleUpdater
    const ratio_t scale_coeffs = 1.0;
    const ratio_t shifted_coeffs = 0.0;
    const vec_num_t max_nbr_size = 6;

    flat_graph_->set_max_nbr_size(max_nbr_size);

    propagate_engine_ss_t propagate_engine(num_vertices_, *dist_func_);
    propagate_engine.set_graph(*flat_graph_);

    auto triangle_updater = propagate_engine.make_updater<triangle_updater_t>(scale_coeffs, shifted_coeffs);

    // Apply triangle pruning for 5 iterations
    propagate_engine.run(5, triangle_updater);

    logger.info(fmt::format("After 5 iterations of triangle pruning:"));
    for (vertex_id_t u = 0; u < std::min(num_vertices_, static_cast<vec_num_t>(5)); ++u) {
        const auto& nbrs = nbrs_arr[u];
        std::string nbr_list;
        for (size_t i = 0; i < nbrs.size(); ++i) {
            nbr_list += fmt::format("({}, {:.3f})", nbrs[i].get_id(), nbrs[i].get_distance());
            if (i < nbrs.size() - 1) nbr_list += ", ";
        }
        logger.info(fmt::format("  v{} -> [{}]", u, nbr_list));
    }
    if (num_vertices_ > 5) {
        logger.info("  ... (showing first 5 vertices)");
    }

    // Verify RNG property
    EXPECT_TRUE(verify_rng_property(nbrs_arr, scale_coeffs, shifted_coeffs))
        << "Graph does not satisfy RNG property after triangle pruning";

    // Verify that pruning reduced edges
    size_t total_edges_after = 0;
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        total_edges_after += nbrs_arr[u].size();
    }

    logger.info(fmt::format("Edges before: {}, after: {}, reduction: {:.1f}%",
                           initial_edges, total_edges_after,
                           100.0 * (initial_edges - total_edges_after) / initial_edges));

    EXPECT_LT(total_edges_after, initial_edges)
        << "Triangle pruning should reduce number of edges";

    // Verify max_nbr_size constraint
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        EXPECT_LE(nbrs_arr[u].size(), max_nbr_size)
            << fmt::format("Vertex {} has {} neighbors, exceeding max_nbr_size={}",
                          u, nbrs_arr[u].size(), max_nbr_size);
    }
}

TEST_F(PropagateEngineCorrectnessTest, IntegratedRandomAndReverseUpdater) {
    // This test integrates both RandomUpdater and ReverseUpdater
    // Step 1: Start with empty graph
    // Step 2: Apply RandomUpdater to generate asymmetric random edges
    // Step 3: Apply ReverseUpdater to make it bidirectional
    // Step 4: Verify bidirectionality

    auto& nbrs_arr = flat_graph_->get_nbrs_arr();

    // Step 1: Start with empty neighbor arrays
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        nbrs_arr[u].clear();
    }

    logger.info("Step 1: Starting with empty graph");

    propagate_engine_ss_t propagate_engine(num_vertices_, *dist_func_);
    propagate_engine.set_graph(*flat_graph_);

    // Step 2: Apply RandomUpdater to generate asymmetric edges
    const vec_num_t rand_gen_size = 5;
    auto random_updater = propagate_engine.make_updater<random_updater_t>(rand_gen_size);

    propagate_engine.run(1, random_updater);

    size_t edges_after_random = 0;
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        edges_after_random += nbrs_arr[u].size();
    }

    logger.info(fmt::format("Step 2: After RandomUpdater: {} edges generated", edges_after_random));

    // Check how many edges are missing their reverse
    int missing_reverse_before = 0;
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        for (const auto& nbr : nbrs_arr[u]) {
            vertex_id_t v = nbr.get_id();
            if (!has_edge(nbrs_arr[v], u)) {
                missing_reverse_before++;
            }
        }
    }

    logger.info(fmt::format("Before reverse updater: {} missing reverse edges", missing_reverse_before));

    // RandomUpdater generates asymmetric edges, so there should be missing reverse edges
    EXPECT_GT(missing_reverse_before, 0)
        << "RandomUpdater should generate asymmetric edges (missing reverse edges)";

    // Step 3: Apply reverse updater
    auto reverse_updater = propagate_engine.make_updater<reverse_updater_t>();
    propagate_engine.run(1, reverse_updater);

    size_t edges_after_reverse = 0;
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        edges_after_reverse += nbrs_arr[u].size();
    }

    logger.info(fmt::format("Step 3: After reverse updater: {} edges (increase: {})",
                           edges_after_reverse,
                           edges_after_reverse - edges_after_random));

    // Verify that edges increased
    EXPECT_GT(edges_after_reverse, edges_after_random)
        << "ReverseUpdater should add reverse edges";

    // Step 4: Verify bidirectionality
    int missing_reverse_after = 0;
    int distance_mismatch = 0;

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        for (const auto& nbr : nbrs_arr[u]) {
            vertex_id_t v = nbr.get_id();
            distance_t forward_dist = nbr.get_distance();

            distance_t reverse_dist;
            if (!has_edge(nbrs_arr[v], u, &reverse_dist)) {
                missing_reverse_after++;
            } else {
                if (std::abs(forward_dist - reverse_dist) > 1e-5) {
                    distance_mismatch++;
                }
            }
        }
    }

    logger.info(fmt::format("Step 4: Bidirectionality check: {} missing reverse, {} distance mismatches",
                           missing_reverse_after, distance_mismatch));

    EXPECT_EQ(missing_reverse_after, 0)
        << "All forward edges should have corresponding reverse edges after reverse updater";
    EXPECT_EQ(distance_mismatch, 0)
        << "Forward and reverse edge distances should match";
}

TEST_F(PropagateEngineCorrectnessTest, ScaledTrianglePruning) {
    // Test with scale_coeffs > 1.0 for more conservative pruning (keeping more edges)
    auto& nbrs_arr = flat_graph_->get_nbrs_arr();

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        nbr_arr_t& nbrs = nbrs_arr[u];
        for (vertex_id_t v = 0; v < num_vertices_; ++v) {
            if (v != u) {
                distance_t dist = compute_distance(u, v);
                nbrs.push_back(nbr_t(v, dist, true));
            }
        }
        std::sort(nbrs.begin(), nbrs.end(),
            [](const nbr_t& a, const nbr_t& b) {
                return a.get_distance() < b.get_distance();
            });
    }

    // Use scale_coeffs = 1.2: threshold = d / 1.2 ≈ 0.83*d (smaller, more conservative)
    const ratio_t scale_coeffs = 1.2;
    const ratio_t shifted_coeffs = 0.0;
    const vec_num_t max_nbr_size = 6;

    flat_graph_->set_max_nbr_size(max_nbr_size);

    propagate_engine_ss_t propagate_engine(num_vertices_, *dist_func_);
    propagate_engine.set_graph(*flat_graph_);

    auto triangle_updater = propagate_engine.make_updater<triangle_updater_t>(scale_coeffs, shifted_coeffs);

    propagate_engine.run(5, triangle_updater);

    logger.info("Testing with scale_coeffs = 1.2 (conservative pruning, more edges):");
    for (vertex_id_t u = 0; u < std::min(num_vertices_, static_cast<vec_num_t>(5)); ++u) {
        const auto& nbrs = nbrs_arr[u];
        std::string nbr_list;
        for (size_t i = 0; i < nbrs.size(); ++i) {
            nbr_list += fmt::format("({}, {:.3f})", nbrs[i].get_id(), nbrs[i].get_distance());
            if (i < nbrs.size() - 1) nbr_list += ", ";
        }
        logger.info(fmt::format("  v{} -> [{}]", u, nbr_list));
    }
    if (num_vertices_ > 5) {
        logger.info("  ... (showing first 5 vertices)");
    }

    // Verify scaled RNG property
    EXPECT_TRUE(verify_rng_property(nbrs_arr, scale_coeffs, shifted_coeffs))
        << "Graph does not satisfy scaled RNG property";

    size_t total_edges = 0;
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        total_edges += nbrs_arr[u].size();
    }

    logger.info(fmt::format("Total edges with scale_coeffs=1.2: {}", total_edges));

    // With scale_coeffs > 1.0, pruning is more conservative
    // Still less than complete graph, but verification is that RNG property holds
    EXPECT_LT(total_edges, num_vertices_ * (num_vertices_ - 1));
}

TEST_F(PropagateEngineCorrectnessTest, NeighborsSortedAfterPruning) {
    // Verify neighbors remain sorted by distance after pruning
    auto& nbrs_arr = flat_graph_->get_nbrs_arr();

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        nbr_arr_t& nbrs = nbrs_arr[u];
        for (vertex_id_t v = 0; v < num_vertices_; ++v) {
            if (v != u) {
                distance_t dist = compute_distance(u, v);
                nbrs.push_back(nbr_t(v, dist, true));
            }
        }
        std::sort(nbrs.begin(), nbrs.end(),
            [](const nbr_t& a, const nbr_t& b) {
                return a.get_distance() < b.get_distance();
            });
    }

    const vec_num_t max_nbr_size = 6;
    flat_graph_->set_max_nbr_size(max_nbr_size);

    propagate_engine_ss_t propagate_engine(num_vertices_, *dist_func_);
    propagate_engine.set_graph(*flat_graph_);

    auto triangle_updater = propagate_engine.make_updater<triangle_updater_t>(1.0, 0.0);

    propagate_engine.run(5, triangle_updater);

    // Check that all neighbor lists remain sorted
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        const auto& nbrs = nbrs_arr[u];
        for (size_t i = 1; i < nbrs.size(); ++i) {
            EXPECT_LE(nbrs[i-1].get_distance(), nbrs[i].get_distance())
                << fmt::format("Neighbors of vertex {} not sorted at position {}", u, i);
        }
    }

    logger.info(fmt::format("All {} vertices have sorted neighbor lists after pruning", num_vertices_));
}

TEST_F(PropagateEngineCorrectnessTest, RandomUpdaterGeneratesEdges) {
    // Test that RandomUpdater generates random edges and writes them to the log table
    auto& nbrs_arr = flat_graph_->get_nbrs_arr();

    // Start with empty neighbor arrays
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        nbrs_arr[u].clear();
    }

    logger.info("Testing RandomUpdater with empty initial graph:");

    propagate_engine_ss_t propagate_engine(num_vertices_, *dist_func_);
    propagate_engine.set_graph(*flat_graph_);

    const vec_num_t rand_gen_size = 5;  // Generate 5 random neighbors per vertex

    auto random_updater = propagate_engine.make_updater<random_updater_t>(rand_gen_size);

    // Run one iteration of random edge generation
    propagate_engine.run(1, random_updater);

    // Count total edges after random generation
    size_t total_edges = 0;
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        total_edges += nbrs_arr[u].size();
    }

    logger.info(fmt::format("After RandomUpdater: {} total edges generated", total_edges));

    // Each vertex should have some edges (up to rand_gen_size, minus self-loops)
    EXPECT_GT(total_edges, 0) << "RandomUpdater should generate some edges";

    // Verify that all edges have valid vertex IDs and distances
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        for (const auto& nbr : nbrs_arr[u]) {
            vertex_id_t v = nbr.get_id();
            distance_t dist = nbr.get_distance();

            EXPECT_LT(v, num_vertices_) << fmt::format("Invalid neighbor ID {} for vertex {}", v, u);
            EXPECT_NE(v, u) << fmt::format("Self-loop detected for vertex {}", u);
            EXPECT_GE(dist, 0.0f) << fmt::format("Negative distance for edge ({}, {})", u, v);

            // Verify distance is correct
            distance_t expected_dist = compute_distance(u, v);
            EXPECT_NEAR(dist, expected_dist, 1e-5)
                << fmt::format("Distance mismatch for edge ({}, {}): got {}, expected {}",
                              u, v, dist, expected_dist);
        }
    }

    // Log some sample edges
    for (vertex_id_t u = 0; u < std::min(num_vertices_, static_cast<vec_num_t>(5)); ++u) {
        const auto& nbrs = nbrs_arr[u];
        std::string nbr_list;
        for (size_t i = 0; i < nbrs.size(); ++i) {
            nbr_list += fmt::format("({}, {:.3f})", nbrs[i].get_id(), nbrs[i].get_distance());
            if (i < nbrs.size() - 1) nbr_list += ", ";
        }
        logger.info(fmt::format("  v{} -> [{}]", u, nbr_list));
    }
}

TEST_F(PropagateEngineCorrectnessTest, RandomUpdaterThreadSafety) {
    // Test that RandomUpdater works correctly in parallel execution
    auto& nbrs_arr = flat_graph_->get_nbrs_arr();

    // Start with empty neighbor arrays
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        nbrs_arr[u].clear();
    }

    propagate_engine_ss_t propagate_engine(num_vertices_, *dist_func_);
    propagate_engine.set_graph(*flat_graph_);

    const vec_num_t rand_gen_size = 10;

    auto random_updater = propagate_engine.make_updater<random_updater_t>(rand_gen_size);

    // Run multiple iterations to stress test thread safety
    propagate_engine.run(3, random_updater);

    // Verify no corruption occurred
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        for (const auto& nbr : nbrs_arr[u]) {
            vertex_id_t v = nbr.get_id();
            EXPECT_LT(v, num_vertices_) << "Invalid neighbor ID after parallel execution";
            EXPECT_NE(v, u) << "Self-loop after parallel execution";
        }
    }

    size_t total_edges = 0;
    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        total_edges += nbrs_arr[u].size();
    }

    logger.info(fmt::format("Thread safety test: {} total edges after 3 iterations", total_edges));
    EXPECT_GT(total_edges, 0) << "Should have generated edges in parallel";
}

// Test with selective scheduling disabled (propagate_engine_noss_t)
TEST_F(PropagateEngineCorrectnessTest, TrianglePruningWithoutSelectiveScheduling) {
    // Create initial complete graph
    auto& nbrs_arr = flat_graph_->get_nbrs_arr();

    for (vertex_id_t u = 0; u < num_vertices_; ++u) {
        nbr_arr_t& nbrs = nbrs_arr[u];
        for (vertex_id_t v = 0; v < num_vertices_; ++v) {
            if (v != u) {
                distance_t dist = compute_distance(u, v);
                nbrs.push_back(nbr_t(v, dist, true));
            }
        }
        std::sort(nbrs.begin(), nbrs.end(),
            [](const nbr_t& a, const nbr_t& b) {
                return a.get_distance() < b.get_distance();
            });
    }

    const ratio_t scale_coeffs = 1.0;
    const ratio_t shifted_coeffs = 0.0;
    const vec_num_t max_nbr_size = 6;

    flat_graph_->set_max_nbr_size(max_nbr_size);

    propagate_engine_noss_t propagate_engine(num_vertices_, *dist_func_);
    propagate_engine.set_graph(*flat_graph_);

    auto triangle_updater = propagate_engine.make_updater<triangle_updater_t>(scale_coeffs, shifted_coeffs);

    propagate_engine.run(5, triangle_updater);

    logger.info("Testing without selective scheduling:");
    for (vertex_id_t u = 0; u < std::min(num_vertices_, static_cast<vec_num_t>(5)); ++u) {
        const auto& nbrs = nbrs_arr[u];
        std::string nbr_list;
        for (size_t i = 0; i < nbrs.size(); ++i) {
            nbr_list += fmt::format("({}, {:.3f})", nbrs[i].get_id(), nbrs[i].get_distance());
            if (i < nbrs.size() - 1) nbr_list += ", ";
        }
        logger.info(fmt::format("  v{} -> [{}]", u, nbr_list));
    }

    EXPECT_TRUE(verify_rng_property(nbrs_arr, scale_coeffs, shifted_coeffs))
        << "Graph does not satisfy RNG property without selective scheduling";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}