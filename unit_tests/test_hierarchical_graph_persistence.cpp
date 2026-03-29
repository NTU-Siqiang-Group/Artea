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

#include <iostream>
#include <filesystem>
#include <chrono>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

class HierarchicalGraphPersistenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Use a small dataset for testing
        config_path = "./configs/datasets.json";
        dataset_name = "sift-1m";
        test_index_dir = "./test_hierarchical_graph_persistence_temp";

        // Clean up any existing test directory
        if (std::filesystem::exists(test_index_dir)) {
            std::filesystem::remove_all(test_index_dir);
        }
    }

    void TearDown() override {
        // Clean up test directory
        if (std::filesystem::exists(test_index_dir)) {
            std::filesystem::remove_all(test_index_dir);
        }
    }

    std::string config_path;
    std::string dataset_name;
    std::string test_index_dir;
};

TEST_F(HierarchicalGraphPersistenceTest, SnapshotAndRestore) {
    // Load dataset
    ARTEA_INFO(fmt::format("Loading dataset: {} from {}", dataset_name, config_path));
    vector_dataset_t dataset(config_path, dataset_name);
    dist_func_t dist_func(dataset.get_base_vecs().get_vec_dim());

    const auto& base_vecs = dataset.get_base_vecs();
    ARTEA_INFO(fmt::format("Dataset loaded: {} vectors, {} dims",
        base_vecs.get_num_vecs(), base_vecs.get_vec_dim()));

    // Create layer configs
    layer_config_t bottom_layer_config(32, 48);
    layer_config_t upper_layer_config(24, 40);

    // Create pruning configs
    artea_graph::pruning_config_t bottom_pruning_config(1.0f, 0.0f);
    artea_graph::pruning_config_t upper_pruning_config(1.0f, 0.0f);
    artea_graph::propagate_config_t propagate_config(4, 14);

    // Create vertices builder config
    greedy_vertices_builder_config_t vertices_builder_config(
        34875.0f, 2.56f, 0.96f, 0.99f, 0.2f, 2048
    );

    // Create hierarchical graph
    ARTEA_INFO("Constructing hierarchical graph...");
    auto start_time = std::chrono::high_resolution_clock::now();

    auto original_graph = artea_graph_factory_t::construct_graph(
        base_vecs,
        bottom_layer_config,
        upper_layer_config,
        bottom_pruning_config,
        upper_pruning_config,
        propagate_config,
        vertices_builder_config
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    ARTEA_INFO(fmt::format("Graph construction completed in {:.2f} s", duration.count() / 1000.0));

    const layer_num_t num_layers = original_graph.get_num_layers();
    ARTEA_INFO(fmt::format("Constructed graph with {} layers", num_layers));

    // Snapshot the graph
    ARTEA_INFO(fmt::format("Snapshotting graph to {}...", test_index_dir));
    nlohmann::json metadata;
    metadata["test"] = "hierarchical_graph_persistence";
    hierarchical_graph_file_manager_t::snapshot(original_graph, test_index_dir, metadata);
    ARTEA_INFO("Snapshot completed");

    // Restore the graph
    ARTEA_INFO("Restoring graph from snapshot...");
    artea_graph_index_t restored_graph = hierarchical_graph_file_manager_t::restore(
        test_index_dir,
        base_vecs
    );
    ARTEA_INFO("Restore completed");

    // Verify: Number of layers
    EXPECT_EQ(restored_graph.get_num_layers(), original_graph.get_num_layers())
        << "Number of layers mismatch";

    // Verify: Entry point
    EXPECT_EQ(restored_graph.get_entry_point(), original_graph.get_entry_point())
        << "Entry point mismatch";

    // Verify: Layer configs
    EXPECT_EQ(restored_graph.bottom_layer_config().max_nbr_size(),
              original_graph.bottom_layer_config().max_nbr_size())
        << "Bottom layer max_nbr_size mismatch";
    EXPECT_EQ(restored_graph.upper_layer_config().max_nbr_size(),
              original_graph.upper_layer_config().max_nbr_size())
        << "Upper layer max_nbr_size mismatch";

    // Verify: Each layer's structure
    for (layer_id_t layer_id = 0; layer_id < num_layers; ++layer_id) {
        const auto& original_layer = original_graph.get_layer_graph(layer_id);
        const auto& restored_layer = restored_graph.get_layer_graph(layer_id);

        // Check number of vertices
        EXPECT_EQ(restored_layer.get_num_vertices(), original_layer.get_num_vertices())
            << fmt::format("Layer {} vertex count mismatch", layer_id);

        // Check layer vecs
        const auto& original_layer_vecs = original_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);
        const auto& restored_layer_vecs = restored_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);

        EXPECT_EQ(restored_layer_vecs.get_num_vecs(), original_layer_vecs.get_num_vecs())
            << fmt::format("Layer {} vecs count mismatch", layer_id);

        // Check graph structure (neighbors)
        const auto& original_nbrs_arr = original_layer.get_nbrs_arr();
        const auto& restored_nbrs_arr = restored_layer.get_nbrs_arr();

        EXPECT_EQ(restored_nbrs_arr.size(), original_nbrs_arr.size())
            << fmt::format("Layer {} nbrs_arr size mismatch", layer_id);

        for (vertex_num_t v = 0; v < original_layer.get_num_vertices(); ++v) {
            const auto& original_nbrs = original_nbrs_arr[v];
            const auto& restored_nbrs = restored_nbrs_arr[v];

            EXPECT_EQ(restored_nbrs.size(), original_nbrs.size())
                << fmt::format("Layer {} vertex {} neighbor count mismatch", layer_id, v);

            for (size_t i = 0; i < original_nbrs.size(); ++i) {
                EXPECT_EQ(restored_nbrs[i], original_nbrs[i])
                    << fmt::format("Layer {} vertex {} neighbor {} mismatch", layer_id, v, i);
            }
        }

        ARTEA_INFO(fmt::format("Layer {} verification passed", layer_id));
    }

    // Verify: Inter-layer links
    const auto& original_inter_layer_links = original_graph.get_inter_layer_links();
    const auto& restored_inter_layer_links = restored_graph.get_inter_layer_links();

    for (layer_id_t layer_id = 1; layer_id < num_layers; ++layer_id) {
        const auto& original_links = original_inter_layer_links.get_layer_links(layer_id);
        const auto& restored_links = restored_inter_layer_links.get_layer_links(layer_id);

        EXPECT_EQ(restored_links.size(), original_links.size())
            << fmt::format("Layer {} inter-layer links size mismatch", layer_id);

        for (size_t i = 0; i < original_links.size(); ++i) {
            EXPECT_EQ(restored_links[i], original_links[i])
                << fmt::format("Layer {} inter-layer link {} mismatch", layer_id, i);
        }

        ARTEA_INFO(fmt::format("Layer {} inter-layer links verification passed", layer_id));
    }

    ARTEA_INFO("All verifications passed!");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
