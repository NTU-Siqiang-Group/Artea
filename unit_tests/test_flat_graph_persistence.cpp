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
#include <vector>
#include <memory>
#include <filesystem>
#include <chrono>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    std::string temp_dir;
    layer_config_t layer_config{16, 32};
    conv_graph::pruning_config_t pruning_config{1.0f, 0.0f};
    conv_graph::propagate_config_t propagate_config{4, 14};
    bool verbose;
} g_config;

class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }
        logger.info(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset_ = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dist_func_ = std::make_unique<dist_func_t>(dataset_->get_base_vecs().get_vec_dim());

        const auto& base_vecs = dataset_->get_base_vecs();
        if (g_config.verbose) {
            logger.info(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
            logger.info(fmt::format("Vector dimension: {}", base_vecs.get_vec_dim()));
        }

        // Create temp directory for snapshots
        std::filesystem::create_directories(g_config.temp_dir);
    }

    void cleanup() {
        if (std::filesystem::exists(g_config.temp_dir)) {
            std::filesystem::remove_all(g_config.temp_dir);
        }
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
};

class FlatGraphPersistenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto& provider = DataProvider::instance();
        dataset_ = &provider.get_dataset();
        dist_func_ = &provider.get_dist_func();
    }

    vector_dataset_t* dataset_;
    dist_func_t* dist_func_;
};

TEST_F(FlatGraphPersistenceTest, FlatGraphSnapshotRestore) {
    const auto& base_vecs = dataset_->get_base_vecs();

    logger.info("Building flat graph for persistence test...");

    // Build original flat graph
    conv_graph_factory_t factory;
    flat_graph_t original_graph = factory.construct_graph(
        base_vecs,
        g_config.layer_config,
        g_config.pruning_config,
        g_config.propagate_config
    );

    logger.info(fmt::format("Original graph built with {} vertices", original_graph.get_num_vertices()));

    // Snapshot the graph
    std::string snapshot_dir = g_config.temp_dir + "/flat_graph_snapshot";
    nlohmann::json metadata;
    metadata["test_name"] = "FlatGraphSnapshotRestore";
    metadata["dataset"] = g_config.dataset_name;

    logger.info(fmt::format("Snapshotting graph to {}", snapshot_dir));
    flat_graph_file_manager_t::snapshot(original_graph, snapshot_dir, metadata);

    // Restore the graph
    logger.info("Restoring graph from snapshot...");
    flat_graph_t restored_graph = flat_graph_file_manager_t::restore(snapshot_dir, base_vecs);

    // Verify consistency
    logger.info("Verifying graph consistency...");

    // Check basic properties
    EXPECT_EQ(original_graph.get_num_vertices(), restored_graph.get_num_vertices())
        << "Number of vertices should match";

    EXPECT_EQ(original_graph.layer_config().max_nbr_size(), restored_graph.layer_config().max_nbr_size())
        << "Max neighbor size should match";

    EXPECT_EQ(original_graph.layer_config().reserved_nbr_size(), restored_graph.layer_config().reserved_nbr_size())
        << "Reserved neighbor size should match";

    EXPECT_FLOAT_EQ(original_graph.pruning_config().scale_coeffs(), restored_graph.pruning_config().scale_coeffs())
        << "Scale coefficients should match";

    EXPECT_FLOAT_EQ(original_graph.pruning_config().shifted_coeffs(), restored_graph.pruning_config().shifted_coeffs())
        << "Shifted coefficients should match";

    EXPECT_EQ(original_graph.propagate_config().num_build_loops(), restored_graph.propagate_config().num_build_loops())
        << "Number of build loops should match";

    EXPECT_EQ(original_graph.propagate_config().num_triu_iters(), restored_graph.propagate_config().num_triu_iters())
        << "Number of triangle updater iterations should match";

    // Check neighbor arrays
    const auto& original_nbrs = original_graph.get_nbrs_arr();
    const auto& restored_nbrs = restored_graph.get_nbrs_arr();

    EXPECT_EQ(original_nbrs.size(), restored_nbrs.size())
        << "Neighbor array size should match";

    uint32_t mismatch_count = 0;
    for (uint32_t i = 0; i < original_nbrs.size(); ++i) {
        if (original_nbrs[i].size() != restored_nbrs[i].size()) {
            mismatch_count++;
            if (mismatch_count <= 5 && g_config.verbose) {
                logger.warn(fmt::format("Vertex {} neighbor count mismatch: {} vs {}",
                    i, original_nbrs[i].size(), restored_nbrs[i].size()));
            }
        } else {
            // Check neighbor content
            for (uint32_t j = 0; j < original_nbrs[i].size(); ++j) {
                EXPECT_EQ(original_nbrs[i][j].get_id(), restored_nbrs[i][j].get_id())
                    << fmt::format("Vertex {} neighbor {} ID mismatch", i, j);
                EXPECT_FLOAT_EQ(original_nbrs[i][j].get_distance(), restored_nbrs[i][j].get_distance())
                    << fmt::format("Vertex {} neighbor {} distance mismatch", i, j);
            }
        }
    }

    EXPECT_EQ(mismatch_count, 0) << "All neighbor arrays should match exactly";

    logger.info("Flat graph snapshot/restore test passed!");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_flat_graph_persistence");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--temp-dir").default_value(std::string("./test_flat_graph_persistence_temp"));
    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.temp_dir = program.get<std::string>("--temp-dir");
    g_config.verbose = program.get<bool>("--verbose");

    // Print test configuration
    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Config path: " << g_config.config_path << std::endl;
    std::cout << "Temp directory: " << g_config.temp_dir << std::endl;
    std::cout << "Verbose: " << (g_config.verbose ? "true" : "false") << std::endl;
    std::cout << "==========================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    // Cleanup
    DataProvider::instance().cleanup();

    return result;
}