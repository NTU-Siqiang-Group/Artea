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
    uint32_t num_samples;
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
        ARTEA_INFO(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset_ = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dist_func_ = std::make_unique<dist_func_t>(dataset_->get_base_vecs().get_vec_dim());
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
};

class DatasetProberTest : public ::testing::Test {};

/**
 * @brief Probe full table: nn_rank 1..128 x all quantiles, plus LID estimate.
 *        Print in chunks of at most 10 nn_rank columns.
 */
TEST_F(DatasetProberTest, Probe) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();

    const auto& base_vecs = dataset.get_base_vecs();
    dataset_prober_t prober(base_vecs, dist_func);

    const vec_num_t num_samples = g_config.num_samples;

    std::vector<float> quantiles = {
        0.001f, 0.01f, 0.05f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.95f, 0.99f
    };

    ARTEA_INFO(fmt::format("Probing dataset with {} samples...", num_samples));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = prober.probe(quantiles, num_samples);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_s = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

    ARTEA_INFO(fmt::format("Probing completed in {:.2f} s ({} samples x 128 ranks)", elapsed_s, num_samples));
    ARTEA_INFO(fmt::format("Estimated LID (Levina-Bickel, k=128): {:.4f}", result.lid));

    // Print table in chunks of at most 10 nn_rank columns
    const uint32_t total_ranks = static_cast<uint32_t>(result.nn_ranks.size());
    const uint32_t cols_per_chunk = 10;

    for (uint32_t chunk_start = 0; chunk_start < total_ranks; chunk_start += cols_per_chunk) {
        uint32_t chunk_end = std::min(chunk_start + cols_per_chunk, total_ranks);

        // Header
        std::string header = fmt::format("  {:>8s}", "quantile");
        for (uint32_t c = chunk_start; c < chunk_end; ++c) {
            header += fmt::format(" {:>10d}", result.nn_ranks[c]);
        }
        ARTEA_INFO(fmt::format("--- nn_rank [{}, {}] ---", result.nn_ranks[chunk_start], result.nn_ranks[chunk_end - 1]));
        ARTEA_INFO(header);

        // Data rows
        for (size_t qi = 0; qi < result.quantiles.size(); ++qi) {
            std::string row = fmt::format("  {:>7.3f}%", result.quantiles[qi] * 100.0f);
            for (uint32_t c = chunk_start; c < chunk_end; ++c) {
                row += fmt::format(" {:>10.2f}", result.table[c][qi]);
            }
            ARTEA_INFO(row);
        }
    }

    // Verify LID is positive and reasonable
    EXPECT_GT(result.lid, 0.0f) << "LID should be positive";

    // Verify: for each quantile, radii should be non-decreasing across nn_ranks
    for (size_t qi = 0; qi < result.quantiles.size(); ++qi) {
        for (uint32_t r = 1; r < total_ranks; ++r) {
            EXPECT_LE(result.table[r - 1][qi], result.table[r][qi])
                << fmt::format("rank ordering violated at quantile={:.3f}: rank {}={:.6f} > rank {}={:.6f}",
                    result.quantiles[qi], r, result.table[r - 1][qi], r + 1, result.table[r][qi]);
        }
    }

    // Verify: for each nn_rank, radii should be non-decreasing across quantiles
    for (uint32_t r = 0; r < total_ranks; ++r) {
        for (size_t qi = 1; qi < result.quantiles.size(); ++qi) {
            EXPECT_LE(result.table[r][qi - 1], result.table[r][qi])
                << fmt::format("quantile ordering violated at rank {}: q={:.3f} ({:.6f}) > q={:.3f} ({:.6f})",
                    r + 1, result.quantiles[qi - 1], result.table[r][qi - 1],
                    result.quantiles[qi], result.table[r][qi]);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_dataset_prober");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--num-samples").default_value(10000u).scan<'u', uint32_t>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.num_samples = program.get<uint32_t>("--num-samples");

    DataProvider::instance().init();

    return RUN_ALL_TESTS();
}
