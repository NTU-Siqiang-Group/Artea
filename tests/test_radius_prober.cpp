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
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>

using namespace artea;
using namespace artea::cpu;

// Typename Definitions
using vec_num_t = uint32_t;
using vec_ele_t = float;
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using vector_array_t = typename computer_traits_t::vector_array_t;
using vector_dataset_t = typename computer_traits_t::vector_dataset_t;
using dist_func_t = typename computer_traits_t::dist_func_t;
using radius_prober_t = typename computer_traits_t::radius_prober_t;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    int num_samples;
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
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
};

class RadiusProberTest : public ::testing::Test {};

TEST_F(RadiusProberTest, ProbeQuantile) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();

    const auto& base_vecs = dataset.get_base_vecs();

    if (g_config.verbose) {
        logger.info(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
        logger.info(fmt::format("Sampling {} vectors for probe", g_config.num_samples));
    }

    radius_prober_t prober(dist_func);

    // Probe 1% quantile with specified number of vectors
    float quantile = 0.01f;
    vec_num_t num_vecs = std::min(g_config.num_samples, static_cast<int>(base_vecs.get_num_vecs()));

    auto result = prober.probe(base_vecs, quantile, num_vecs);

    logger.info(fmt::format("Quantile Result:"));
    logger.info(fmt::format("  {:.1f}% quantile: {:.4f}", result.quantile * 100.0f, result.radius));
    logger.info(fmt::format("  Vectors sampled: {}", result.num_vecs_sampled));
    logger.info(fmt::format("  Distances computed: {}", result.num_distances_computed));

    // Verify result
    EXPECT_EQ(result.quantile, quantile);
    EXPECT_EQ(result.num_vecs_sampled, num_vecs);
    EXPECT_EQ(result.num_distances_computed, (num_vecs * (num_vecs - 1)) / 2);
    EXPECT_GT(result.radius, 0.0f);
}

TEST_F(RadiusProberTest, ProbeMultipleQuantiles) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();

    const auto& base_vecs = dataset.get_base_vecs();

    if (g_config.verbose) {
        logger.info(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
        logger.info(fmt::format("Sampling {} vectors for probe", g_config.num_samples));
    }

    radius_prober_t prober(dist_func);

    vec_num_t num_vecs = std::min(g_config.num_samples, static_cast<int>(base_vecs.get_num_vecs()));

    // Probe different quantiles
    std::vector<float> quantiles = {0.01f, 0.05f, 0.10f, 0.25f, 0.50f, 0.75f, 0.99f};
    std::vector<typename radius_prober_t::ProbeResult> results;

    logger.info("Multiple Quantile Results:");
    for (float q : quantiles) {
        auto result = prober.probe(base_vecs, q, num_vecs);
        results.push_back(result);

        logger.info(fmt::format("  {:.1f}% quantile: {:.4f}",
            result.quantile * 100.0f, result.radius));
    }

    // Verify results are sorted (higher quantiles should have higher or equal values)
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i-1].radius, results[i].radius);
    }
}

TEST_F(RadiusProberTest, ProbeSmallQuantiles) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();

    const auto& base_vecs = dataset.get_base_vecs();

    if (g_config.verbose) {
        logger.info(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
        logger.info(fmt::format("Sampling {} vectors for probe", g_config.num_samples));
    }

    radius_prober_t prober(dist_func);

    vec_num_t num_vecs = std::min(g_config.num_samples, static_cast<int>(base_vecs.get_num_vecs()));

    // Probe very small quantiles (0.01%, 0.05%, 0.15%)
    std::vector<float> quantiles = {0.0001f, 0.0005f, 0.0015f};
    std::vector<typename radius_prober_t::ProbeResult> results;

    logger.info("Small Quantile Results:");
    for (float q : quantiles) {
        auto result = prober.probe(base_vecs, q, num_vecs);
        results.push_back(result);

        logger.info(fmt::format("  {:.2f}% quantile: {:.4f}",
            result.quantile * 100.0f, result.radius));
    }

    // Verify results are sorted (higher quantiles should have higher or equal values)
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i-1].radius, results[i].radius);
    }

    // Verify all radii are positive
    for (const auto& result : results) {
        EXPECT_GT(result.radius, 0.0f);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_radius_prober");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("-n", "--num-samples").default_value(10000).scan<'i', int>();
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
    g_config.num_samples = program.get<int>("--num-samples");
    g_config.verbose = program.get<bool>("--verbose");

    DataProvider::instance().init();

    return RUN_ALL_TESTS();
}
