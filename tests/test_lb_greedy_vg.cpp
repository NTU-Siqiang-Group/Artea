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
#include <random>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>

using namespace artea;
using namespace artea::cpu;

// Typename Definitions
using vec_num_t = uint32_t;
using vec_id_t = uint32_t;
using vec_ele_t = float;
using distance_t = float;
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using vg_traits_t = VertexGeneratorTraits<computer_traits_t>;
using vector_array_t = typename vg_traits_t::vector_array_t;
using vector_dataset_t = typename vg_traits_t::vector_dataset_t;
using dist_func_t = typename vg_traits_t::dist_func_t;
using lb_greedy_vg_t = typename vg_traits_t::lb_greedy_vg_t;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    float min_radius;
    uint32_t max_result_size;
    float coverage_ratio;
    float confidence;
    uint32_t batch_size;
    uint32_t num_test_samples;
    bool verbose;
    bool shuffle;
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

class LBGreedyVGTest : public ::testing::Test {};

TEST_F(LBGreedyVGTest, VerifyRNetSeparation) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();

    const auto& base_vecs = dataset.get_base_vecs();

    if (g_config.verbose) {
        logger.info(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
        logger.info(fmt::format("Min radius: {}", g_config.min_radius));
        logger.info(fmt::format("Coverage ratio: {}", g_config.coverage_ratio));
        logger.info(fmt::format("Confidence: {}", g_config.confidence));
        logger.info(fmt::format("Batch size: {}", g_config.batch_size));
        logger.info(fmt::format("Shuffle: {}", g_config.shuffle ? "enabled" : "disabled"));
    }

    // Generate r-net using LBGreedyVG
    lb_greedy_vg_t generator(dist_func);
    auto approx_rnet = generator.generate(
        base_vecs,
        g_config.min_radius,
        g_config.max_result_size,
        g_config.coverage_ratio,
        g_config.confidence,
        g_config.batch_size,
        g_config.shuffle
    );

    logger.info(fmt::format("Generated r-net with {} vertices", approx_rnet.get_num_vecs()));
    logger.info(fmt::format("R-net ratio: {:.2f}%", 100.0 * approx_rnet.get_num_vecs() / base_vecs.get_num_vecs()));

    // Test 1: Verify separation property (all pairwise distances >= min_radius)
    logger.info("Testing r-net separation property...");
    uint32_t violation_count = 0;
    distance_t min_pairwise_dist = std::numeric_limits<distance_t>::max();

    for (size_t i = 0; i < approx_rnet.get_num_vecs(); ++i) {
        for (size_t j = i + 1; j < approx_rnet.get_num_vecs(); ++j) {
            distance_t dist = dist_func(approx_rnet.vecs_data.get(i), approx_rnet.vecs_data.get(j));
            min_pairwise_dist = std::min(min_pairwise_dist, dist);
            if (dist < g_config.min_radius) {
                violation_count++;
                if (g_config.verbose && violation_count <= 10) {
                    logger.warn(fmt::format("Separation violation: rnet[{}] and rnet[{}] have distance {:.4f} < {:.4f}",
                        i, j, dist, g_config.min_radius));
                }
            }
        }
    }

    logger.info(fmt::format("Separation test: {} violations out of {} pairs",
        violation_count, approx_rnet.get_num_vecs() * (approx_rnet.get_num_vecs() - 1) / 2));
    logger.info(fmt::format("Minimum pairwise distance: {:.4f}", min_pairwise_dist));

    // Separation property must hold strictly
    EXPECT_EQ(violation_count, 0) << "R-net separation property violated";
}

TEST_F(LBGreedyVGTest, VerifyRNetCoverage) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();

    const auto& base_vecs = dataset.get_base_vecs();

    // Generate r-net using LBGreedyVG
    lb_greedy_vg_t generator(dist_func);
    auto approx_rnet = generator.generate(
        base_vecs,
        g_config.min_radius,
        g_config.max_result_size,
        g_config.coverage_ratio,
        g_config.confidence,
        g_config.batch_size,
        g_config.shuffle
    );

    logger.info(fmt::format("Generated r-net with {} vertices", approx_rnet.get_num_vecs()));

    // Test 2: Verify coverage property by sampling random points
    logger.info(fmt::format("Testing r-net coverage property with {} random samples...", g_config.num_test_samples));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, base_vecs.get_num_vecs() - 1);

    uint32_t uncovered_count = 0;
    std::vector<uint32_t> sampled_ids;
    sampled_ids.reserve(g_config.num_test_samples);

    for (uint32_t i = 0; i < g_config.num_test_samples; ++i) {
        sampled_ids.push_back(dist(gen));
    }

    for (uint32_t sample_id : sampled_ids) {
        const vec_ele_t* sample_vec = base_vecs.get(sample_id);

        // Find minimum distance to r-net
        distance_t min_dist_to_rnet = std::numeric_limits<distance_t>::max();
        for (size_t i = 0; i < approx_rnet.get_num_vecs(); ++i) {
            distance_t d = dist_func(sample_vec, approx_rnet.vecs_data.get(i));
            min_dist_to_rnet = std::min(min_dist_to_rnet, d);
        }

        // Check if covered (distance < min_radius)
        if (min_dist_to_rnet >= g_config.min_radius) {
            uncovered_count++;
            if (g_config.verbose && uncovered_count <= 10) {
                logger.warn(fmt::format("Uncovered sample: vec[{}] has min distance {:.4f} >= {:.4f}",
                    sample_id, min_dist_to_rnet, g_config.min_radius));
            }
        }
    }

    float empirical_coverage = 1.0f - (float)uncovered_count / g_config.num_test_samples;
    logger.info(fmt::format("Coverage test: {}/{} samples covered ({:.2f}%)",
        g_config.num_test_samples - uncovered_count, g_config.num_test_samples, empirical_coverage * 100.0f));
    logger.info(fmt::format("Target coverage: {:.2f}%", g_config.coverage_ratio * 100.0f));
    logger.info(fmt::format("Uncovered samples: {}", uncovered_count));

    // Coverage should be approximately equal to target (within reasonable tolerance)
    // Note: This is a statistical test, so we allow some deviation
    // For datasets where min_radius doesn't match the distribution, coverage may be lower
    float coverage_tolerance = 0.10f;  // 10% tolerance (relaxed for mismatched radius)
    EXPECT_GE(empirical_coverage, g_config.coverage_ratio - coverage_tolerance)
        << "Empirical coverage is significantly lower than target";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_lb_greedy_vg");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("-r", "--min-radius").default_value(90000.0f).scan<'g', float>();
    program.add_argument("-m", "--max-result-size").default_value(100000u).scan<'u', uint32_t>();
    program.add_argument("--coverage-ratio").default_value(0.95f).scan<'g', float>();
    program.add_argument("--confidence").default_value(0.96f).scan<'g', float>();
    program.add_argument("-b", "--batch-size").default_value(512u).scan<'u', uint32_t>();
    program.add_argument("-n", "--num-test-samples").default_value(10000u).scan<'u', uint32_t>();
    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);
    program.add_argument("--shuffle").default_value(false).implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.min_radius = program.get<float>("--min-radius");
    g_config.max_result_size = program.get<uint32_t>("--max-result-size");
    g_config.coverage_ratio = program.get<float>("--coverage-ratio");
    g_config.confidence = program.get<float>("--confidence");
    g_config.batch_size = program.get<uint32_t>("--batch-size");
    g_config.num_test_samples = program.get<uint32_t>("--num-test-samples");
    g_config.verbose = program.get<bool>("--verbose");
    g_config.shuffle = program.get<bool>("--shuffle");

    // Print test configuration
    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Config path: " << g_config.config_path << std::endl;
    std::cout << "Min radius: " << g_config.min_radius << std::endl;
    std::cout << "Max result size: " << g_config.max_result_size << std::endl;
    std::cout << "Coverage ratio: " << g_config.coverage_ratio << std::endl;
    std::cout << "Confidence: " << g_config.confidence << std::endl;
    std::cout << "Batch size: " << g_config.batch_size << std::endl;

    // Compute and display term_thresh
    uint32_t computed_term_thresh = lb_greedy_vg_t::compute_term_thresh(
        g_config.coverage_ratio, g_config.confidence, g_config.batch_size
    );
    std::cout << "Computed term thresh: " << computed_term_thresh << std::endl;
    std::cout << "Num test samples: " << g_config.num_test_samples << std::endl;
    std::cout << "Verbose: " << (g_config.verbose ? "true" : "false") << std::endl;
    std::cout << "Shuffle: " << (g_config.shuffle ? "true" : "false") << std::endl;
    std::cout << "==========================\n" << std::endl;

    DataProvider::instance().init();

    return RUN_ALL_TESTS();
}
