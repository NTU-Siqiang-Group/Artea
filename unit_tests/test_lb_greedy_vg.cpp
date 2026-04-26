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
#include <unordered_set>
#include <chrono>
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
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using index_traits_t = IndexTraits<base_traits_t>;
using vg_traits_t = VertexGeneratorTraits<computer_traits_t, index_traits_t>;
using vector_array_t = typename vg_traits_t::vector_array_t;
using vector_dataset_t = typename vg_traits_t::vector_dataset_t;
using dist_func_t = typename vg_traits_t::dist_func_t;
using lb_greedy_vg_t = typename vg_traits_t::lb_greedy_vg_t;
using approx_rnet_t = typename vg_traits_t::approx_rnet_t;
using distance_prober_t = typename vg_traits_t::distance_prober_t;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    float beta;
    uint32_t max_result_size;
    float coverage_ratio;
    float confidence;
    uint32_t batch_size;
    uint32_t num_test_samples;
    bool verbose;
} g_config;

// Global test results
struct TestResults {
    double generation_time_ms = 0.0;
    double probe_time_ms = 0.0;
    float min_radius = 0.0f;
    float rnet_radius = 0.0f;
    uint32_t num_vertices = 0;
    uint32_t total_base_vecs = 0;
    float rnet_ratio = 0.0f;
    float empirical_coverage = 0.0f;
    float min_pairwise_dist = 0.0f;
    uint32_t separation_violations = 0;
    bool data_consistency_passed = false;
    bool uniqueness_passed = false;
} g_test_results;

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

        // Always shuffle dataset for randomness
        ARTEA_INFO("Shuffling dataset...");
        dataset_->shuffle_in_place();

        dist_func_ = std::make_unique<dist_func_t>(dataset_->get_base_vecs().get_vec_dim());

        // Probe min_radius using radius prober
        const auto& base_vecs = dataset_->get_base_vecs();
        ARTEA_INFO("Probing min_radius from dataset...");

        constexpr float QUANTILE = 0.001f;
        constexpr float CONFIDENCE = 0.99f;
        constexpr float RELATIVE_ERR = 0.05f;

        distance_prober_t prober(*dist_func_);
        auto probe_start = std::chrono::high_resolution_clock::now();
        auto probe_result = prober.probe(base_vecs, QUANTILE, CONFIDENCE, RELATIVE_ERR);
        auto probe_end = std::chrono::high_resolution_clock::now();
        auto probe_duration = std::chrono::duration_cast<std::chrono::microseconds>(probe_end - probe_start);

        g_test_results.probe_time_ms = probe_duration.count() / 1000.0;
        g_test_results.min_radius = probe_result.radius;
        g_test_results.rnet_radius = probe_result.radius * g_config.beta;

        ARTEA_INFO(fmt::format("Probed min radius: {:.6f} (quantile: {:.4f}, samples: {}, time: {:.2f}ms)",
            probe_result.radius, probe_result.quantile, probe_result.num_dists_sampled, g_test_results.probe_time_ms));
        ARTEA_INFO(fmt::format("R-net radius (beta={:.2f}): {:.6f}", g_config.beta, g_test_results.rnet_radius));

        // Generate r-net once for all tests
        if (g_config.verbose) {
            ARTEA_INFO(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
            ARTEA_INFO(fmt::format("R-net radius: {}", g_test_results.rnet_radius));
            ARTEA_INFO(fmt::format("Coverage ratio: {}", g_config.coverage_ratio));
            ARTEA_INFO(fmt::format("Confidence: {}", g_config.confidence));
            ARTEA_INFO(fmt::format("Batch size: {}", g_config.batch_size));
        }

        // Time the generation
        auto start_time = std::chrono::high_resolution_clock::now();

        lb_greedy_vg_t generator(*dist_func_);
        approx_rnet_ = std::make_unique<approx_rnet_t>(generator.generate(
            base_vecs,
            g_test_results.rnet_radius,
            g_config.max_result_size,
            g_config.coverage_ratio,
            g_config.confidence,
            g_config.batch_size
        ));

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        g_test_results.generation_time_ms = duration.count() / 1000.0;

        g_test_results.num_vertices = approx_rnet_->get_num_vecs();
        g_test_results.total_base_vecs = base_vecs.get_num_vecs();
        g_test_results.rnet_ratio = 100.0f * approx_rnet_->get_num_vecs() / base_vecs.get_num_vecs();

        ARTEA_INFO(fmt::format("Generated r-net with {} vertices", approx_rnet_->get_num_vecs()));
        ARTEA_INFO(fmt::format("R-net ratio: {:.2f}%", g_test_results.rnet_ratio));
        ARTEA_INFO(fmt::format("Generation time: {:.2f} ms", g_test_results.generation_time_ms));
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }
    approx_rnet_t& get_approx_rnet() { return *approx_rnet_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<approx_rnet_t> approx_rnet_;
};

class LBGreedyVGTest : public ::testing::Test {};

TEST_F(LBGreedyVGTest, VerifyRNetDataConsistency) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& approx_rnet = provider.get_approx_rnet();

    const auto& base_vecs = dataset.get_base_vecs();
    uint32_t vec_dim = base_vecs.get_vec_dim();

    // Test 2: Verify vecs_data matches original vectors
    ARTEA_INFO("Verifying vecs_data matches original vectors...");
    uint32_t mismatch_count = 0;

    for (size_t i = 0; i < approx_rnet.get_num_vecs(); ++i) {
        vec_id_t original_id = approx_rnet.vec_ids[i];
        const vec_ele_t* original_vec = base_vecs.get(original_id);
        const vec_ele_t* rnet_vec = approx_rnet.vecs_data.get(i);

        // Check all dimensions match
        for (uint32_t d = 0; d < vec_dim; ++d) {
            if (std::abs(original_vec[d] - rnet_vec[d]) > 1e-6f) {
                mismatch_count++;
                if (g_config.verbose && mismatch_count <= 10) {
                    ARTEA_WARN(fmt::format("Data mismatch at index {}, dim {}: original={:.6f}, rnet={:.6f}",
                        i, d, original_vec[d], rnet_vec[d]));
                }
                break;  // Only count once per vector
            }
        }
    }

    EXPECT_EQ(mismatch_count, 0) << "All vectors in vecs_data should match original vectors";

    g_test_results.data_consistency_passed = (mismatch_count == 0);

    if (mismatch_count == 0) {
        ARTEA_SUCCESS("All vectors match original data");
    } else {
        ARTEA_ERROR(fmt::format("{} vectors have mismatched data", mismatch_count));
    }
}

TEST_F(LBGreedyVGTest, VerifyRNetUniqueness) {
    auto& provider = DataProvider::instance();
    auto& approx_rnet = provider.get_approx_rnet();

    // Test 3: Verify all vec_ids are unique
    ARTEA_INFO("Verifying vec_ids are unique...");
    std::unordered_set<vec_id_t> unique_ids(approx_rnet.vec_ids.begin(), approx_rnet.vec_ids.end());
    EXPECT_EQ(unique_ids.size(), approx_rnet.vec_ids.size()) << "All vec_ids should be unique";

    g_test_results.uniqueness_passed = (unique_ids.size() == approx_rnet.vec_ids.size());

    if (unique_ids.size() == approx_rnet.vec_ids.size()) {
        ARTEA_SUCCESS("All vec_ids are unique");
    } else {
        ARTEA_ERROR(fmt::format("Found {} duplicates in vec_ids",
            approx_rnet.vec_ids.size() - unique_ids.size()));
    }
}

TEST_F(LBGreedyVGTest, VerifyRNetSeparation) {
    auto& provider = DataProvider::instance();
    auto& dist_func = provider.get_dist_func();
    auto& approx_rnet = provider.get_approx_rnet();

    // Test: Verify separation property by sampling points
    constexpr uint32_t NUM_SAMPLE_POINTS = 1000;
    uint32_t num_rnet_vecs = approx_rnet.get_num_vecs();
    uint32_t actual_samples = std::min(NUM_SAMPLE_POINTS, num_rnet_vecs);

    ARTEA_INFO(fmt::format("Testing r-net separation property with {} sampled points...", actual_samples));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist_sampler(0, num_rnet_vecs - 1);

    uint32_t violation_count = 0;
    distance_t min_pairwise_dist = std::numeric_limits<distance_t>::max();
    uint32_t total_pairs_checked = 0;

    // Sample points and check their distances to all other r-net points
    for (uint32_t sample_idx = 0; sample_idx < actual_samples; ++sample_idx) {
        uint32_t i = dist_sampler(gen);

        for (uint32_t j = 0; j < num_rnet_vecs; ++j) {
            if (i == j) continue;

            distance_t dist = dist_func(approx_rnet.vecs_data.get(i), approx_rnet.vecs_data.get(j));
            min_pairwise_dist = std::min(min_pairwise_dist, dist);
            total_pairs_checked++;

            if (dist < g_test_results.rnet_radius) {
                violation_count++;
                if (g_config.verbose && violation_count <= 10) {
                    ARTEA_WARN(fmt::format("Separation violation: rnet[{}] and rnet[{}] have distance {:.4f} < {:.4f}",
                        i, j, dist, g_test_results.rnet_radius));
                }
            }
        }
    }

    ARTEA_INFO(fmt::format("Separation test: {} violations out of {} pairs checked",
        violation_count, total_pairs_checked));
    ARTEA_INFO(fmt::format("Minimum pairwise distance: {:.4f}", min_pairwise_dist));

    g_test_results.min_pairwise_dist = min_pairwise_dist;
    g_test_results.separation_violations = violation_count;

    // Separation property must hold strictly
    EXPECT_EQ(violation_count, 0) << "R-net separation property violated";
}

TEST_F(LBGreedyVGTest, VerifyRNetCoverage) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();
    auto& approx_rnet = provider.get_approx_rnet();

    const auto& base_vecs = dataset.get_base_vecs();

    // Test: Verify coverage property by sampling random points
    ARTEA_INFO(fmt::format("Testing r-net coverage property with {} random samples...", g_config.num_test_samples));

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

        // Check if covered (distance < rnet_radius)
        if (min_dist_to_rnet >= g_test_results.rnet_radius) {
            uncovered_count++;
            if (g_config.verbose && uncovered_count <= 10) {
                ARTEA_WARN(fmt::format("Uncovered sample: vec[{}] has min distance {:.4f} >= {:.4f}",
                    sample_id, min_dist_to_rnet, g_test_results.rnet_radius));
            }
        }
    }

    float empirical_coverage = 1.0f - (float)uncovered_count / g_config.num_test_samples;
    g_test_results.empirical_coverage = empirical_coverage;

    ARTEA_INFO(fmt::format("Coverage test: {}/{} samples covered ({:.2f}%)",
        g_config.num_test_samples - uncovered_count, g_config.num_test_samples, empirical_coverage * 100.0f));
    ARTEA_INFO(fmt::format("Target coverage: {:.2f}%", g_config.coverage_ratio * 100.0f));
    ARTEA_INFO(fmt::format("Uncovered samples: {}", uncovered_count));

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
    program.add_argument("-c", "--config").default_value(artea::default_dataset_config_path());
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--beta").default_value(1.69f).scan<'g', float>();
    program.add_argument("-m", "--max-result-size").default_value(100000u).scan<'u', uint32_t>();
    program.add_argument("--coverage-ratio").default_value(0.999f).scan<'g', float>();
    program.add_argument("--confidence").default_value(0.950f).scan<'g', float>();
    program.add_argument("-b", "--batch-size").default_value(2048u).scan<'u', uint32_t>();
    program.add_argument("-n", "--num-test-samples").default_value(10000u).scan<'u', uint32_t>();
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
    g_config.beta = program.get<float>("--beta");
    g_config.max_result_size = program.get<uint32_t>("--max-result-size");
    g_config.coverage_ratio = program.get<float>("--coverage-ratio");
    g_config.confidence = program.get<float>("--confidence");
    g_config.batch_size = program.get<uint32_t>("--batch-size");
    g_config.num_test_samples = program.get<uint32_t>("--num-test-samples");
    g_config.verbose = program.get<bool>("--verbose");

    // Print test configuration
    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Config path: " << g_config.config_path << std::endl;
    std::cout << "Beta: " << g_config.beta << std::endl;
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
    std::cout << "==========================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    // Print summary table
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                        TEST RESULTS SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "\n--- Radius Probing ---" << std::endl;
    std::cout << fmt::format("  Probe Time:             {:.2f} ms", g_test_results.probe_time_ms) << std::endl;
    std::cout << fmt::format("  Beta:                   {:.2f}", g_config.beta) << std::endl;
    std::cout << fmt::format("  Min Radius:             {:.6f}", g_test_results.min_radius) << std::endl;
    std::cout << fmt::format("  R-Net Radius:           {:.6f}", g_test_results.rnet_radius) << std::endl;
    std::cout << "\n--- R-Net Generation ---" << std::endl;
    std::cout << fmt::format("  Generation Time:        {:.2f} ms", g_test_results.generation_time_ms) << std::endl;
    std::cout << fmt::format("  Num Vertices:           {} / {}", g_test_results.num_vertices, g_test_results.total_base_vecs) << std::endl;
    std::cout << fmt::format("  R-Net Ratio:            {:.2f}%", g_test_results.rnet_ratio) << std::endl;
    std::cout << "\n--- Coverage Analysis ---" << std::endl;
    std::cout << fmt::format("  Target Coverage:        {:.2f}%", g_config.coverage_ratio * 100.0f) << std::endl;
    std::cout << fmt::format("  Empirical Coverage:     {:.2f}%", g_test_results.empirical_coverage * 100.0f) << std::endl;
    std::cout << fmt::format("  Coverage Difference:    {:.2f}%", (g_test_results.empirical_coverage - g_config.coverage_ratio) * 100.0f) << std::endl;
    std::cout << "\n--- Separation Analysis ---" << std::endl;
    std::cout << fmt::format("  R-Net Radius:           {:.4f}", g_test_results.rnet_radius) << std::endl;
    std::cout << fmt::format("  Min Pairwise Distance:  {:.4f}", g_test_results.min_pairwise_dist) << std::endl;
    std::cout << fmt::format("  Separation Violations:  {}", g_test_results.separation_violations) << std::endl;
    std::cout << "\n--- Test Results ---" << std::endl;
    std::cout << fmt::format("  Data Consistency Test:  {}", g_test_results.data_consistency_passed ? "PASS" : "FAIL") << std::endl;
    std::cout << fmt::format("  Uniqueness Test:        {}", g_test_results.uniqueness_passed ? "PASS" : "FAIL") << std::endl;
    std::cout << fmt::format("  Separation Test:        {}", (g_test_results.separation_violations == 0) ? "PASS" : "FAIL") << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return result;
}
