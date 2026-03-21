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
 * @FilePath: /Artea/tests/test_random_seq.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#include <gtest/gtest.h>
#include <argparse/argparse.hpp> // Assumes p-ranav/argparse usage
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <thread>
#include <random>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <artea/cpu/framework/artea.hpp>

using namespace artea;
using namespace artea::cpu;

using vec_num_t = uint32_t;
using vec_ele_t = float;
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t>;
using random_seq_t = typename base_traits_t::random_seq_t;

// -----------------------------------------------------------------------------
// Global Configuration
// -----------------------------------------------------------------------------
struct TestConfig {
    uint32_t num_vecs = 1024;           // Range [0, num_vecs - 1]
    uint32_t num_rand_nbrs = 1000000;   // Number of samples to generate
};

// Global instance to be accessible within tests
TestConfig g_config;

// -----------------------------------------------------------------------------
// Helper Functions
// -----------------------------------------------------------------------------

/**
 * Calculates statistics, logs them using Artea logger, and asserts correctness.
 *
 * @tparam T The data type of the vector elements.
 * @param data The generated random data.
 * @param num_vecs The upper bound (exclusive) of the generation range.
 * @param method_name Name of the method being tested for logging purposes.
 */
template<typename T>
void verify_and_log_stats(const std::vector<T>& data, const T num_vecs, const std::string& method_name) {
    if (data.empty()) return;

    // Calculate Mean
    long double sum = 0.0;
    for (const auto& val : data) {
        sum += static_cast<long double>(val);
    }
    long double mean = sum / data.size();

    // Calculate Variance
    long double sq_sum = 0.0;
    for (const auto& val : data) {
        long double diff = static_cast<long double>(val) - mean;
        sq_sum += diff * diff;
    }
    long double variance = sq_sum / data.size();

    // Theoretical Values for Uniform Distribution [0, N-1]
    long double N = static_cast<long double>(num_vecs);
    long double theoretical_mean = (N - 1.0) / 2.0;
    long double theoretical_variance = (N * N - 1.0) / 12.0;

    // Log results using Artea
    logger.info(fmt::format("--- Statistics for {} ---", method_name));
    logger.info(fmt::format("    - Generated Mean:         {:.4f} (Theoretical: {:.4f})", mean, theoretical_mean));
    logger.info(fmt::format("    - Generated Variance:     {:.4f} (Theoretical: {:.4f})", variance, theoretical_variance));
    logger.info(fmt::format("    - Generated Std Deviation: {:.4f}", std::sqrt(variance)));

    // GTest Assertions: Verify correctness with a tolerance (e.g., 5% error margin)
    // Note: Statistical tests can be flaky; tolerance should be generous enough for random seeds.
    double tolerance = 0.05;
    EXPECT_NEAR(mean, theoretical_mean, theoretical_mean * tolerance)
        << "Mean is out of expected theoretical range for " << method_name;

    EXPECT_NEAR(variance, theoretical_variance, theoretical_variance * tolerance)
        << "Variance is out of expected theoretical range for " << method_name;
}

// -----------------------------------------------------------------------------
// Test Suite
// -----------------------------------------------------------------------------

class RandomSeqTest : public ::testing::Test {
protected:
    using vec_id_t = uint32_t;

    void SetUp() override {
        // Log the start of a test case
        logger.info("----------------------------------------------------------");
    }

    void TearDown() override {
        // Log the end of a test case
        logger.info("----------------------------------------------------------");
    }
};

TEST_F(RandomSeqTest, CppStandardLibrarySerial) {
    logger.info(" -> Running Quality Check for C++ Standard Library (Serial)...");

    std::vector<vec_id_t> random_data(g_config.num_rand_nbrs);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<vec_id_t> distrib(0, g_config.num_vecs - 1);

    for (uint32_t i = 0; i < g_config.num_rand_nbrs; ++i) {
        random_data[i] = distrib(gen);
    }

    verify_and_log_stats(random_data, g_config.num_vecs, "C++ Standard Library (Serial)");
}

TEST_F(RandomSeqTest, MklSingleThreaded) {
    logger.info(" -> Running Quality Check for Single-Threaded Vectorized (MKL)...");

    random_seq_t mkl_rng(g_config.num_vecs);
    std::vector<vec_id_t> random_data(g_config.num_rand_nbrs);

    mkl_rng.generate(random_data, g_config.num_rand_nbrs);

    verify_and_log_stats(random_data, g_config.num_vecs, "Single-Threaded Vectorized (MKL)");
}

TEST_F(RandomSeqTest, MklMultiThreaded) {
    size_t num_threads = std::thread::hardware_concurrency();
    logger.info(fmt::format(" -> Running Quality Check for Multi-Threaded Vectorized (MKL) with {} threads...", num_threads));

    random_seq_t mkl_rng(g_config.num_vecs);
    std::vector<vec_id_t> random_data(g_config.num_rand_nbrs);

    // Using TBB to generate in parallel chunks
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, g_config.num_rand_nbrs),
        [&](const tbb::blocked_range<uint32_t>& r) {
            std::vector<vec_id_t> local_data(r.size());
            // Generate logic
            mkl_rng.generate(local_data, r.size());
            // Copy back to main buffer
            std::copy(local_data.data(), local_data.data() + r.size(), random_data.data() + r.begin());
        });

    verify_and_log_stats(random_data, g_config.num_vecs, "Multi-Threaded Vectorized (MKL)");
}

// -----------------------------------------------------------------------------
// Main Entry Point
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize Argument Parser
    argparse::ArgumentParser program("test_random_seq");

    program.add_argument("-n", "--num_vecs")
        .help("Upper bound for random number generation (exclusive)")
        .scan<'u', uint32_t>()
        .default_value(uint32_t{1024});

    program.add_argument("-s", "--num_samples")
        .help("Number of random samples to generate for quality check")
        .scan<'u', uint32_t>()
        .default_value(uint32_t{1000000});

    try {
        // Parse known arguments, leaving unknown ones for GTest if any
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Update global configuration
    g_config.num_vecs = program.get<uint32_t>("--num_vecs");
    g_config.num_rand_nbrs = program.get<uint32_t>("--num_samples");

    logger.info("==========================================================");
    logger.info("      Starting Random Number Generator Test Suite (GTest)");
    logger.info(fmt::format("      Config: Vecs={}, Samples={}", g_config.num_vecs, g_config.num_rand_nbrs));
    logger.info("==========================================================");

    return RUN_ALL_TESTS();
}