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
 * @FilePath: /Artea/tests/test_simd_linear.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Comprehensive benchmark suite for SIMD Linear Transformation (Dot Product + Bias)
 *               using Google Test for correctness and Google Benchmark for performance.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <filesystem>
// External Libraries
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>
// Artea Headers
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/utils/simd_linear.hpp> // Include the new Linear header
// Faiss Headers
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/impl/AuxIndexStructures.h>
// HNSWLib Headers
#include <hnswlib/hnswlib.h>

using namespace artea;
using namespace artea::cpu;

// Typename Definitions
using base_traits_t = BaseTraits<uint32_t, float, false>;
// We use DOT metric here as the underlying operation is Dot Product
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::DOT>;
using vector_dataset_t = typename base_traits_t::vector_dataset_t;

// Define aliases for SIMDLinear with different unroll sizes
template <std::size_t UnrollSize>
using artea_simd_linear_t = SIMDLinear<computer_traits_t, UnrollSize>;

// --- Configuration & Data Provider ---

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
} g_config;

/**
 * @brief Singleton class to manage Dataset loading.
 */
class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider instance;
        return instance;
    }

    void init() {
        if (g_config.dataset_name.empty() || g_config.config_path.empty()) {
            throw std::runtime_error("Dataset name and config path must be provided.");
        }

        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }

        logger.info(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));

        auto dataset = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        const auto& base_vecs = dataset->get_base_vecs();
        const auto& query_vecs = dataset->get_query_vecs();

        dim_ = base_vecs.get_vec_dim();

        if (base_vecs.get_num_vecs() < 4 || query_vecs.get_num_vecs() < 1) {
             throw std::runtime_error("Dataset too small (need at least 1 query and 4 base vectors).");
        }

        // Allocate memory for cached test vectors
        query_vec_.resize(dim_);
        targets_.resize(4 * dim_);

        // cache: Use the first query vector to simulate the Projection Vector 'a'
        std::memcpy(query_vec_.data(), query_vecs.get(0), dim_ * sizeof(float));

        // cache: Use the first 4 base vectors as targets 'x'
        for(int i = 0; i < 4; ++i) {
            const float* src = base_vecs.get(i);
            std::memcpy(targets_.data() + i * dim_, src, dim_ * sizeof(float));
        }

        // Initialize a random bias 'b' for testing (e.g., 4.25)
        bias_ = 4.25f;

        logger.info(fmt::format("Dataset loaded successfully. Dimension: {}", dim_));
    }

    uint32_t get_dim() const { return dim_; }
    // Acts as vector 'a' (Projection Vector)
    float* get_vec_a() { return query_vec_.data(); }
    // Acts as vector 'x' (Data Vector)
    float* get_vec_x(int idx) { return targets_.data() + idx * dim_; }
    // The scalar bias 'b'
    float get_bias() const { return bias_; }

private:
    DataProvider() = default;

    uint32_t dim_ = 0;
    float bias_ = 0.0f;
    std::vector<float> query_vec_;
    std::vector<float> targets_; // Stores 4 concatenated vectors
};

// Helper: Naive C++ Implementation for Ax + b (Ground Truth)
float cpp_linear_transform(const float* a, const float* x, float b, size_t d) {
    float dot = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        dot += a[i] * x[i];
    }
    return dot + b;
}

// --- PART 1: Google Test (Correctness) ---

class LinearCorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        // DataProvider is initialized in main
    }
};

TEST_F(LinearCorrectnessTest, VerifyAgainstGroundTruth) {
    auto& provider = DataProvider::instance();
    uint32_t dim = provider.get_dim();
    float* a = provider.get_vec_a();   // Projection vector
    float b = provider.get_bias();     // Scalar bias

    // Check all 4 targets
    for (int i = 0; i < 4; ++i) {
        float* x = provider.get_vec_x(i); // Data vector

        // 0. Ground Truth: Direct C++ calculation
        float gt = cpp_linear_transform(a, x, b, dim);

        // 1. Artea Variants
        artea_simd_linear_t<1> artea_u1(dim);
        artea_simd_linear_t<2> artea_u2(dim);
        artea_simd_linear_t<4> artea_u4(dim);

        EXPECT_NEAR(artea_u1(a, x, b), gt, 1e-4) << "Artea Linear Unroll-1 failed at idx " << i;
        EXPECT_NEAR(artea_u2(a, x, b), gt, 1e-4) << "Artea Linear Unroll-2 failed at idx " << i;
        EXPECT_NEAR(artea_u4(a, x, b), gt, 1e-4) << "Artea Linear Unroll-4 failed at idx " << i;

        // 2. Faiss Reference (Dot Product + Manual Bias Add)
        // Faiss doesn't have "Ax+b" kernel, so we use dot product and add b manually to verify consistency
        float faiss_dot = faiss::fvec_inner_product(a, x, dim);
        EXPECT_NEAR(faiss_dot + b, gt, 1e-4) << "Faiss IP + Bias check failed at idx " << i;

        // 3. HNSWLib Reference
        hnswlib::InnerProductSpace ip_space(dim);
        float hnsw_dist = ip_space.get_dist_func()(a, x, ip_space.get_dist_func_param());
        // HNSW IP distance is (1.0 - dot), so dot = 1.0 - dist
        float hnsw_dot = 1.0f - hnsw_dist;
        EXPECT_NEAR(hnsw_dot + b, gt, 1e-4) << "HNSWLib IP + Bias check failed at idx " << i;
    }
}

// --- PART 2: Google Benchmark (Performance) ---

// 1. Artea Linear Benchmarks
template <std::size_t UnrollSize>
static void BM_Artea_Linear(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    artea_simd_linear_t<UnrollSize> linear_func(provider.get_dim());

    float* a = provider.get_vec_a();
    float* x = provider.get_vec_x(0);
    float b = provider.get_bias();

    for (auto _ : state) {
        benchmark::DoNotOptimize(linear_func(a, x, b));
    }
}
BENCHMARK_TEMPLATE(BM_Artea_Linear, 1)->Name("Artea_Linear_Unroll_1");
BENCHMARK_TEMPLATE(BM_Artea_Linear, 2)->Name("Artea_Linear_Unroll_2");
BENCHMARK_TEMPLATE(BM_Artea_Linear, 4)->Name("Artea_Linear_Unroll_4");

// 2. C++ Baseline (Naive Loop)
static void BM_Cpp_Direct_Linear(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    float* a = provider.get_vec_a();
    float* x = provider.get_vec_x(0);
    float b = provider.get_bias();
    uint32_t dim = provider.get_dim();

    for (auto _ : state) {
        benchmark::DoNotOptimize(cpp_linear_transform(a, x, b, dim));
    }
}
BENCHMARK(BM_Cpp_Direct_Linear)->Name("Cpp_Direct_Linear_Transform");

// 3. Reference: Faiss IP + Bias (Simulated)
// This shows the overhead of calling Faiss IP and then adding a float in C++
static void BM_Ref_Faiss_IP_Plus_Bias(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    float* a = provider.get_vec_a();
    float* x = provider.get_vec_x(0);
    float b = provider.get_bias();
    uint32_t dim = provider.get_dim();

    for (auto _ : state) {
        float dot = faiss::fvec_inner_product(a, x, dim);
        benchmark::DoNotOptimize(dot + b);
    }
}
BENCHMARK(BM_Ref_Faiss_IP_Plus_Bias)->Name("Ref_Faiss_IP_Plus_Bias");

// --- Main Entry ---

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // 1. Parse Arguments
    argparse::ArgumentParser program("test_simd_linear");

    program.add_argument("-c", "--config").default_value(std::string("../datasets.json"))
           .help("Path to dataset config file");
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"))
           .help("Dataset name (e.g. sift-1m).");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");

    // 2. Initialize Data
    logger.info("Initializing Data Provider from Dataset for Linear Tests...");
    try {
        DataProvider::instance().init();
    } catch (const std::exception& e) {
        logger.error(fmt::format("Failed to initialize dataset: {}", e.what()));
        return 1;
    }

    // 3. Run Google Test (Correctness Check)
    logger.info("==========================================================");
    logger.info(" -> Running Linear (ax+b) Correctness Tests (GTest)...");
    logger.info("==========================================================");
    int gtest_result = RUN_ALL_TESTS();

    if (::testing::GTEST_FLAG(list_tests) || gtest_result != 0) {
        return gtest_result;
    }

    if (gtest_result != 0) {
        logger.error("Correctness tests failed! Aborting benchmarks.");
        return gtest_result;
    }

    // 4. Run Google Benchmark (Performance)
    logger.info("==========================================================");
    logger.info(" -> Running Linear (ax+b) Performance Benchmarks...");
    logger.info("==========================================================");
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

    return 0;
}