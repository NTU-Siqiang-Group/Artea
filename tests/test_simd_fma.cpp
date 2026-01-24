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
 * @FilePath: /Artea/tests/test_simd_fma.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Comprehensive benchmark suite for SIMD FMA (Dot Product) calculations
 *               using Google Test for correctness and Google Benchmark for performance.
 *               (Dataset Only Version)
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
#include <artea/cpu/utils/simd_fma.hpp> // Include the new FMA header
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
// We use DOT metric here to represent Inner Product / FMA context
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::DOT>;
using vector_dataset_t = typename base_traits_t::vector_dataset_t;

// Define aliases for SIMDFMA with different unroll sizes
template <std::size_t UnrollSize>
using artea_simd_fma_t = SIMDFMA<computer_traits_t, UnrollSize>;

// --- Configuration & Data Provider ---

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
} g_config;

/**
 * @brief Singleton class to manage Dataset loading.
 *        Ensures data is loaded once and shared between GTest and Benchmark.
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

        // cache: Use the first query vector
        std::memcpy(query_vec_.data(), query_vecs.get(0), dim_ * sizeof(float));

        // cache: Use the first 4 base vectors as targets
        for(int i = 0; i < 4; ++i) {
            const float* src = base_vecs.get(i);
            std::memcpy(targets_.data() + i * dim_, src, dim_ * sizeof(float));
        }

        logger.info(fmt::format("Dataset loaded successfully. Dimension: {}", dim_));

        // Initialize Faiss Index for DistanceComputer (using IndexFlatIP for Inner Product)
        faiss_index_ = std::make_unique<faiss::IndexFlatIP>(dim_);
        faiss_index_->add(4, targets_.data());
    }

    uint32_t get_dim() const { return dim_; }
    float* get_query() { return query_vec_.data(); }
    float* get_target(int idx) { return targets_.data() + idx * dim_; }

    // Returns pointer to raw array of 4 vectors
    float* get_targets_raw() { return targets_.data(); }

    faiss::IndexFlatIP* get_faiss_index() { return faiss_index_.get(); }

private:
    DataProvider() = default;

    uint32_t dim_ = 0;
    std::vector<float> query_vec_;
    std::vector<float> targets_; // Stores 4 concatenated vectors
    std::unique_ptr<faiss::IndexFlatIP> faiss_index_;
};

// Helper: Naive C++ Implementation for Dot Product (Ground Truth)
float cpp_dot_product(const float* x, const float* y, const size_t d) {
    float res = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        res += x[i] * y[i];
    }
    return res;
}

// --- PART 1: Google Test (Correctness) ---

class FMACorrectnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        // DataProvider is initialized in main
    }
};

TEST_F(FMACorrectnessTest, VerifyAgainstGroundTruth) {
    auto& provider = DataProvider::instance();
    uint32_t dim = provider.get_dim();
    float* q = provider.get_query();

    // Check all 4 targets
    for (int i = 0; i < 4; ++i) {
        float* t = provider.get_target(i);
        float gt = cpp_dot_product(q, t, dim);

        // 1. Artea Variants
        artea_simd_fma_t<1> artea_u1(dim);
        artea_simd_fma_t<2> artea_u2(dim);
        artea_simd_fma_t<4> artea_u4(dim);

        EXPECT_NEAR(artea_u1(q, t), gt, 1e-5) << "Artea FMA Unroll-1 failed at idx " << i;
        EXPECT_NEAR(artea_u2(q, t), gt, 1e-5) << "Artea FMA Unroll-2 failed at idx " << i;
        EXPECT_NEAR(artea_u4(q, t), gt, 1e-5) << "Artea FMA Unroll-4 failed at idx " << i;

        // 2. Faiss fvec_inner_product
        EXPECT_NEAR(faiss::fvec_inner_product(q, t, dim), gt, 1e-4) << "Faiss fvec_inner_product failed at idx " << i;

        // 3. HNSWLib (Note: InnerProductSpace in HNSWLib usually returns 1.0f - dot_product)
        hnswlib::InnerProductSpace ip_space(dim);
        float hnsw_res = ip_space.get_dist_func()(q, t, ip_space.get_dist_func_param());
        // Recover dot product: dot = 1.0f - dist
        float hnsw_dot = 1.0f - hnsw_res;
        EXPECT_NEAR(hnsw_dot, gt, 1e-4) << "HNSWLib failed at idx " << i;
    }

    // 4. Faiss DistanceComputer (Batch & Single)
    auto index = provider.get_faiss_index();
    auto computer = std::unique_ptr<faiss::DistanceComputer>(index->get_distance_computer());
    computer->set_query(q);

    // Single
    // Faiss IndexFlatIP returns the Dot Product directly (larger is better), unlike HNSWLib
    EXPECT_NEAR((*computer)(0), cpp_dot_product(q, provider.get_target(0), dim), 1e-4);

    // Batch
    float d0, d1, d2, d3;
    computer->distances_batch_4(0, 1, 2, 3, d0, d1, d2, d3);
    EXPECT_NEAR(d0, cpp_dot_product(q, provider.get_target(0), dim), 1e-4);
    EXPECT_NEAR(d3, cpp_dot_product(q, provider.get_target(3), dim), 1e-4);
}

// --- PART 2: Google Benchmark (Performance) ---

// 1. Artea FMA Benchmarks
template <std::size_t UnrollSize>
static void BM_Artea_FMA(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    artea_simd_fma_t<UnrollSize> fma_func(provider.get_dim());
    float* q = provider.get_query();
    float* t = provider.get_target(0);

    for (auto _ : state) {
        benchmark::DoNotOptimize(fma_func(q, t));
    }
}
BENCHMARK_TEMPLATE(BM_Artea_FMA, 1)->Name("Artea_FMA_Unroll_1");
BENCHMARK_TEMPLATE(BM_Artea_FMA, 2)->Name("Artea_FMA_Unroll_2");
BENCHMARK_TEMPLATE(BM_Artea_FMA, 4)->Name("Artea_FMA_Unroll_4");

// 2. HNSWLib InnerProduct Benchmark
static void BM_HNSWLib_IP(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    hnswlib::InnerProductSpace ip_space(provider.get_dim());
    auto func = ip_space.get_dist_func();
    void* param = ip_space.get_dist_func_param();

    float* q = provider.get_query();
    float* t = provider.get_target(0);

    for (auto _ : state) {
        benchmark::DoNotOptimize(func(q, t, param));
    }
}
BENCHMARK(BM_HNSWLib_IP)->Name("HNSWLib_IP_AVX512");

// 3. Faiss Direct IP Benchmark
static void BM_Faiss_Fvec_IP(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    float* q = provider.get_query();
    float* t = provider.get_target(0);
    uint32_t dim = provider.get_dim();

    for (auto _ : state) {
        benchmark::DoNotOptimize(faiss::fvec_inner_product(q, t, dim));
    }
}
BENCHMARK(BM_Faiss_Fvec_IP)->Name("Faiss_fvec_inner_product");

// 4. Faiss DistanceComputer (Single) - IP
static void BM_Faiss_DistComp_Single_IP(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    auto computer = std::unique_ptr<faiss::DistanceComputer>(
        provider.get_faiss_index()->get_distance_computer()
    );
    computer->set_query(provider.get_query());

    for (auto _ : state) {
        benchmark::DoNotOptimize((*computer)(0));
    }
}
BENCHMARK(BM_Faiss_DistComp_Single_IP)->Name("Faiss_DistComp_Single_IP");

// 5. Faiss DistanceComputer (Batch 4) - IP
static void BM_Faiss_DistComp_Batch4_IP(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    auto computer = std::unique_ptr<faiss::DistanceComputer>(
        provider.get_faiss_index()->get_distance_computer()
    );
    computer->set_query(provider.get_query());

    float d0, d1, d2, d3;

    for (auto _ : state) {
        computer->distances_batch_4(0, 1, 2, 3, d0, d1, d2, d3);
        benchmark::DoNotOptimize(d0);
    }
    state.SetItemsProcessed(state.iterations() * 4);
}
BENCHMARK(BM_Faiss_DistComp_Batch4_IP)->Name("Faiss_DistComp_Batch4_IP");

// 6. Direct C++ (Baseline) - Dot Product
static void BM_Cpp_Direct_Dot(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    float* q = provider.get_query();
    float* t = provider.get_target(0);
    uint32_t dim = provider.get_dim();

    for (auto _ : state) {
        benchmark::DoNotOptimize(cpp_dot_product(q, t, dim));
    }
}
BENCHMARK(BM_Cpp_Direct_Dot)->Name("Cpp_Direct_DotProduct");

// --- Main Entry ---

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // 1. Parse Arguments
    argparse::ArgumentParser program("test_simd_fma");

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
    logger.info("Initializing Data Provider from Dataset for FMA Tests...");
    try {
        DataProvider::instance().init();
    } catch (const std::exception& e) {
        logger.error(fmt::format("Failed to initialize dataset: {}", e.what()));
        return 1;
    }

    // 3. Run Google Test (Correctness Check)
    logger.info("==========================================================");
    logger.info(" -> Running FMA Correctness Tests (GTest)...");
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
    logger.info(" -> Running FMA Performance Benchmarks (Google Benchmark)...");
    logger.info("==========================================================");
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

    return 0;
}