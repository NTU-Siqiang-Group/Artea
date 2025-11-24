/*
 * @FilePath: /Artea/tests/test_simd_distance.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-22 16:30:00
 * @Date: 2025-10-23 19:14:32
 * @Description: Comprehensive benchmark comparing Artea SIMD distance with different
 *               unroll factors, Faiss direct functions, Faiss DistanceComputer,
 *               and HNSWLib AVX512 implementation.
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>
#include <memory>
#include <array>

#include <fmt/format.h>

// Artea headers
#include <artea/cpu/simd_distance.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/definitions.hpp>
#include <artea/logger.hpp>

// Faiss headers
#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/distances.h>

// HNSWLib headers
#include <hnswlib/hnswlib.h>

// Helper function to generate a vector with random data
void generate_random_vector(float* vec, const std::size_t dim) {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
}

// Helper function for direct C++ L2 square distance calculation
float cpp_L2sqr(const float* x, const float* y, const std::size_t d) {
    float res = 0.0f;
    for (std::size_t i = 0; i < d; ++i) {
        const float diff = x[i] - y[i];
        res += diff * diff;
    }
    return res;
}

int main() {

    // --- 1. Test Parameters ---
    constexpr artea::vec_dim_t DIM = 128; // Standard dimension
    constexpr int NUM_CORRECTNESS_TESTS = 100;
    constexpr int NUM_VECTORS_FOR_BENCHMARK = 10000000; // Total distance calculations

    // Ensure benchmark count is divisible by 4 for batch testing
    static_assert(NUM_VECTORS_FOR_BENCHMARK % 4 == 0, "Benchmark count must be divisible by 4");

    artea::logger.info("Starting SIMD Distance Test Suite");
    artea::logger.info(fmt::format("Vector Dimension: {}, Total Ops: {}", DIM, NUM_VECTORS_FOR_BENCHMARK));
    artea::logger.info("-------------------------------------");

    // --- 2. Data Preparation ---
    // Query vector (1 vector)
    artea::cpu::VectorArray<int, float> query_arr(1, DIM);
    float* query_vec = query_arr.get(0);

    // Database vectors (4 vectors for batch testing)
    artea::cpu::VectorArray<int, float> db_arr(4, DIM);

    // Initialize Faiss Flat Index (Acts as the data provider for DistanceComputer)
    faiss::IndexFlatL2 faiss_index(DIM);

    // Instantiate Artea distance calculators
    artea::cpu::SIMDDistance<float, artea::cpu::DistanceMetrics::EUCLIDEAN, 1> artea_dist_unroll_1(DIM);
    artea::cpu::SIMDDistance<float, artea::cpu::DistanceMetrics::EUCLIDEAN, 2> artea_dist_unroll_2(DIM);
    artea::cpu::SIMDDistance<float, artea::cpu::DistanceMetrics::EUCLIDEAN, 4> artea_dist_unroll_4(DIM);

    // Instantiate HNSWLib Space and Function
    // We create an L2Space. The constructor of L2Space automatically selects the best SIMD implementation
    // (AVX512/AVX/SSE) based on compilation flags (USE_AVX512) and runtime capabilities.
    hnswlib::L2Space hnsw_l2_space(DIM);
    hnswlib::DISTFUNC<float> hnsw_dist_func = hnsw_l2_space.get_dist_func();
    void* hnsw_dist_param = hnsw_l2_space.get_dist_func_param();

    // --- 3. Correctness Test ---
    artea::logger.info(fmt::format("Running {} correctness tests...", NUM_CORRECTNESS_TESTS));
    const float tolerance = 1e-4f;

    for (int k = 0; k < NUM_CORRECTNESS_TESTS; ++k) {
        // Refresh data
        generate_random_vector(query_vec, DIM);
        for(int i=0; i<4; ++i) {
            generate_random_vector(db_arr.get(i), DIM);
        }

        faiss_index.reset();
        faiss_index.add(4, db_arr.get_all());

        // Ground Truth
        float d_cpp_0 = cpp_L2sqr(query_vec, db_arr.get(0), DIM);
        float d_cpp_1 = cpp_L2sqr(query_vec, db_arr.get(1), DIM);
        float d_cpp_2 = cpp_L2sqr(query_vec, db_arr.get(2), DIM);
        float d_cpp_3 = cpp_L2sqr(query_vec, db_arr.get(3), DIM);

        // A. Verify Artea & Faiss Direct
        float d_artea_u1 = artea_dist_unroll_1(query_vec, db_arr.get(0));
        float d_artea_u2 = artea_dist_unroll_2(query_vec, db_arr.get(0));
        float d_artea_u4 = artea_dist_unroll_4(query_vec, db_arr.get(0));
        float d_faiss_func = faiss::fvec_L2sqr(query_vec, db_arr.get(0), DIM);

        // HNSWLib Calculation
        float d_hnsw = hnsw_dist_func(query_vec, db_arr.get(0), hnsw_dist_param);

        assert(std::abs(d_artea_u1 - d_cpp_0) < tolerance);
        assert(std::abs(d_artea_u2 - d_cpp_0) < tolerance);
        assert(std::abs(d_artea_u4 - d_cpp_0) < tolerance);
        assert(std::abs(d_faiss_func - d_cpp_0) < tolerance);

        // Verify HNSWLib
        assert(std::abs(d_hnsw - d_cpp_0) < tolerance);

        // B. Verify Faiss DistanceComputer
        std::unique_ptr<faiss::DistanceComputer> computer(faiss_index.get_distance_computer());
        computer->set_query(query_vec);

        float d0_single = (*computer)(0);
        float d1_single = (*computer)(1);
        float d2_single = (*computer)(2);
        float d3_single = (*computer)(3);

        assert(std::abs(d0_single - d_cpp_0) < tolerance);
        assert(std::abs(d1_single - d_cpp_1) < tolerance);
        assert(std::abs(d2_single - d_cpp_2) < tolerance);
        assert(std::abs(d3_single - d_cpp_3) < tolerance);

        float d0_batch, d1_batch, d2_batch, d3_batch;
        computer->distances_batch_4(0, 1, 2, 3, d0_batch, d1_batch, d2_batch, d3_batch);

        assert(std::abs(d0_single - d0_batch) < tolerance);
    }

    artea::logger.success(fmt::format("Correctness test PASSED! (Artea, Faiss, HNSWLib verified)"));
    artea::logger.info("-------------------------------------");

    // --- 4. Performance Benchmark ---
    artea::logger.info("Running performance benchmark...");

    volatile float dummy_accumulator = 0.0f;

    // Prepare fixed data for stable benchmarking
    generate_random_vector(query_vec, DIM);
    for(int i=0; i<4; ++i) generate_random_vector(db_arr.get(i), DIM);

    faiss_index.reset();
    faiss_index.add(4, db_arr.get_all());

    float* target_ptr = db_arr.get(0);

    // 4.1 Artea (Unroll=1)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_VECTORS_FOR_BENCHMARK; ++i) {
            dummy_accumulator += artea_dist_unroll_1(query_vec, target_ptr);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> duration = end - start;
        artea::logger.info(fmt::format("  Artea (Unroll=1):           {:.4f} us/op", duration.count() / NUM_VECTORS_FOR_BENCHMARK));
    }

    // 4.2 Artea (Unroll=2)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_VECTORS_FOR_BENCHMARK; ++i) {
            dummy_accumulator += artea_dist_unroll_2(query_vec, target_ptr);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> duration = end - start;
        artea::logger.info(fmt::format("  Artea (Unroll=2):           {:.4f} us/op", duration.count() / NUM_VECTORS_FOR_BENCHMARK));
    }

    // 4.3 Artea (Unroll=4)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_VECTORS_FOR_BENCHMARK; ++i) {
            dummy_accumulator += artea_dist_unroll_4(query_vec, target_ptr);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> duration = end - start;
        artea::logger.info(fmt::format("  Artea (Unroll=4):           {:.4f} us/op", duration.count() / NUM_VECTORS_FOR_BENCHMARK));
    }

    // 4.4 HNSWLib (AVX512 - Auto Selected)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_VECTORS_FOR_BENCHMARK; ++i) {
            dummy_accumulator += hnsw_dist_func(query_vec, target_ptr, hnsw_dist_param);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> duration = end - start;
        artea::logger.info(fmt::format("  HNSWLib (AVX512):           {:.4f} us/op", duration.count() / NUM_VECTORS_FOR_BENCHMARK));
    }

    // 4.5 Faiss (fvec_L2sqr)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_VECTORS_FOR_BENCHMARK; ++i) {
            dummy_accumulator += faiss::fvec_L2sqr(query_vec, target_ptr, DIM);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> duration = end - start;
        artea::logger.info(fmt::format("  Faiss (fvec_L2sqr):         {:.4f} us/op", duration.count() / NUM_VECTORS_FOR_BENCHMARK));
    }

    // 4.6 Faiss (DistanceComputer op())
    {
        std::unique_ptr<faiss::DistanceComputer> computer(faiss_index.get_distance_computer());
        computer->set_query(query_vec);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_VECTORS_FOR_BENCHMARK; ++i) {
            dummy_accumulator += (*computer)(0);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> duration = end - start;
        artea::logger.info(fmt::format("  Faiss (DistComp::op()):     {:.4f} us/op", duration.count() / NUM_VECTORS_FOR_BENCHMARK));
    }

    // 4.7 Faiss (DistanceComputer batch4)
    {
        std::unique_ptr<faiss::DistanceComputer> computer(faiss_index.get_distance_computer());
        computer->set_query(query_vec);

        int loop_count = NUM_VECTORS_FOR_BENCHMARK / 4;
        float d0, d1, d2, d3;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < loop_count; ++i) {
            computer->distances_batch_4(0, 1, 2, 3, d0, d1, d2, d3);
            dummy_accumulator += (d0 + d1 + d2 + d3);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> duration = end - start;
        artea::logger.info(fmt::format("  Faiss (DistComp::batch4):   {:.4f} us/op", duration.count() / NUM_VECTORS_FOR_BENCHMARK));
    }

    // 4.8 Direct C++
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_VECTORS_FOR_BENCHMARK; ++i) {
            dummy_accumulator += cpp_L2sqr(query_vec, target_ptr, DIM);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> duration = end - start;
        artea::logger.info(fmt::format("  Direct C++ (Compiler SIMD): {:.4f} us/op", duration.count() / NUM_VECTORS_FOR_BENCHMARK));
    }

    artea::logger.info("-------------------------------------");

    return 0;
}