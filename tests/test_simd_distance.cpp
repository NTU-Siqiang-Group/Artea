/*
 * @FilePath: /yeweitang/Artea/tests/test_simd_distance.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-05 12:50:21
 * @Date: 2025-10-23 19:14:32
 * @Description: Test the SIMDDistance class and compare with Faiss and direct C++.
 *               Refactored to use VectorArray for memory management and added multiple correctness tests.
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>

#include <fmt/format.h> // For fmt::format

// Artea headers
#include <artea/logger.hpp> // Added the Artea Logger
#include <artea/cpu/simd_distance.hpp>
#include <artea/cpu/vector_array.hpp> // Replaced manual memory management with VectorArray
#include <artea/types.hpp>

// Faiss header for CPU distance functions
#include <faiss/utils/distances.h>

namespace {
    // Instantiate the Artea Logger
    artea::ArteaLogger logger("SIMD_Distance_Test");
}   // anonymous namespace

// Helper function to generate a vector with random data
void generate_random_vector(float* vec, const std::size_t dim) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
}

// Helper function for direct C++ L2 square distance calculation (non-SIMD)
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
    constexpr artea::vec_dim_t DIM = 128; // Must be a multiple of 16 for AVX512 with float
    constexpr int NUM_CORRECTNESS_TESTS = 100; // Number of times to run the correctness check
    constexpr int NUM_VECTORS_FOR_BENCHMARK = 10000000;

    logger.info("Starting SIMD Distance Test");
    logger.info(fmt::format("Vector Dimension: {}", DIM));
    logger.info("-------------------------------------");

    // --- 2. Data Preparation ---
    // Use VectorArray to manage aligned memory for our two test vectors.
    artea::cpu::VectorArray<int, float> test_vectors(2, DIM);
    float* vec1 = test_vectors.get(0);
    float* vec2 = test_vectors.get(1);

    // --- 3. Correctness Test ---
    logger.info(fmt::format("Running {} correctness tests...", NUM_CORRECTNESS_TESTS));
    
    // Instantiate our distance calculator
    artea::cpu::SIMDDistance<float, artea::cpu::DistanceMetrics::EUCLIDEAN> artea_dist_calculator(DIM);
    const float tolerance = 1e-4f;

    for (int i = 0; i < NUM_CORRECTNESS_TESTS; ++i) {
        // Generate new random data for each test run
        generate_random_vector(vec1, DIM);
        generate_random_vector(vec2, DIM);

        // Calculate distance using our implementation
        float artea_distance = artea_dist_calculator(vec1, vec2);

        // Calculate distance using Faiss's implementation
        float faiss_distance = faiss::fvec_L2sqr(vec1, vec2, DIM);

        // Calculate distance using direct C++
        float cpp_distance = cpp_L2sqr(vec1, vec2, DIM);

        // Compare results with a small tolerance for floating point inaccuracies
        assert(std::abs(artea_distance - faiss_distance) < tolerance);
        assert(std::abs(faiss_distance - cpp_distance) < tolerance);
    }
    
    logger.success("Correctness test PASSED!");

    // We can print the results of the last test case as a sample
    logger.debug(fmt::format("Sample Artea Result:      {}", artea_dist_calculator(vec1, vec2)));
    logger.debug(fmt::format("Sample Faiss Result:      {}", faiss::fvec_L2sqr(vec1, vec2, DIM)));
    logger.debug(fmt::format("Sample Direct C++ Result: {}", cpp_L2sqr(vec1, vec2, DIM)));
    logger.info("-------------------------------------");

    // --- 4. Performance Benchmark ---
    logger.info("Running performance benchmark...");
    logger.info(fmt::format("Calculating {} distances...", NUM_VECTORS_FOR_BENCHMARK));

    volatile float dummy_result = 0.0f; // Use volatile to prevent compiler from optimizing away the loop

    // Use the last generated vectors for the benchmark
    generate_random_vector(vec1, DIM);
    generate_random_vector(vec2, DIM);

    // Benchmark Artea
    auto start_artea = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_VECTORS_FOR_BENCHMARK; ++i) {
        dummy_result += artea_dist_calculator(vec1, vec2);
    }
    auto end_artea = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> artea_duration = end_artea - start_artea;
    double artea_time_per_dist = artea_duration.count() / NUM_VECTORS_FOR_BENCHMARK;

    // Benchmark Faiss
    auto start_faiss = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_VECTORS_FOR_BENCHMARK; ++i) {
        dummy_result += faiss::fvec_L2sqr(vec1, vec2, DIM);
    }
    auto end_faiss = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> faiss_duration = end_faiss - start_faiss;
    double faiss_time_per_dist = faiss_duration.count() / NUM_VECTORS_FOR_BENCHMARK;

    // Benchmark Direct C++
    auto start_cpp = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_VECTORS_FOR_BENCHMARK; ++i) {
        dummy_result += cpp_L2sqr(vec1, vec2, DIM);
    }
    auto end_cpp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> cpp_duration = end_cpp - start_cpp;
    double cpp_time_per_dist = cpp_duration.count() / NUM_VECTORS_FOR_BENCHMARK;
    
    logger.info("Performance Results:");
    logger.info(fmt::format("  Artea (SIMDDistance): {:.4f} microseconds per calculation.", artea_time_per_dist));
    logger.info(fmt::format("  Faiss (fvec_L2sqr):   {:.4f} microseconds per calculation.", faiss_time_per_dist));
    logger.info(fmt::format("  Direct C++:           {:.4f} microseconds per calculation.", cpp_time_per_dist));
    logger.info("-------------------------------------");


    // --- 5. Cleanup ---
    // No manual cleanup needed. The 'test_vectors' VectorArray will automatically
    // deallocate its memory when it goes out of scope.

    return 0;
}