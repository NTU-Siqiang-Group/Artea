/*
 * @FilePath: /Artea/tests/testSimdDistance.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-10-23 19:48:38
 * @Date: 2025-10-23 19:14:32
 * @Description: Test the SIMDDistance class
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cmath>
#include <immintrin.h> // For _mm_malloc and _mm_free

// Artea headers
#include <artea/cpu/simd_distance.hpp>
#include <artea/types.hpp>

// Faiss header for CPU distance functions
#include <faiss/utils/distances.h>

// Helper function to create aligned memory for SIMD
template<typename T>
T* aligned_alloc(size_t size) {
    return static_cast<T*>(_mm_malloc(size * sizeof(T), 64)); // 64-byte alignment for AVX512
}

// Helper function to generate a vector with random data
void generate_random_vector(float* vec, size_t dim) {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < dim; ++i) {
        vec[i] = dist(rng);
    }
}

int main() {
    // --- 1. Test Parameters ---
    constexpr artea::vec_dim_t DIM = 128; // Must be a multiple of 16 for AVX512 with float
    constexpr int NUM_VECTORS_FOR_BENCHMARK = 10000000;

    std::cout << "Starting SIMD Distance Test" << std::endl;
    std::cout << "Vector Dimension: " << DIM << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    // --- 2. Data Preparation ---
    float* vec1 = aligned_alloc<float>(DIM);
    float* vec2 = aligned_alloc<float>(DIM);
    generate_random_vector(vec1, DIM);
    generate_random_vector(vec2, DIM);

    // --- 3. Correctness Test ---
    std::cout << "Running correctness test..." << std::endl;
    
    // Instantiate our distance calculator
    artea::cpu::SIMDDistance<float, artea::cpu::SIMDDistanceType::EUCLIDEAN> artea_dist_calculator(DIM);

    // Calculate distance using our implementation
    float artea_distance = artea_dist_calculator(vec1, vec2);

    // Calculate distance using Faiss's implementation
    // fvec_L2sqr computes the squared L2 (Euclidean) distance for floats
    float faiss_distance = faiss::fvec_L2sqr(vec1, vec2, DIM);

    // Compare results with a small tolerance for floating point inaccuracies
    const float tolerance = 1e-4f;
    assert(std::abs(artea_distance - faiss_distance) < tolerance);
    
    std::cout << "Correctness test PASSED!" << std::endl;
    std::cout << "  Artea Result: " << artea_distance << std::endl;
    std::cout << "  Faiss Result: " << faiss_distance << std::endl;
    std::cout << "-------------------------------------" << std::endl;


    // --- 4. Performance Benchmark ---
    std::cout << "Running performance benchmark..." << std::endl;
    std::cout << "Calculating " << NUM_VECTORS_FOR_BENCHMARK << " distances..." << std::endl;

    volatile float dummy_result = 0.0f; // Use volatile to prevent compiler from optimizing away the loop

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

    std::cout << "Performance Results:" << std::endl;
    std::cout << "  Artea (SIMDDistance): " << artea_time_per_dist << " microseconds per calculation." << std::endl;
    std::cout << "  Faiss (fvec_L2sqr): " << faiss_time_per_dist << " microseconds per calculation." << std::endl;
    std::cout << "-------------------------------------" << std::endl;


    // --- 5. Cleanup ---
    _mm_free(vec1);
    _mm_free(vec2);

    return 0;
}