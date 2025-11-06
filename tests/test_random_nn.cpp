// tests/test_random_nn.cpp

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>
#include <iomanip>

#include "artea/cpu/random_nn.hpp"
#include "artea/cpu/array.hpp"

// A simple helper function to calculate and print statistics
template<typename T>
void print_stats(const artea::cpu::Array<T>& data, const T num_vecs, const std::string& method_name) {
    long double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i];
    }
    long double mean = sum / data.size();

    long double sq_sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        sq_sum += (static_cast<long double>(data[i]) - mean) * (static_cast<long double>(data[i]) - mean);
    }
    long double variance = sq_sum / data.size();
    
    // Theoretical values for a discrete uniform distribution [0, N-1]
    long double N = static_cast<long double>(num_vecs);
    long double theoretical_mean = (N - 1.0) / 2.0;
    long double theoretical_variance = (N * N - 1.0) / 12.0;

    std::cout << "\n--- Statistics for " << method_name << " ---" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Generated Mean:         " << mean << " (Theoretical: " << theoretical_mean << ")" << std::endl;
    std::cout << "Generated Variance:     " << variance << " (Theoretical: " << theoretical_variance << ")" << std::endl;
    std::cout << "Generated Std Deviation: " << std::sqrt(variance) << std::endl;
}

int main() {
    // Benchmark parameters
    const uint32_t num_vecs = 1024; // The upper bound for random numbers [0, 1023]
    const uint32_t num_rand_nbrs = 10000000; // Generate 10 million random numbers
    using vec_id_t = uint32_t;

    std::cout << "Starting random number generation benchmark..." << std::endl;
    std::cout << "Generating " << num_rand_nbrs << " numbers in the range [0, " << num_vecs - 1 << "]." << std::endl;

    artea::cpu::Array<vec_id_t> random_data(num_rand_nbrs);

    // --- 1. Benchmark Serial Implementation ---
    {
        artea::cpu::RandomNN<uint32_t, vec_id_t, artea::cpu::RandomGenType::SERIAL> serial_rng(num_vecs);

        std::cout << "\n[1] Benchmarking Serial (std::mt19937)..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        // Call generate directly, no more .template keyword needed
        serial_rng.generate(random_data, num_rand_nbrs);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = end - start;
        
        std::cout << "Time taken: " << duration_ms.count() << " ms" << std::endl;
        
        // Quality check
        print_stats(random_data, num_vecs, "Serial");
    }

    // --- 2. Benchmark Vectorized (MKL) Implementation ---
#ifdef MKL_ENABLED
    {
        artea::cpu::RandomNN<uint32_t, vec_id_t, artea::cpu::RandomGenType::VECTORIZED> mkl_rng(num_vecs);

        std::cout << "\n[2] Benchmarking Vectorized (Intel MKL)..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();

        // Call generate directly
        mkl_rng.generate(random_data, num_rand_nbrs);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = end - start;
        
        std::cout << "Time taken: " << duration_ms.count() << " ms" << std::endl;
        
        // Quality check
        print_stats(random_data, num_vecs, "Vectorized (MKL)");
    }
#else
    {
        std::cout << "\n[2] Vectorized (Intel MKL) benchmark skipped." << std::endl;
        std::cout << "   Reason: Compiled without Intel MKL support." << std::endl;
    }
#endif

    return 0;
}