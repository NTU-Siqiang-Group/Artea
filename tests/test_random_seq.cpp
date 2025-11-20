// tests/test_random_seq.cpp

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <random>

#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/random_seq.hpp>
#include <artea/cpu/array.hpp>
#include <artea/logger.hpp>

// Helper function to calculate and print statistics using the artea::logger
template<typename T>
void print_stats(const std::vector<T>& data, const T num_vecs, const std::string& method_name) {
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

    long double N = static_cast<long double>(num_vecs);
    long double theoretical_mean = (N - 1.0) / 2.0;
    long double theoretical_variance = (N * N - 1.0) / 12.0;

    artea::logger.info(fmt::format("--- Statistics for {} ---", method_name));
    artea::logger.info(fmt::format("    - Generated Mean:         {:.4f} (Theoretical: {:.4f})", mean, theoretical_mean));
    artea::logger.info(fmt::format("    - Generated Variance:     {:.4f} (Theoretical: {:.4f})", variance, theoretical_variance));
    artea::logger.info(fmt::format("    - Generated Std Deviation: {:.4f}", std::sqrt(variance)));
}

int main() {
    using vec_id_t = uint32_t;

    // --- Parameters Definition ---
    // Part 1: Parameters for the quality check (smaller sample size)
    const uint32_t quality_num_vecs = 1024;
    const uint32_t quality_num_rand_nbrs = 1000000; // 1 million

    // Part 2: Parameters for the performance benchmark (larger sample size)
    const uint32_t perf_num_vecs = 1024;
    const uint32_t perf_num_rand_nbrs = 100000000; // 100 million

    artea::logger.info("==========================================================");
    artea::logger.info("      Starting Random Number Generator Test Suite");
    artea::logger.info("==========================================================");

    // ====================================================================
    // Part 1: Quality & Correctness Check
    // ====================================================================
    artea::logger.info(fmt::format("\n[PART 1] Quality Check: Generating {} numbers in [0, {}]", quality_num_rand_nbrs, quality_num_vecs - 1));

    artea::cpu::RandomSeq<uint32_t> mkl_rng_quality(quality_num_vecs);

    // C++ Standard Library (Serial)
    {
        artea::logger.info("\n -> Running Quality Check for C++ Standard Library (Serial)...");
        std::vector<vec_id_t> random_data(quality_num_rand_nbrs);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<vec_id_t> distrib(0, quality_num_vecs - 1);
        for (uint32_t i = 0; i < quality_num_rand_nbrs; ++i) random_data[i] = distrib(gen);
        print_stats(random_data, quality_num_vecs, "C++ Standard Library (Serial)");
    }

    // MKL Single-Threaded
    {
        artea::logger.info("\n -> Running Quality Check for Single-Threaded Vectorized (MKL)...");
        std::vector<vec_id_t> random_data(quality_num_rand_nbrs);
        mkl_rng_quality.generate(random_data, quality_num_rand_nbrs);
        print_stats(random_data, quality_num_vecs, "Single-Threaded Vectorized (MKL)");
    }

    // MKL Multi-Threaded
    {
        size_t num_threads = std::thread::hardware_concurrency();
        artea::logger.info(fmt::format("\n -> Running Quality Check for Multi-Threaded Vectorized (MKL) with {} threads...", num_threads));
        std::vector<vec_id_t> random_data(quality_num_rand_nbrs);
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, quality_num_rand_nbrs),
            [&](const tbb::blocked_range<uint32_t>& r) {
                std::vector<vec_id_t> local_data(r.size());
                mkl_rng_quality.generate(local_data, r.size());
                std::copy(local_data.data(), local_data.data() + r.size(), random_data.data() + r.begin());
            });
        print_stats(random_data, quality_num_vecs, "Multi-Threaded Vectorized (MKL)");
    }

    // ====================================================================
    // Part 2: Performance Benchmark
    // ====================================================================
    artea::logger.success("\n==========================================================");
    artea::logger.info(fmt::format("[PART 2] Performance Benchmark: Generating {} numbers in [0, {}]", perf_num_rand_nbrs, perf_num_vecs - 1));
    artea::logger.success("==========================================================");

    artea::cpu::RandomSeq<uint32_t> mkl_rng_perf(perf_num_vecs);

    // C++ Standard Library (Serial)
    {
        artea::logger.info("\n[0] Benchmarking C++ Standard Library (Serial)...");
        std::vector<vec_id_t> random_data(perf_num_rand_nbrs);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<vec_id_t> distrib(0, perf_num_vecs - 1);
        auto start = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < perf_num_rand_nbrs; ++i) random_data[i] = distrib(gen);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = end - start;
        artea::logger.success(fmt::format("    -> Time taken: {:.4f} ms", duration_ms.count()));
    }

    // MKL Single-Threaded
    {
        artea::logger.info("\n[1] Benchmarking Single-Threaded Vectorized (Intel MKL)...");
        std::vector<vec_id_t> random_data(perf_num_rand_nbrs);
        auto start = std::chrono::high_resolution_clock::now();
        mkl_rng_perf.generate(random_data, perf_num_rand_nbrs);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = end - start;
        artea::logger.success(fmt::format("    -> Time taken: {:.4f} ms", duration_ms.count()));
    }

    // MKL Multi-Threaded (Pure Generation)
    {
        size_t num_threads = std::thread::hardware_concurrency();
        artea::logger.info(fmt::format("\n[2] Benchmarking Multi-Threaded Vectorized (Intel MKL) with {} threads...", num_threads));

        auto start = std::chrono::high_resolution_clock::now();
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, perf_num_rand_nbrs),
            [&](const tbb::blocked_range<uint32_t>& r) {
                // Each thread creates its own buffer to store the results.
                std::vector<vec_id_t> local_data(r.size());

                // Perform the generation.
                mkl_rng_perf.generate(local_data, r.size());

                // For pure performance testing, we DO NOT copy the results back.
                // The `local_data` buffer is discarded when the lambda scope ends.
            });
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = end - start;
        artea::logger.success(fmt::format("    -> Time taken: {:.4f} ms", duration_ms.count()));
    }

    artea::logger.info("\nTest suite finished.");
    return 0;
}