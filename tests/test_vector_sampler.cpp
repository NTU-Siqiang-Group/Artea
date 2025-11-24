/*
 * @FilePath: /Artea/tests/test_vector_sampler.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-22 14:24:02
 * @Date: 2025-11-22 11:36:59
 * @Description: Test suite for the VectorSampler class in Artea.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <random>
#include <cstring>

#include <artea/cpu/vector_sampler.hpp>
#include <artea/cpu/vector_array.hpp> // Assuming this exists and works with the sampler
#include <artea/logger.hpp>

// Helper function to verify if a vector exists in the source data
// This is a naive O(N*D) check, used only for small quality checks
template <typename T>
bool vector_exists_in_source(const T* query_vec, const T* source_data, size_t num_source, size_t dim) {
    for (size_t i = 0; i < num_source; ++i) {
        const T* target = source_data + i * dim;
        bool match = true;
        for (size_t d = 0; d < dim; ++d) {
            if (query_vec[d] != target[d]) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

int main() {
    // Define types for the test
    using vertex_num_t = uint32_t;
    using vec_ele_t = float;

    artea::logger.info("==========================================================");
    artea::logger.info("      Starting Vector Sampler Test Suite");
    artea::logger.info("==========================================================");

    // ====================================================================
    // Part 1: Quality & Correctness Check
    // ====================================================================
    {
        const vertex_num_t num_vecs = 1000;
        const uint32_t dim = 4;
        const float sampling_ratio = 0.1f; // Expect 100 vectors

        artea::logger.info(fmt::format("\n[PART 1] Quality Check: Source [{} x {}], Ratio {:.2f}", num_vecs, dim, sampling_ratio));

        // Create and populate a mock source array
        // We fill it with deterministic data: vector i = {i, i, i, i}
        artea::cpu::VectorArray<vertex_num_t, vec_ele_t> source_arr(num_vecs, dim);
        vec_ele_t* raw_data = source_arr.get_all();
        for (vertex_num_t i = 0; i < num_vecs; ++i) {
            for (uint32_t d = 0; d < dim; ++d) {
                raw_data[i * dim + d] = static_cast<vec_ele_t>(i);
            }
        }

        // Perform sampling
        artea::cpu::VectorSampler<vertex_num_t, vec_ele_t> sampler;
        auto sampled_arr = sampler(source_arr, sampling_ratio);

        // Verification 1: Check output dimensions
        vertex_num_t expected_count = static_cast<vertex_num_t>(num_vecs * sampling_ratio);
        if (sampled_arr.get_num_vecs() == expected_count && sampled_arr.get_vec_dim() == dim) {
            artea::logger.success(fmt::format("    -> Dimension Check Passed: Got {} vectors of dim {}", sampled_arr.get_num_vecs(), sampled_arr.get_vec_dim()));
        } else {
            artea::logger.error(fmt::format("    -> Dimension Check Failed: Expected {} x {}, Got {} x {}",
                expected_count, dim, sampled_arr.get_num_vecs(), sampled_arr.get_vec_dim()));
        }

        // Verification 2: Check data integrity (sampled vectors must exist in source)
        // We check the first few sampled vectors
        size_t check_count = std::min((size_t)10, (size_t)sampled_arr.get_num_vecs());
        bool integrity_passed = true;
        const vec_ele_t* sampled_data = sampled_arr.get_all();
        const vec_ele_t* source_data = source_arr.get_all();

        for (size_t i = 0; i < check_count; ++i) {
            if (!vector_exists_in_source(sampled_data + i * dim, source_data, num_vecs, dim)) {
                integrity_passed = false;
                artea::logger.error(fmt::format("    -> Integrity Failed: Sampled vector {} not found in source!", i));
                break;
            }
        }

        if (integrity_passed) {
            artea::logger.success("    -> Data Integrity Check Passed (Sampled vectors exist in source)");
        }
    }

    // ====================================================================
    // Part 2: Performance Benchmark
    // ====================================================================
    {
        const vertex_num_t num_vecs = 1000000; // 1 Million vectors
        const uint32_t dim = 128;
        const float sampling_ratio = 0.1f;     // Target: 100k vectors

        artea::logger.success("\n==========================================================");
        artea::logger.info(fmt::format("[PART 2] Performance Benchmark: Source [{} x {}], Ratio {:.2f}", num_vecs, dim, sampling_ratio));
        artea::logger.success("==========================================================");

        // Initialize large source array with random data
        artea::logger.info("    -> Initializing source data (this might take a moment)...");
        artea::cpu::VectorArray<vertex_num_t, vec_ele_t> source_arr(num_vecs, dim);
        // Note: In a real benchmark, we might not need to fill it with meaningful data,
        // but touching memory ensures pages are allocated.
        vec_ele_t* raw_data = source_arr.get_all();
        // Simple fill to avoid page faults dominating the benchmark
        std::fill(raw_data, raw_data + (size_t)num_vecs * dim, 1.0f);

        artea::cpu::VectorSampler<vertex_num_t, vec_ele_t> sampler;

        // Measure sampling time
        artea::logger.info("    -> Running parallel sampling...");
        auto start = std::chrono::high_resolution_clock::now();

        auto sampled_arr = sampler(source_arr, sampling_ratio);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms = end - start;

        artea::logger.success(fmt::format("    -> Sampling Finished."));
        artea::logger.success(fmt::format("    -> Output Size: {} vectors", sampled_arr.get_num_vecs()));
        artea::logger.success(fmt::format("    -> Time taken:  {:.4f} ms", duration_ms.count()));

        double throughput = (sampled_arr.get_num_vecs() * dim * sizeof(vec_ele_t)) / (duration_ms.count() / 1000.0) / 1024.0 / 1024.0;
        artea::logger.info(fmt::format("    -> Approx Throughput (Write): {:.2f} MB/s", throughput));
    }

    artea::logger.info("\nTest suite finished.");
    return 0;
}