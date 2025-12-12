// Copyright 2025 Weitang Ye
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
 * @FilePath: /Artea/tests/test_neighbor_search.cpp
 * @Author: Chandler (Weitang Ye)
 * @Description: Benchmark comparing Branching vs Branchless Binary Search on Neighbor arrays.
 */

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>

#include <omp.h>
#include <argparse/argparse.hpp>

#include <artea/cpu/index/neighbor.hpp>
#include <artea/cpu/utils/array_search.hpp>
#include <artea/cpu/containers/allocator.hpp>
#include <artea/common/logger.hpp>
#include <artea/definitions.hpp>

using namespace artea;
using namespace artea::cpu;

// Define types for convenience
using nbr_t = Neighbor<uint32_t, float>;
using nbr_container_t = cache_aligned_container_t<nbr_t>;
using workload_container_t = std::vector<nbr_container_t>;

/**
 * @brief Standard Binary Search with Branching (The "Bad" implementation for comparison).
 *        This mimics the logic before your branchless optimization.
 */
template <typename T, typename container_t, typename array_index_t, typename compare_func_t>
auto branching_binary_search(
    const container_t& arr,
    const T& target,
    const compare_func_t compare_func
) -> array_index_t {
    array_index_t left = 0;
    array_index_t right = static_cast<array_index_t>(arr.size());

    // Standard branching logic which suffers from misprediction
    while (left < right) {
        array_index_t mid = left + (right - left) / 2;
        if (compare_func(arr[mid], target)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    if (left < static_cast<array_index_t>(arr.size()) && !compare_func(target, arr[left])) {
        return left;
    }
    return invalid_vertex_id<array_index_t>();
}

/**
 * @brief Clears the CPU cache by writing to a large dummy vector.
 *        Based on lscpu, L3 is 48MB, so 1GB is more than sufficient.
 */
void clear_cpu_cache() {
    artea::logger.debug("Clearing CPU cache...");
    constexpr size_t SIZE = 1024 * 1024 * 256; // 256M ints = 1GB
    // Use volatile to prevent compiler optimization
    std::vector<int> dummy(SIZE);

    // Write to memory to force cache lines to be populated with junk
    for (size_t i = 0; i < SIZE; i += 16) { // Stride to touch different cache lines
        dummy[i] = static_cast<int>(i);
    }

    // Read back to ensure execution
    volatile int sink = 0;
    for (size_t i = 0; i < SIZE; i += 16) {
        sink += dummy[i];
    }
    (void)sink; // Suppress unused warning
}

int main(int argc, char *argv[]) {
    // 1. Setup Argument Parser
    argparse::ArgumentParser program("neighbor_search_benchmark");

    program.add_argument("--len")
        .help("Length of each Neighbor array")
        .default_value(size_t(512))
        .scan<'u', size_t>();

    program.add_argument("--count")
        .help("Number of arrays to generate")
        .default_value(size_t(10000))
        .scan<'u', size_t>();

    program.add_argument("--ops")
        .help("Number of search operations per array")
        .default_value(size_t(10))
        .scan<'u', size_t>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    size_t array_len = program.get<size_t>("--len");
    size_t num_arrays = program.get<size_t>("--count");
    size_t ops_per_array = program.get<size_t>("--ops");
    size_t total_queries = num_arrays * ops_per_array;

    artea::logger.info(fmt::format("Benchmark Config: Array Length = {}, Array Count = {}, Ops/Array = {}, Total Queries = {}",
        array_len, num_arrays, ops_per_array, total_queries));

    // 2. Prepare Data (Workload)
    artea::logger.info("Generating workload data...");

    workload_container_t workload;
    workload.reserve(num_arrays);

    struct Query {
        nbr_t target;
        uint32_t array_idx;
    };
    std::vector<Query> queries;
    queries.reserve(total_queries);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_val(0.0f, 1000.0f);
    std::uniform_int_distribution<uint32_t> id_dist(0, 100000);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    for (size_t i = 0; i < num_arrays; ++i) {
        nbr_container_t vec;
        vec.reserve(array_len);

        // Generate random neighbors
        for (size_t j = 0; j < array_len; ++j) {
            float d = dist_val(rng);
            vec.emplace_back(nbr_t::create_new(id_dist(rng), d));
        }

        // Sort them
        std::sort(vec.begin(), vec.end(), NeighborComparator<uint32_t, float>);

        // Generate queries for this array
        std::uniform_int_distribution<size_t> idx_dist(0, array_len - 1);

        for (size_t k = 0; k < ops_per_array; ++k) {
            nbr_t target_val = {0, 0.0f, true};

            // 50% chance to search for an existing element
            // 50% chance to search for a likely non-existing element
            if (prob_dist(rng) > 0.5f) {
                target_val = vec[idx_dist(rng)];
            } else {
                // Generate a random distance. It might exist by chance, but unlikely to match ID too if checked strictly.
                // To ensure "not found" behavior more reliably for binary search, we can pick a value larger than max.
                // But mixing random values is more realistic for "miss" cases in index search.
                float random_d = dist_val(rng);
                // Occasionally pick a value larger than max to force full traversal
                if (prob_dist(rng) > 0.8f && !vec.empty()) {
                    random_d = vec.back().get_distance() + 100.0f;
                }
                target_val = nbr_t::create_new(id_dist(rng), random_d);
            }
            queries.push_back({target_val, static_cast<uint32_t>(i)});
        }

        workload.push_back(std::move(vec));
    }

    // ---------------------------------------------------------
    // Verification: Correctness Test
    // ---------------------------------------------------------
    {
        artea::logger.info("Verifying correctness of search algorithms...");
        // Check a subset of arrays to save time
        size_t verify_count = std::min(num_arrays, size_t(100));

        for (size_t i = 0; i < verify_count; ++i) {
            const auto& vec = workload[i];
            if (vec.empty()) continue;

            // Case 1: Element Exists (Found)
            std::uniform_int_distribution<size_t> idx_dist_verify(0, vec.size() - 1);
            size_t expected_idx = idx_dist_verify(rng);
            nbr_t target_found = vec[expected_idx];

            auto res_br = branching_binary_search<nbr_t, nbr_container_t, uint32_t>(vec, target_found, NeighborComparator<uint32_t, float>);
            auto res_bl = array_search<nbr_t, nbr_container_t, uint32_t, search_method_t::BINARY_SEARCH>(vec, target_found, NeighborComparator<uint32_t, float>);

            auto is_equal = [&](const nbr_t& a, const nbr_t& b) {
                return !NeighborComparator<uint32_t, float>(a, b) && !NeighborComparator<uint32_t, float>(b, a);
            };

            if (res_br == invalid_vertex_id<uint32_t>() || !is_equal(vec[res_br], target_found)) {
                artea::logger.error(fmt::format("Verification FAILED (Found Case): Branching search failed on array {}", i));
                return 1;
            }
            if (res_bl == invalid_vertex_id<uint32_t>() || !is_equal(vec[res_bl], target_found)) {
                artea::logger.error(fmt::format("Verification FAILED (Found Case): Branchless search failed on array {}", i));
                return 1;
            }

            // Case 2: Element Does Not Exist (Not Found)
            // Create a target with distance larger than the max + offset
            nbr_t target_missing = nbr_t::create_new(0, vec.back().get_distance() + 5000.0f);

            auto res_br_miss = branching_binary_search<nbr_t, nbr_container_t, uint32_t>(vec, target_missing, NeighborComparator<uint32_t, float>);
            auto res_bl_miss = array_search<nbr_t, nbr_container_t, uint32_t, search_method_t::BINARY_SEARCH>(vec, target_missing, NeighborComparator<uint32_t, float>);

            if (res_br_miss != invalid_vertex_id<uint32_t>()) {
                artea::logger.error(fmt::format("Verification FAILED (Not Found Case): Branching search returned index {} for missing element.", res_br_miss));
                return 1;
            }
            if (res_bl_miss != invalid_vertex_id<uint32_t>()) {
                artea::logger.error(fmt::format("Verification FAILED (Not Found Case): Branchless search returned index {} for missing element.", res_bl_miss));
                return 1;
            }
        }
        artea::logger.success("Verification Passed: Both algorithms behave correctly.");
    }

    // ---------------------------------------------------------
    // Test 1: Standard Branching Binary Search (Baseline)
    // ---------------------------------------------------------
    {
        clear_cpu_cache();
        artea::logger.info("Starting Branching Binary Search (Baseline)...");

        auto start = std::chrono::high_resolution_clock::now();

        size_t found_count = 0;
        volatile size_t check_sum = 0;

        for (const auto& q : queries) {
            const auto& arr = workload[q.array_idx];
            auto idx = branching_binary_search<nbr_t, nbr_container_t, uint32_t>(arr, q.target, NeighborComparator<uint32_t, float>);

            if (idx != invalid_vertex_id<uint32_t>()) {
                found_count++;
                check_sum += idx;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        artea::logger.success(fmt::format("Branching Search: Total Time = {:.2f} ms", elapsed.count()));
        artea::logger.success(fmt::format("Branching Search: Average Latency = {:.4f} ns", (elapsed.count() * 1e6) / total_queries));
        artea::logger.debug(fmt::format("Branching Found Count: {} / {}", found_count, total_queries));
    }

    // ---------------------------------------------------------
    // Test 2: Artea Branchless Binary Search (Optimized)
    // ---------------------------------------------------------
    {
        clear_cpu_cache();
        artea::logger.info("Starting Artea Branchless Binary Search...");

        auto start = std::chrono::high_resolution_clock::now();

        size_t found_count = 0;
        volatile size_t check_sum = 0;

        for (const auto& q : queries) {
            const auto& arr = workload[q.array_idx];
            // Use the library function
            auto idx = array_search<nbr_t, nbr_container_t, uint32_t, search_method_t::BINARY_SEARCH>(arr, q.target, NeighborComparator<uint32_t, float>);

            if (idx != invalid_vertex_id<uint32_t>()) {
                found_count++;
                check_sum += idx;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        artea::logger.success(fmt::format("Branchless Search: Total Time = {:.2f} ms", elapsed.count()));
        artea::logger.success(fmt::format("Branchless Search: Average Latency = {:.4f} ns", (elapsed.count() * 1e6) / total_queries));
        artea::logger.debug(fmt::format("Branchless Found Count: {} / {}", found_count, total_queries));
    }

    // ---------------------------------------------------------
    // Test 3: Linear Search (For context)
    // ---------------------------------------------------------
    // Only run if array length is small and total ops isn't huge
    if (array_len <= 2048 && total_queries <= 10000000) {
        clear_cpu_cache();
        artea::logger.info("Starting Linear Search (For context)...");

        auto start = std::chrono::high_resolution_clock::now();

        size_t found_count = 0;
        volatile size_t check_sum = 0;

        for (const auto& q : queries) {
            const auto& arr = workload[q.array_idx];
            auto idx = array_search<nbr_t, nbr_container_t, uint32_t, search_method_t::LINEAR_SEARCH>(arr, q.target, NeighborComparator<uint32_t, float>);

            if (idx != invalid_vertex_id<uint32_t>()) {
                found_count++;
                check_sum += idx;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;

        artea::logger.success(fmt::format("Linear Search: Total Time = {:.2f} ms", elapsed.count()));
        artea::logger.success(fmt::format("Linear Search: Average Latency = {:.4f} ns", (elapsed.count() * 1e6) / total_queries));
    }

    return 0;
}