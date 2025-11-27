/*
 * @FilePath: /Artea/tests/test_recom_nn.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-27 11:53:56
 * @Date: 2025-11-15 15:14:18
 * @Description:
 */

#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

#include <fmt/format.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/common/logger.hpp>
#include <artea/definitions.hpp>
#include <artea/cpu/utils/allocator.hpp>
#include <artea/cpu/propagation/concurrent_recom_nn.hpp>
#include <artea/cpu/propagation/locked_recom_nn.hpp>

// Define common types for the test
using vertex_num_t = uint32_t;
using vec_ele_t = float;
using nbr_t = artea::Neighbor<vertex_num_t, vec_ele_t>;
using nbr_arr_t = std::vector<nbr_t>;

// --- Benchmark Configuration ---
/**
 * @brief Total number of vertices in the test graph.
 */
constexpr vertex_num_t NUM_VERTICES = 1000000;
/**
 * @brief Number of neighbor edges to generate and append for each vertex.
 */
constexpr int EDGES_PER_VERTEX = 64;
/**
 * @brief The capacity of the recommendation buffer for each vertex.
 */
constexpr vertex_num_t RECOM_BUF_SIZE = EDGES_PER_VERTEX * 4;
// ---

// --- Start of Serial Reference Implementation ---

/**
 * @brief A simple, single-threaded reference implementation for correctness checking.
 */
class SerialRecomNN {

public:
    SerialRecomNN(vertex_num_t num_vertices, vertex_num_t recom_buf_size) {
        _recom_buf.resize(num_vertices);
        for (auto& vec : _recom_buf) {
            vec.reserve(recom_buf_size);
        }
    }

    auto append_edge(vertex_num_t src, const nbr_t& nbr) -> void {
        _recom_buf[src].push_back(nbr);
    }

    auto flush() -> void {
        for (auto& vec : _recom_buf) {
            std::sort(vec.begin(), vec.end(), artea::NeighborComparator<vertex_num_t, vec_ele_t>);
        }
    }

    auto get_recom_nbrs(vertex_num_t src) -> const nbr_arr_t& {
        return _recom_buf[src];
    }

    auto get_all_data() -> const std::vector<nbr_arr_t>& {
        return _recom_buf;
    }

private:

    std::vector<nbr_arr_t> _recom_buf;
};

// --- End of Serial Reference Implementation ---


// --- Start of Test Utilities ---

/**
 * @brief Generates random edge data for testing.
 * @param num_vertices Total number of vertices in the graph.
 * @param edges_per_vertex Number of edges to generate for each vertex.
 * @return A vector of tuples, where each tuple is {src, {dest, dist}}.
 */
auto generate_test_data(vertex_num_t num_vertices, int edges_per_vertex)
    -> std::vector<std::pair<vertex_num_t, nbr_t>>
{
    artea::logger.info(fmt::format("Generating test data: {} vertices, {} edges per vertex...", num_vertices, edges_per_vertex));
    std::vector<std::pair<vertex_num_t, nbr_t>> data;
    data.reserve(static_cast<size_t>(num_vertices) * edges_per_vertex);

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<vertex_num_t> vid_distribution(0, num_vertices - 1);
    std::uniform_real_distribution<vec_ele_t> distance_distribution(0.0f, 1.0f);

    for (vertex_num_t i = 0; i < num_vertices * edges_per_vertex; ++i) {
        vertex_num_t src = vid_distribution(rng);
        vertex_num_t dest = vid_distribution(rng);
        vec_ele_t distance = distance_distribution(rng);
        data.emplace_back(src, nbr_t{dest, distance});
    }
    artea::logger.success("Test data generated.");
    return data;
}

/**
 * @brief Checks if the results from a test implementation match the serial reference.
 * @param serial_data The ground truth data from SerialRecomNN.
 * @param test_impl A pointer to the concurrent implementation instance after flush.
 * @return True if results match, false otherwise.
 */
template<typename T>
bool run_correctness_check(
    const std::vector<nbr_arr_t>& serial_data,
    T* test_impl
) {
    artea::logger.info("Running correctness check...");
    for (vertex_num_t i = 0; i < serial_data.size(); ++i) {
        const auto& serial_nbrs = serial_data[i];
        const auto& test_nbrs = test_impl->get_recom_nbrs(i);

        if (serial_nbrs.size() != test_nbrs.size()) {
            artea::logger.error(fmt::format(
                "Correctness FAILED: Mismatch in neighbor count for vertex {}. Expected {}, got {}.",
                i, serial_nbrs.size(), test_nbrs.size()
            ));
            return false;
        }

        for (size_t j = 0; j < serial_nbrs.size(); ++j) {
            if (serial_nbrs[j].dest != test_nbrs[j].dest ||
                std::abs(serial_nbrs[j].distance - test_nbrs[j].distance) > 1e-6f) {
                artea::logger.error(fmt::format(
                    "Correctness FAILED: Mismatch in neighbor data for vertex {} at index {}.", i, j
                ));
                return false;
            }
        }
    }

    artea::logger.success("Correctness check PASSED.");
    return true;
}


/**
 * @brief Runs the full benchmark suite for a given RecommendedNN implementation.
 * @tparam RecomNN_T The type of the class to test (e.g., ConcurrentRecomNN).
 * @param class_name A string name for logging purposes.
 * @param num_vertices The number of vertices.
 * @param recom_buf_size The buffer size for neighbors.
 * @param test_data The workload data.
 * @param serial_data The ground truth for correctness check.
 */
template<typename RecomNN_T>
void run_benchmark_for(
    const std::string& class_name,
    vertex_num_t num_vertices,
    vertex_num_t recom_buf_size,
    const std::vector<std::pair<vertex_num_t, nbr_t>>& test_data,
    const std::vector<nbr_arr_t>& serial_data
) {
    artea::logger.info(fmt::format("\n--- Starting Benchmark for {} ---", class_name));

    RecomNN_T recom_nn(num_vertices, recom_buf_size);
    std::chrono::high_resolution_clock::time_point start_time, end_time;

    // 1. Test parallel append_edge
    start_time = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, test_data.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                recom_nn.append_edge(test_data[i].first, test_data[i].second);
            }
        }
    );
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> append_us = end_time - start_time;
    artea::logger.info(fmt::format("1. Parallel append_edge took: {:.5f} us per edge", append_us.count() / test_data.size()));

    // 2. Test flush
    start_time = std::chrono::high_resolution_clock::now();
    recom_nn.flush();
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> flush_us = end_time - start_time;
    artea::logger.info(fmt::format("2. flush_impl took: {:.5f} us per edge", flush_us.count() / test_data.size()));

    // Correctness check after flush
    run_correctness_check(serial_data, &recom_nn);

    // 3. Test get_recom_nbrs
    start_time = std::chrono::high_resolution_clock::now();
    volatile size_t total_neighbors = 0; // Use volatile to prevent optimization
    for (vertex_num_t i = 0; i < num_vertices; ++i) {
        total_neighbors += recom_nn.get_recom_nbrs(i).size();
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> get_us = end_time - start_time;
    artea::logger.info(fmt::format("3. get_recom_nbrs_impl took: {:.5f} us per edge", get_us.count() / total_neighbors));

    // 4. Test clear
    start_time = std::chrono::high_resolution_clock::now();
    recom_nn.clear();
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> clear_us = end_time - start_time;
    artea::logger.info(fmt::format("4. clear_impl took: {:.5f} us", clear_us.count()));
    artea::logger.success(fmt::format("--- Benchmark for {} Finished ---", class_name));
}


// --- End of Test Utilities ---

int main() {

    artea::logger.info("Starting RecommendedNN correctness and performance test.");

    auto test_data = generate_test_data(NUM_VERTICES, EDGES_PER_VERTEX);

    // --- Run Serial Reference Implementation to get ground truth ---
    artea::logger.info("\n--- Running Serial Implementation for Ground Truth ---");
    SerialRecomNN serial_nn(NUM_VERTICES, RECOM_BUF_SIZE);
    for (const auto& edge : test_data) {
        serial_nn.append_edge(edge.first, edge.second);
    }
    serial_nn.flush();
    const auto& serial_data = serial_nn.get_all_data();
    artea::logger.success("Serial ground truth generated successfully.");
    // ---

    // Run benchmark for ConcurrentRecomNN
    run_benchmark_for<artea::cpu::ConcurrentRecomNN<vertex_num_t, vec_ele_t>>(
        "ConcurrentRecomNN (TBB)",
        NUM_VERTICES,
        RECOM_BUF_SIZE,
        test_data,
        serial_data
    );

    // Run benchmark for LockedRecomNN
    run_benchmark_for<artea::cpu::LockedRecomNN<vertex_num_t, vec_ele_t>>(
        "LockedRecomNN (Mutex)",
        NUM_VERTICES,
        RECOM_BUF_SIZE,
        test_data,
        serial_data
    );

    artea::logger.success("\nAll tests completed!");
    return 0;
}