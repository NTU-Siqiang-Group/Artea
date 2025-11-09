#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <memory>
#include <cmath>
#include <cstring>

#include "artea/cpu/concurrent_graph.hpp"
#include "artea/cpu/baseline_graph.hpp"
#include "artea/logger.hpp"

namespace {
    artea::ArteaLogger logger("TestConcurrentGraph");
    using nbr_t = artea::cpu::IndexGraph<uint32_t, float>::nbr_t;

    struct AppendOperation {
        uint32_t src;
        nbr_t nbr;
    };
}

// Correctness Test Runner (unchanged)
void run_correctness_test() {
    logger.info("=========================================================");
    logger.info("            Running Correctness Test                     ");
    logger.info("=========================================================");

    const uint32_t num_vertices = 10000;
    const uint32_t num_nbrs_per_vertex = 32;
    const uint64_t total_insertions = 1000000;
    const uint32_t num_threads = omp_get_max_threads() > 1 ? omp_get_max_threads() : 2;

    logger.info(fmt::format("Configuration: {} vertices, {} nbrs/vertex, {} insertions, {} threads.",
        num_vertices, num_nbrs_per_vertex, total_insertions, num_threads));
    
    std::vector<AppendOperation> operations(total_insertions);
    { std::mt19937 rng(12345); std::uniform_int_distribution<uint32_t> id_dist(0, num_vertices-1); std::uniform_real_distribution<float> dist_dist(0.0f, 100.0f); for(auto& op : operations) op = {id_dist(rng), {id_dist(rng), dist_dist(rng)}}; }
    
    artea::cpu::ConcurrentGraph<uint32_t, float> concurrent_graph(num_vertices, num_nbrs_per_vertex, total_insertions / num_vertices);
    artea::cpu::BaselineGraph<uint32_t, float> baseline_graph(num_vertices, num_nbrs_per_vertex, total_insertions / num_vertices);
    
    logger.info("Applying the same operations to both graphs concurrently...");
    #pragma omp parallel for num_threads(num_threads)
    for (uint64_t i = 0; i < total_insertions; ++i) { const auto& op = operations[i]; concurrent_graph.append_nbr(op.src, op.nbr); baseline_graph.append_nbr(op.src, op.nbr); }
    
    logger.info("Merging neighbors for both graphs...");
    concurrent_graph.merge_nbrs();
    baseline_graph.merge_nbrs();

    logger.info("Verifying final states by comparing canonically sorted results...");
    bool all_match = true;
    auto canonical_comparator = [](const nbr_t& a, const nbr_t& b) { if (std::abs(a.distance - b.distance) > 1e-7) return a.distance < b.distance; return a.dest < b.dest; };

    for (uint32_t v = 0; v < num_vertices; ++v) {
        auto& concurrent_nbrs_arr = concurrent_graph.fetch_nbrs(v);
        auto& baseline_nbrs_arr = baseline_graph.fetch_nbrs(v);
        if (concurrent_nbrs_arr.get_num_element() != baseline_nbrs_arr.get_num_element()) {
            logger.error(fmt::format("Mismatch at vertex {}: Valid element count differs! ConcurrentGraph has {}, BaselineGraph has {}.", v, concurrent_nbrs_arr.get_num_element(), baseline_nbrs_arr.get_num_element()));
            all_match = false;
            break;
        }
        size_t valid_size = concurrent_nbrs_arr.get_num_element();
        if (valid_size > 0) {
            std::vector<nbr_t> concurrent_copy(concurrent_nbrs_arr.data(), concurrent_nbrs_arr.data() + valid_size);
            std::vector<nbr_t> baseline_copy(baseline_nbrs_arr.data(), baseline_nbrs_arr.data() + valid_size);
            std::sort(concurrent_copy.begin(), concurrent_copy.end(), canonical_comparator);
            std::sort(baseline_copy.begin(), baseline_copy.end(), canonical_comparator);
            for (size_t i = 0; i < valid_size; ++i) {
                if (std::abs(concurrent_copy[i].distance - baseline_copy[i].distance) > 1e-7 || concurrent_copy[i].dest != baseline_copy[i].dest) {
                    logger.error(fmt::format("Mismatch at vertex {}, neighbor {} (after canonical sort):", v, i));
                    logger.error(fmt::format("  - ConcurrentGraph: dest={}, distance={}", concurrent_copy[i].dest, concurrent_copy[i].distance));
                    logger.error(fmt::format("  - BaselineGraph:   dest={}, distance={}", baseline_copy[i].dest, baseline_copy[i].distance));
                    all_match = false;
                    break;
                }
            }
        }
        if (!all_match) break;
    }

    if (all_match) {
        logger.info("Per-vertex check passed. Verifying global average distance checksum...");
        double concurrent_total_dist = 0.0;
        uint64_t concurrent_total_count = 0;
        #pragma omp parallel for reduction(+:concurrent_total_dist, concurrent_total_count)
        for(uint32_t v = 0; v < num_vertices; ++v) {
            const auto& nbrs = concurrent_graph.fetch_nbrs(v);
            size_t count = nbrs.get_num_element();
            concurrent_total_count += count;
            for(size_t i = 0; i < count; ++i) { concurrent_total_dist += nbrs[i].distance; }
        }
        double baseline_total_dist = 0.0;
        uint64_t baseline_total_count = 0;
        #pragma omp parallel for reduction(+:baseline_total_dist, baseline_total_count)
        for(uint32_t v = 0; v < num_vertices; ++v) {
            const auto& nbrs = baseline_graph.fetch_nbrs(v);
            size_t count = nbrs.get_num_element();
            baseline_total_count += count;
            for(size_t i = 0; i < count; ++i) { baseline_total_dist += nbrs[i].distance; }
        }
        double concurrent_avg = (concurrent_total_count == 0) ? 0.0 : concurrent_total_dist / concurrent_total_count;
        double baseline_avg = (baseline_total_count == 0) ? 0.0 : baseline_total_dist / baseline_total_count;
        logger.info(fmt::format("ConcurrentGraph average distance: {:.8f} ({} total neighbors)", concurrent_avg, concurrent_total_count));
        logger.info(fmt::format("BaselineGraph average distance:   {:.8f} ({} total neighbors)", baseline_avg, baseline_total_count));
        const double epsilon = 1e-9;
        if (std::abs(concurrent_avg - baseline_avg) > epsilon || concurrent_total_count != baseline_total_count) {
            logger.error("Global checksum FAILED: Average distances or total counts do not match.");
            all_match = false;
        } else {
            logger.success("Global checksum PASSED: Average distances are consistent.");
        }
    }

    if (all_match) { logger.success("Correctness test PASSED."); } 
    else { logger.error("Correctness test FAILED."); }
}


// --- Performance Test Runner ---
void run_performance_test() {
    logger.info("=========================================================");
    logger.info("               Running Performance Test                  ");
    logger.info("=========================================================");
    
    const uint32_t num_vertices = 100000;
    const uint32_t num_nbrs_per_vertex = 64;
    const uint32_t num_threads = omp_get_max_threads();
    const uint64_t total_insertions = 1000 * num_vertices;

    logger.info(fmt::format("Configuration: {} vertices, {} nbrs/vertex, {} OpenMP threads, {} total insertions.",
        num_vertices, num_nbrs_per_vertex, num_threads, total_insertions));
    
    logger.info("Pre-generating insertion data to isolate benchmark timing...");
    std::vector<AppendOperation> operations(total_insertions);
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        std::mt19937 local_rng(std::chrono::high_resolution_clock::now().time_since_epoch().count() + thread_id);
        std::uniform_int_distribution<uint32_t> vertex_id_dist(0, num_vertices - 1);
        std::uniform_real_distribution<float> distance_dist(0.0f, 100.0f);
        #pragma omp for
        for (uint64_t i = 0; i < total_insertions; ++i) {
            operations[i] = {vertex_id_dist(local_rng), {vertex_id_dist(local_rng), distance_dist(local_rng)}};
        }
    }
    logger.info("Data generation complete.");
    
    logger.info("Phase 1: Benchmarking concurrent append operations...");
    
    auto benchmark_append = [&](auto& graph, const std::string& graph_name) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        #pragma omp parallel for num_threads(num_threads)
        for (uint64_t i = 0; i < total_insertions; ++i) {
            const auto& op = operations[i];
            graph.append_nbr(op.src, op.nbr);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        
        double total_seconds = duration.count();
        double mops = static_cast<double>(total_insertions) / total_seconds / 1'000'000.0;
        double ms_per_op = (total_seconds * 1'000'000.0) / total_insertions;

        std::string message = fmt::format(
            "{} append phase took: {:.4f} seconds.\n"
            "  -> Throughput: {:6f} M-Ops/sec\n"
            "  -> Latency:    {:6f} ms/op",
            graph_name, total_seconds, mops, ms_per_op
        );

        if (graph_name.find("ConcurrentGraph") != std::string::npos) {
            logger.success(message);
        } else {
            logger.info(message);
        }
    };
    
    artea::cpu::ConcurrentGraph<uint32_t, float> concurrent_graph(num_vertices, num_nbrs_per_vertex, total_insertions / num_vertices);
    benchmark_append(concurrent_graph, "ConcurrentGraph");
    
    artea::cpu::BaselineGraph<uint32_t, float> baseline_graph(num_vertices, num_nbrs_per_vertex, total_insertions / num_vertices);
    benchmark_append(baseline_graph, "BaselineGraph (std::mutex)");
    logger.info("---------------------------------------------------------");

    logger.info("Phase 2: Benchmarking parallel merge operations...");
    auto start_concurrent_merge = std::chrono::high_resolution_clock::now();
    concurrent_graph.merge_nbrs();
    auto end_concurrent_merge = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> concurrent_merge_duration = end_concurrent_merge - start_concurrent_merge;
    logger.success(fmt::format("ConcurrentGraph merge phase took: {:.4f} seconds.", concurrent_merge_duration.count()));

    auto start_baseline_merge = std::chrono::high_resolution_clock::now();
    baseline_graph.merge_nbrs();
    auto end_baseline_merge = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> baseline_merge_duration = end_baseline_merge - start_baseline_merge;
    logger.info(fmt::format("BaselineGraph merge phase took: {:.4f} seconds.", baseline_merge_duration.count()));
    logger.info("---------------------------------------------------------");
    logger.success("Performance test finished.");
}

int main() {
    try {
        run_correctness_test();
        run_performance_test();
    } catch (const std::exception& e) {
        logger.error(fmt::format("An exception occurred: {}", e.what()));
        return 1;
    }
    return 0;
}