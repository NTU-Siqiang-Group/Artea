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
 * @FilePath: /Artea/apps/conv_graph_ann.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: ANN benchmark for convergent graph
 */

#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <filesystem>
#include <chrono>
#include <vector>

using namespace artea;
using namespace artea::cpu;

struct BenchmarkResult {
    double query_time_ms;
    double avg_query_time_us;
    double throughput_qps;
    float recall;
};

auto find_latest_index(const std::string& base_dir, const std::string& dataset_name) -> std::string {
    std::string subdir = "conv_graph." + dataset_name;
    std::filesystem::path search_path = std::filesystem::path(base_dir) / subdir;

    if (!std::filesystem::exists(search_path)) {
        throw std::runtime_error(fmt::format("Index directory not found: {}", search_path.string()));
    }

    std::filesystem::path latest_path;
    std::filesystem::file_time_type latest_time;
    bool found = false;

    for (const auto& entry : std::filesystem::directory_iterator(search_path)) {
        if (entry.is_directory()) {
            auto current_time = std::filesystem::last_write_time(entry.path());
            if (!found || current_time > latest_time) {
                latest_time = current_time;
                latest_path = entry.path();
                found = true;
            }
        }
    }

    if (!found) {
        throw std::runtime_error(fmt::format("No index found in {}", search_path.string()));
    }

    return latest_path.string();
}

auto run_benchmark(
    compact::descent_graph_router_t& router,
    const vector_array_t& query_vecs,
    const idlist_array_t& groundtruth,
    const vector_array_t& base_vecs,
    dist_func_t& dist_func,
    uint32_t topk
) -> BenchmarkResult {
    auto start_time = std::chrono::high_resolution_clock::now();

    knn_results_t results = router.batch_query(query_vecs);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    BenchmarkResult result;
    result.query_time_ms = duration.count() / 1000.0;
    result.avg_query_time_us = static_cast<double>(duration.count()) / query_vecs.get_num_vecs();
    result.throughput_qps = query_vecs.get_num_vecs() * 1000000.0 / duration.count();

    recall_estimator_t recall_estimator;
    result.recall = recall_estimator.calculate_recall_at_k(
        results,
        groundtruth,
        topk,
        query_vecs.get_num_vecs()
    );

    return result;
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("conv_graph_ann");
    program.add_description("ANN benchmark for convergent graph");

    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    program.add_argument("-i", "--index")
        .help("Index directory path (if not specified, load latest index)");

    program.add_argument("--index-base-dir")
        .default_value(std::string("graph_index_repo/artea/"))
        .help("Base directory for index search");

    program.add_argument("-k", "--topk")
        .default_value(20u)
        .scan<'u', uint32_t>()
        .help("Top-k results to return");

    program.add_argument("--candidate-queue-size")
        .default_value(80u)
        .scan<'u', uint32_t>()
        .help("Candidate queue size for search");

    program.add_argument("--extracted-nbr-size")
        .scan<'u', uint32_t>()
        .help("Extracted neighbor size (defaults to graph's max_nbr_size)");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    std::string config_path = program.get<std::string>("--config");
    std::string dataset_name = program.get<std::string>("--dataset");
    uint32_t topk = program.get<uint32_t>("--topk");
    uint32_t candidate_queue_size = program.get<uint32_t>("--candidate-queue-size");

    // Determine index path
    std::string index_path;
    if (program.is_used("--index")) {
        index_path = program.get<std::string>("--index");
    } else {
        std::string index_base_dir = program.get<std::string>("--index-base-dir");
        index_path = find_latest_index(index_base_dir, dataset_name);
        ARTEA_INFO(fmt::format("Using latest index: {}", index_path));
    }

    // Load dataset
    ARTEA_INFO(fmt::format("Loading dataset: {} from {}", dataset_name, config_path));
    vector_dataset_t dataset(config_path, dataset_name);
    dist_func_t dist_func(dataset.get_base_vecs().get_vec_dim());

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& groundtruth = dataset.get_gt_vecs();

    ARTEA_INFO(fmt::format("Dataset: {} base vectors, {} query vectors",
        base_vecs.get_num_vecs(), query_vecs.get_num_vecs()));

    // Load flat graph
    ARTEA_INFO(fmt::format("Loading descent graph from {}...", index_path));
    conv_graph::index_t descent_graph = flat_graph_file_manager_t::restore<conv_graph::index_t>(index_path, base_vecs);

    // Calculate and output index size
    index_size_calculator_t index_size_calc;
    auto index_size_info = index_size_calc.calculate_descent_graph_size(descent_graph);

    // Determine extracted neighbor size
    vertex_num_t extracted_nbr_size = program.is_used("--extracted-nbr-size")
        ? program.get<uint32_t>("--extracted-nbr-size")
        : descent_graph.layer_config().max_nbr_size();

    // Print graph construction configuration
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    CONVERGENT GRAPH CONFIGURATION" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "\n--- Dataset ---" << std::endl;
    std::cout << fmt::format("  Name:                   {}", dataset_name) << std::endl;
    std::cout << fmt::format("  Base vectors:           {}", base_vecs.get_num_vecs()) << std::endl;
    std::cout << fmt::format("  Query vectors:          {}", query_vecs.get_num_vecs()) << std::endl;
    std::cout << fmt::format("  Vector dimension:       {}", base_vecs.get_vec_dim()) << std::endl;
    std::cout << "\n--- Graph Construction Config ---" << std::endl;
    std::cout << fmt::format("  Index size:             {:.2f} MB ({} bytes)", index_size_info.total_mb, index_size_info.total_bytes) << std::endl;
    std::cout << fmt::format("  Max nbr size:           {}", descent_graph.layer_config().max_nbr_size()) << std::endl;
    std::cout << fmt::format("  Reserved nbr size:      {}", descent_graph.layer_config().reserved_nbr_size()) << std::endl;
    std::cout << fmt::format("  Scale coeffs:           {}", descent_graph.pruning_config().scale_coeffs()) << std::endl;
    std::cout << fmt::format("  Shifted coeffs:         {}", descent_graph.pruning_config().shifted_coeffs()) << std::endl;
    std::cout << fmt::format("  Build loops:            {}", descent_graph.propagate_config().num_build_loops()) << std::endl;
    std::cout << fmt::format("  Triangle updater iters: {}", descent_graph.propagate_config().num_triu_iters()) << std::endl;
    std::cout << "\n--- Query Config ---" << std::endl;
    std::cout << fmt::format("  Top-k:                  {}", topk) << std::endl;
    std::cout << fmt::format("  Candidate queue size:   {}", candidate_queue_size) << std::endl;
    std::cout << fmt::format("  Extracted nbr size:     {}", extracted_nbr_size) << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;

    ARTEA_INFO(fmt::format("Using extracted neighbor size: {}", extracted_nbr_size));

    // Convert to flat search graph
    ARTEA_INFO("Converting to flat search graph...");
    compact::descent_graph_t compact_descent_graph = descent_graph_compactor_t::compact_graph(
        descent_graph,
        extracted_nbr_size
    );

    // Create router
    compact::descent_graph_router_t router(
        base_vecs,
        dist_func,
        compact_descent_graph,
        topk,
        candidate_queue_size
    );
    router.initialize();

    ARTEA_INFO(fmt::format("Router initialized: topk={}, candidate_queue_size={}",
        topk, candidate_queue_size));

    // Run benchmark iterations (200 total, use last 100 for statistics)
    const uint32_t total_iterations = 200;
    const uint32_t warmup_iterations = 100;
    std::vector<BenchmarkResult> results;

    for (uint32_t i = 0; i < total_iterations; ++i) {
        auto result = run_benchmark(router, query_vecs, groundtruth, base_vecs, dist_func, topk);

        ARTEA_INFO(fmt::format("Iter {}: {:.2f} ms, {:.2f} QPS, Recall@{}={:.4f}",
            i + 1, result.query_time_ms, result.throughput_qps, topk, result.recall));

        // Only collect statistics for last 100 iterations
        if (i >= warmup_iterations) {
            results.push_back(result);
        }
    }

    // Compute averages from last 100 iterations
    double avg_query_time_ms = 0.0;
    double avg_throughput = 0.0;
    double avg_recall = 0.0;
    for (const auto& result : results) {
        avg_query_time_ms += result.query_time_ms;
        avg_throughput += result.throughput_qps;
        avg_recall += result.recall;
    }
    avg_query_time_ms /= results.size();
    avg_throughput /= results.size();
    avg_recall /= results.size();

    // Print summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    CONVERGENT GRAPH ANN BENCHMARK SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << fmt::format("Dataset: {}", dataset_name) << std::endl;
    std::cout << fmt::format("Index: {}", index_path) << std::endl;
    std::cout << fmt::format("Top-k: {}", topk) << std::endl;
    std::cout << fmt::format("Candidate queue size: {}", candidate_queue_size) << std::endl;
    std::cout << fmt::format("Total iterations: {}, Statistics from last: {}", total_iterations, results.size()) << std::endl;
    std::cout << fmt::format("\n--- Average Results (Last {} Iterations) ---", results.size()) << std::endl;
    std::cout << fmt::format("Average Query Time: {:.2f} ms", avg_query_time_ms) << std::endl;
    std::cout << fmt::format("Average Throughput: {:.2f} QPS", avg_throughput) << std::endl;
    std::cout << fmt::format("Average Recall@{}: {:.4f}", topk, avg_recall) << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
