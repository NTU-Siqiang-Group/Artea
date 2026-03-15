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
 * @FilePath: /Artea/apps/artea_graph_ann.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: ANN benchmark for hierarchical Artea graph
 */

#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <filesystem>
#include <chrono>
#include <vector>

using namespace artea;
using namespace artea::cpu;
using namespace artea::cpu::default_context;

struct BenchmarkResult {
    double query_time_ms;
    double avg_query_time_us;
    double throughput_qps;
    float recall;
};

auto find_latest_index(const std::string& base_dir, const std::string& dataset_name) -> std::string {
    std::string subdir = "artea_graph." + dataset_name;
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
    hierarchical_graph_router_t& router,
    const vector_array_t& query_vecs,
    const idlist_array_t& groundtruth,
    const vector_array_t& base_vecs,
    dist_func_t& dist_func,
    uint32_t topk
) -> BenchmarkResult {
    auto start_time = std::chrono::high_resolution_clock::now();

    idlist_array_t results = router.batch_query(query_vecs);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    BenchmarkResult result;
    result.query_time_ms = duration.count() / 1000.0;
    result.avg_query_time_us = static_cast<double>(duration.count()) / query_vecs.get_num_vecs();
    result.throughput_qps = query_vecs.get_num_vecs() * 1000000.0 / duration.count();

    recall_estimator_t recall_estimator(dist_func);
    auto recall_metrics = recall_estimator.calculate_recall_at_k(
        results,
        groundtruth,
        query_vecs,
        base_vecs
    );
    result.recall = recall_metrics.strict_recall;

    return result;
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("artea_graph_ann");
    program.add_description("ANN benchmark for hierarchical Artea graph");

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
        .default_value(50u)
        .scan<'u', uint32_t>()
        .help("Bottom layer candidate queue size");

    program.add_argument("--bl-extracted-nbr-size")
        .scan<'u', uint32_t>()
        .help("Bottom layer extracted neighbor size (defaults to graph's max_nbr_size)");

    program.add_argument("--ul-extracted-nbr-size")
        .scan<'u', uint32_t>()
        .help("Upper layer extracted neighbor size (defaults to graph's max_nbr_size)");

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
        logger.info(fmt::format("Using latest index: {}", index_path));
    }

    // Load dataset
    logger.info(fmt::format("Loading dataset: {} from {}", dataset_name, config_path));
    vector_dataset_t dataset(config_path, dataset_name);
    dist_func_t dist_func(dataset.get_base_vecs().get_vec_dim());

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& groundtruth = dataset.get_gt_vecs();

    logger.info(fmt::format("Dataset: {} base vectors, {} query vectors",
        base_vecs.get_num_vecs(), query_vecs.get_num_vecs()));

    // Load hierarchical graph
    logger.info(fmt::format("Loading hierarchical graph from {}...", index_path));

    hierarchical_graph_t hierarchical_graph = hierarchical_graph_file_manager_t::restore(
        index_path,
        base_vecs
    );

    logger.info(fmt::format("Loaded hierarchical graph with {} layers", hierarchical_graph.get_num_layers()));

    // Determine extracted neighbor sizes
    vertex_num_t bl_extracted_nbr_size = program.is_used("--bl-extracted-nbr-size")
        ? program.get<uint32_t>("--bl-extracted-nbr-size")
        : hierarchical_graph.bottom_layer_config().max_nbr_size();

    vertex_num_t ul_extracted_nbr_size = program.is_used("--ul-extracted-nbr-size")
        ? program.get<uint32_t>("--ul-extracted-nbr-size")
        : hierarchical_graph.upper_layer_config().max_nbr_size();

    logger.info(fmt::format("Using extracted neighbor sizes: bottom={}, upper={}",
        bl_extracted_nbr_size, ul_extracted_nbr_size));

    // Convert to hierarchical search graph
    logger.info("Converting to hierarchical search graph...");
    auto hierarchical_search_graph = search_graph_converter_t::from_hierarchical_graph(
        hierarchical_graph,
        bl_extracted_nbr_size,
        ul_extracted_nbr_size
    );

    // Create router
    hierarchical_graph_router_t router(
        base_vecs,
        dist_func,
        hierarchical_search_graph,
        topk,
        candidate_queue_size
    );
    router.initialize();

    logger.info(fmt::format("Router initialized: topk={}, candidate_queue_size={}",
        topk, candidate_queue_size));

    // Warm-up run
    logger.info("Running warm-up...");
    run_benchmark(router, query_vecs, groundtruth, base_vecs, dist_func, topk);
    logger.info("Warm-up completed");

    // Run 3 benchmark iterations
    std::vector<BenchmarkResult> results;
    for (int i = 0; i < 3; ++i) {
        logger.info(fmt::format("Running benchmark iteration {}...", i + 1));
        auto result = run_benchmark(router, query_vecs, groundtruth, base_vecs, dist_func, topk);

        logger.info(fmt::format("  Query time: {:.2f} ms", result.query_time_ms));
        logger.info(fmt::format("  Avg query time: {:.2f} us", result.avg_query_time_us));
        logger.info(fmt::format("  Throughput: {:.2f} QPS", result.throughput_qps));
        logger.info(fmt::format("  Recall@{}: {:.4f}", topk, result.recall));

        results.push_back(result);
    }

    // Compute averages
    double avg_throughput = 0.0;
    double avg_recall = 0.0;
    for (const auto& result : results) {
        avg_throughput += result.throughput_qps;
        avg_recall += result.recall;
    }
    avg_throughput /= results.size();
    avg_recall /= results.size();

    // Print summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    ARTEA HIERARCHICAL GRAPH ANN BENCHMARK SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << fmt::format("Dataset: {}", dataset_name) << std::endl;
    std::cout << fmt::format("Index: {}", index_path) << std::endl;
    std::cout << fmt::format("Top-k: {}", topk) << std::endl;
    std::cout << fmt::format("Candidate queue size: {}", candidate_queue_size) << std::endl;
    std::cout << "\n--- Individual Runs ---" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << fmt::format("Run {}: Throughput = {:.2f} QPS, Recall@{} = {:.4f}",
            i + 1, results[i].throughput_qps, topk, results[i].recall) << std::endl;
    }
    std::cout << "\n--- Average Results ---" << std::endl;
    std::cout << fmt::format("Average Throughput: {:.2f} QPS", avg_throughput) << std::endl;
    std::cout << fmt::format("Average Recall@{}: {:.4f}", topk, avg_recall) << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
