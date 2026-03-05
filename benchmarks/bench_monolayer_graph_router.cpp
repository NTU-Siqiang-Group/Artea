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
 * @FilePath: /Artea/benchmarks/bench_monolayer_graph_router.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Benchmark monolayer graph router search performance
 */

#include "monolayer_graph_router_bench.hpp"
#include <argparse/argparse.hpp>
#include <fstream>
#include <set>
#include <iostream>

using namespace artea;
using namespace artea::benchmarks;
using namespace artea::cpu::default_context;
using namespace arena_benchmark;

// Define global variables
BenchConfig artea::benchmarks::g_config;
std::map<std::string, std::unique_ptr<vector_dataset_t>> artea::benchmarks::g_datasets;
std::map<std::string, idlist_array_t> artea::benchmarks::g_search_results;

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_monolayer_graph_router");
    program.add_description("Benchmark for monolayer graph router");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to dataset configuration file");

    // Workload file
    program.add_argument("-w", "--workload")
        .required()
        .help("Path to search workload JSON file");

    // Search parameters
    program.add_argument("-k", "--topk")
        .default_value(std::vector<uint32_t>{10})
        .scan<'u', uint32_t>()
        .nargs(argparse::nargs_pattern::at_least_one)
        .help("Top-K values to test");

    program.add_argument("--candidate-queue-sizes")
        .default_value(std::vector<uint32_t>{32})
        .scan<'u', uint32_t>()
        .nargs(argparse::nargs_pattern::at_least_one)
        .help("Candidate queue sizes to test");

    // Benchmark control
    program.add_argument("-r", "--repetitions")
        .default_value(int64_t(10))
        .scan<'i', int64_t>()
        .help("Number of repetitions for benchmarks (default: 10)");

    program.add_argument("-e", "--export")
        .default_value(std::string("results"))
        .help("Export path for benchmark results");

    program.add_argument("-h", "--help")
        .default_value(false)
        .implicit_value(true)
        .help("Show this help message");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    if (program.get<bool>("--help")) {
        std::cout << program;
        return 0;
    }

    // Parse configuration
    g_config.config_path = program.get<std::string>("--config");
    g_config.export_path = program.get<std::string>("--export");
    g_config.repetitions = program.get<int64_t>("--repetitions");
    g_config.warm_up = 3;  // Fixed warm-up to 3 runs

    std::string workload_path = program.get<std::string>("--workload");
    auto topk_values = program.get<std::vector<uint32_t>>("--topk");
    auto queue_sizes = program.get<std::vector<uint32_t>>("--candidate-queue-sizes");

    // Load indices from workload file
    logger.info(fmt::format("Loading workload from: {}", workload_path));

    std::ifstream workload_file(workload_path);
    if (!workload_file.is_open()) {
        std::cerr << "Failed to open workload file: " << workload_path << std::endl;
        return 1;
    }

    nlohmann::json workload_json;
    workload_file >> workload_json;

    struct IndexInfo {
        std::string index_path;
        std::string dataset_name;
        std::string algorithm;
        uint32_t max_nbr_size;
        uint32_t extracted_nbr_size;
        double scale_coeffs;
        double shifted_coeffs;
        uint32_t num_outer_iters;
        uint32_t num_inner_iters;
    };
    std::vector<IndexInfo> index_infos;

    for (const auto& entry : workload_json) {
        IndexInfo info;
        info.algorithm = entry["algorithm"].get<std::string>();
        info.dataset_name = entry["params"]["dataset"].get<std::string>();
        info.max_nbr_size = entry["params"]["max_nbr_size"].get<uint32_t>();
        info.extracted_nbr_size = entry["params"]["extracted_nbr_size"].get<uint32_t>();
        info.scale_coeffs = entry["params"]["scale_coeffs"].get<double>();
        info.shifted_coeffs = entry["params"]["shifted_coeffs"].get<double>();
        info.num_outer_iters = entry["params"]["num_outer_iters"].get<uint32_t>();
        info.num_inner_iters = entry["params"]["num_inner_iters"].get<uint32_t>();

        // file_path is relative to project root directory
        info.index_path = entry["file_path"].get<std::string>();

        index_infos.push_back(info);
        logger.info(fmt::format("  Found index: {} (dataset: {})",
            info.index_path, info.dataset_name));
    }

    logger.info(fmt::format("Benchmark Configuration:"));
    logger.info(fmt::format("  Config path: {}", g_config.config_path));
    logger.info(fmt::format("  Repetitions: {}", g_config.repetitions));
    logger.info(fmt::format("  Warm-up: {} (fixed)", g_config.warm_up));
    logger.info(fmt::format("  Trimmed average: enabled"));
    logger.info(fmt::format("  Export path: {}", g_config.export_path));

    // Load all required datasets
    std::set<std::string> unique_datasets;
    for (const auto& info : index_infos) {
        unique_datasets.insert(info.dataset_name);
    }

    for (const auto& dataset_name : unique_datasets) {
        logger.info(fmt::format("Loading dataset '{}'...", dataset_name));
        DataProvider::init(dataset_name);
    }

    // Create parameter combinations
    for (const auto& info : index_infos) {
        for (uint32_t topk : topk_values) {
            for (uint32_t queue_size : queue_sizes) {
                SearchParams params;
                params.topk = topk;
                params.candidate_queue_size = queue_size;
                params.index_path = info.index_path;
                params.dataset_name = info.dataset_name;
                params.algorithm = info.algorithm;
                params.max_nbr_size = info.max_nbr_size;
                params.extracted_nbr_size = info.extracted_nbr_size;
                params.scale_coeffs = info.scale_coeffs;
                params.shifted_coeffs = info.shifted_coeffs;
                params.num_outer_iters = info.num_outer_iters;
                params.num_inner_iters = info.num_inner_iters;
                g_config.param_sets.push_back(params);
            }
        }
    }

    // Create arena benchmark instance
    ArenaBenchmark bench;

    // Register benchmarks for each parameter set
    for (const auto& params : g_config.param_sets) {
        // Load the index for this parameter set
        IndexProvider::instance().load_index(params.index_path, params.dataset_name, params.extracted_nbr_size);

        // Format: algorithm/dataset/mns_ens_sc_sh_noi_nii/topk_queue
        std::string bench_name = fmt::format(
            "{}/{}/{}_{}_{:.2f}_{:.2f}_{}_{}/{}_{}",
            params.algorithm,
            params.dataset_name,
            params.max_nbr_size,
            params.extracted_nbr_size,
            params.scale_coeffs,
            params.shifted_coeffs,
            params.num_outer_iters,
            params.num_inner_iters,
            params.topk,
            params.candidate_queue_size
        );

        const auto& dataset = DataProvider::get_dataset(params.dataset_name);
        const vertex_num_t num_queries = dataset.get_num_query_vecs();

        bench.register_benchmark(bench_name, make_benchmark_func(params, bench_name))
            .repetitions(static_cast<int>(g_config.repetitions))
            .workload_scale(num_queries)
            .time_unit(benchmark::kMillisecond)
            .extra_info(fmt::format(
                "topk={}, queue_size={}, max_nbr_size={}, extracted_nbr_size={}",
                params.topk,
                params.candidate_queue_size,
                params.max_nbr_size,
                params.extracted_nbr_size
            ));
    }

    // Run benchmarks with warm-up, trimmed average, and export results
    bench.trimmed_avg_enabled(true)
         .warm_up(static_cast<int>(g_config.warm_up))
         .run_all(argc, argv)
         .export_results(g_config.export_path);

    const auto logs_dir = std::filesystem::path(g_config.export_path) / "benchmark_logs";
    const auto results_dir = std::filesystem::path(g_config.export_path) / "benchmark_results";

    std::cout << std::endl;
    logger.info(fmt::format("Export completed:"));
    logger.info(fmt::format("  Repetition logs: {}", logs_dir.string()));
    logger.info(fmt::format("  Summary results: {}", results_dir.string()));

    // Calculate and output recall for each benchmark
    std::cout << std::endl << "======================= Recall Evaluation =========================" << std::endl;
    for (const auto& params : g_config.param_sets) {
        std::string bench_name = fmt::format(
            "{}/{}/{}_{}_{:.2f}_{:.2f}_{}_{}/{}_{}",
            params.algorithm,
            params.dataset_name,
            params.max_nbr_size,
            params.extracted_nbr_size,
            params.scale_coeffs,
            params.shifted_coeffs,
            params.num_outer_iters,
            params.num_inner_iters,
            params.topk,
            params.candidate_queue_size
        );

        const auto& dataset = DataProvider::get_dataset(params.dataset_name);
        const vec_dim_t dim = dataset.get_vec_dim();

        // Create distance function and recall estimator
        dist_func_t dist_func(dim);
        recall_estimator_t recall_estimator(dist_func);

        // Get search results
        const auto& results = g_search_results.at(bench_name);

        std::cout << std::endl;
        logger.info(fmt::format("Benchmark: {}", bench_name));

        // Calculate recall
        RecallMetrics metrics = recall_estimator.calculate_recall_at_k(
            results,
            dataset.get_gt_vecs(),
            dataset.get_query_vecs(),
            dataset.get_base_vecs()
        );
    }

    return 0;
}
