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

#include <arena_benchmark/arena_benchmark.hpp>
#include <benchmark/benchmark.h>
#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <memory>
#include <filesystem>

using namespace artea;
using namespace artea::cpu;
using namespace arena_benchmark;

struct GraphParams {
    layer_config_t layer_config;
    conv_graph::edges_builder_config_t edges_builder_config;
};

struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
    std::string export_path;
    int64_t repetitions;
    int64_t warm_up;
    std::vector<GraphParams> param_sets;
};

BenchConfig g_config;

class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init() {
        dataset_ = std::make_unique<vector_dataset_t>(
            g_config.config_path,
            g_config.dataset_name
        );

        dim_ = dataset_->get_vec_dim();
        num_base_vecs_ = dataset_->get_num_base_vecs();

        logger.info(fmt::format("Dataset loaded:"));
        logger.info(fmt::format("  Dimension: {}", dim_));
        logger.info(fmt::format("  Base vectors: {}", num_base_vecs_));
    }

    vec_dim_t get_dim() const { return dim_; }
    vertex_num_t get_num_base_vecs() const { return num_base_vecs_; }
    const vector_dataset_t& get_dataset() const { return *dataset_; }

private:
    vec_dim_t dim_;
    vertex_num_t num_base_vecs_;
    std::unique_ptr<vector_dataset_t> dataset_;
};

auto make_benchmark_func(const GraphParams& params) {
    return [params](benchmark::State& state) {
        auto& provider = DataProvider::instance();
        const auto& dataset = provider.get_dataset();
        const vertex_num_t num_vertices = provider.get_num_base_vecs();

        conv_graph_factory_t conv_graph_factory;

        for (auto _ : state) {
            // Construct the graph
            flat_graph_t flat_graph = conv_graph_factory.construct_graph(
                dataset,
                params.layer_config,
                params.edges_builder_config
            );

            // Prevent optimization from removing the work
            benchmark::DoNotOptimize(flat_graph);
            benchmark::ClobberMemory();
        }

        state.SetItemsProcessed(state.iterations() * num_vertices);
    };
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_conv_graph_factory");
    program.add_description("Benchmark for convergent graph factory");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    // Benchmark control
    program.add_argument("-r", "--repetitions")
        .default_value(int64_t(5))
        .scan<'i', int64_t>()
        .help("Number of repetitions for benchmarks");

    program.add_argument("-w", "--warm-up")
        .default_value(int64_t(1))
        .scan<'i', int64_t>()
        .help("Number of warm-up repetitions");

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

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.export_path = program.get<std::string>("--export");
    g_config.repetitions = program.get<int64_t>("--repetitions");
    g_config.warm_up = program.get<int64_t>("--warm-up");

    // Initialize parameter sets
    g_config.param_sets = {
        // Set 1: max_nbrs=32, reserved=64, outer_iters=4, inner_iters=14, scale=1.10, shift=0.00
        {layer_config_t(32, 64), conv_graph::edges_builder_config_t(1.10, 0.00, 4, 14)},
        // Set 2: max_nbrs=64, reserved=128, outer_iters=4, inner_iters=14, scale=1.00, shift=0.00
        {layer_config_t(64, 128), conv_graph::edges_builder_config_t(1.00, 0.00, 4, 14)}
    };

    logger.info(fmt::format("Benchmark Configuration:"));
    logger.info(fmt::format("  Dataset: {}", g_config.dataset_name));
    logger.info(fmt::format("  Config path: {}", g_config.config_path));
    logger.info(fmt::format("  Export path: {}", g_config.export_path));
    logger.info(fmt::format("  Benchmark repetitions: {}", g_config.repetitions));
    logger.info(fmt::format("  Warm-up repetitions: {}", g_config.warm_up));
    logger.info(fmt::format("  Number of parameter sets: {}", g_config.param_sets.size()));

    DataProvider::instance().init();

    const auto& provider = DataProvider::instance();
    const vertex_num_t num_vertices = provider.get_num_base_vecs();

    // Create arena benchmark instance
    ArenaBenchmark bench;

    // Register benchmarks for each parameter set
    for (size_t i = 0; i < g_config.param_sets.size(); ++i) {
        const auto& params = g_config.param_sets[i];

        std::string bench_name = fmt::format("BM_ConvGraphFactory_Set{}", i + 1);

        bench.register_benchmark(bench_name, make_benchmark_func(params))
            .repetitions(static_cast<int>(g_config.repetitions))
            .workload_scale(num_vertices)
            .time_unit(benchmark::kMillisecond)
            .extra_info(fmt::format(
                "vertices={}, max_nbrs={}, reserved_nbrs={}, outer_iters={}, inner_iters={}, scale={:.2f}, shift={:.2f}",
                num_vertices,
                params.layer_config.max_nbr_size(),
                params.layer_config.reserved_nbr_size(),
                params.edges_builder_config.num_outer_iters(),
                params.edges_builder_config.num_triu_iters(),
                params.edges_builder_config.scale_coeffs(),
                params.edges_builder_config.shifted_coeffs()
            ));
    }

    // Run benchmarks with warm-up and export results
    bench.warm_up(static_cast<int>(g_config.warm_up))
         .run_all(argc, argv)
         .export_results(g_config.export_path);

    const auto logs_dir = std::filesystem::path(g_config.export_path) / "benchmark_logs";
    const auto results_dir = std::filesystem::path(g_config.export_path) / "benchmark_results";

    logger.info(fmt::format("\nExport completed:"));
    logger.info(fmt::format("  Repetition logs: {}", logs_dir.string()));
    logger.info(fmt::format("  Summary results: {}", results_dir.string()));

    return 0;
}
