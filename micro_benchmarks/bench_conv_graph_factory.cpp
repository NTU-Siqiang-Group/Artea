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

#include <benchmark/benchmark.h>
#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/default_context.hpp>
#include <memory>

using namespace artea;
using namespace artea::cpu;
using namespace artea::cpu::default_context;

struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
    vertex_num_t max_nbr_size;
    vertex_num_t reserved_nbr_size;
    ratio_t scale_coeffs;
    ratio_t shifted_coeffs;
    iter_t num_outer_iters;
    iter_t num_inner_iters;
    int64_t repetitions;
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

static void BM_ConvGraphFactory(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& dataset = provider.get_dataset();
    const vertex_num_t num_vertices = provider.get_num_base_vecs();

    conv_graph_factory_t conv_graph_factory;

    for (auto _ : state) {
        // Construct the graph
        flat_graph_t flat_graph = conv_graph_factory.construct_graph(
            dataset,
            g_config.max_nbr_size,
            g_config.reserved_nbr_size,
            g_config.scale_coeffs,
            g_config.shifted_coeffs,
            g_config.num_outer_iters,
            g_config.num_inner_iters
        );

        // Prevent optimization from removing the work
        benchmark::DoNotOptimize(flat_graph);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vertices);
    state.SetLabel(fmt::format(
        "vertices={}, max_nbrs={}, outer_iters={}, inner_iters={}, scale={:.2f}, shift={:.2f}",
        num_vertices,
        g_config.max_nbr_size,
        g_config.num_outer_iters,
        g_config.num_inner_iters,
        g_config.scale_coeffs,
        g_config.shifted_coeffs
    ));
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

    // Algorithm parameters
    program.add_argument("--max-nbrs")
        .default_value(32)
        .scan<'i', int>()
        .help("Maximum neighbor size after pruning");

    program.add_argument("--reserved-nbrs")
        .default_value(64)
        .scan<'i', int>()
        .help("Reserved neighbor array size for the graph");

    program.add_argument("--scale-coeffs")
        .default_value(1.0)
        .scan<'g', double>()
        .help("Scale coefficient for triangle inequality pruning");

    program.add_argument("--shifted-coeffs")
        .default_value(0.0)
        .scan<'g', double>()
        .help("Shifted coefficient for triangle inequality pruning");

    program.add_argument("--outer-iters")
        .default_value(4)
        .scan<'i', int>()
        .help("Number of outer iterations (recommended: 4)");

    program.add_argument("--inner-iters")
        .default_value(14)
        .scan<'i', int>()
        .help("Number of inner iterations (recommended: 14)");

    // Benchmark control
    program.add_argument("-r", "--repetitions")
        .default_value(int64_t(3))
        .scan<'i', int64_t>()
        .help("Number of repetitions for benchmarks");

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
    g_config.max_nbr_size = static_cast<vertex_num_t>(program.get<int>("--max-nbrs"));
    g_config.reserved_nbr_size = static_cast<vertex_num_t>(program.get<int>("--reserved-nbrs"));
    g_config.scale_coeffs = static_cast<ratio_t>(program.get<double>("--scale-coeffs"));
    g_config.shifted_coeffs = static_cast<ratio_t>(program.get<double>("--shifted-coeffs"));
    g_config.num_outer_iters = static_cast<iter_t>(program.get<int>("--outer-iters"));
    g_config.num_inner_iters = static_cast<iter_t>(program.get<int>("--inner-iters"));
    g_config.repetitions = program.get<int64_t>("--repetitions");

    logger.info(fmt::format("Benchmark Configuration:"));
    logger.info(fmt::format("  Dataset: {}", g_config.dataset_name));
    logger.info(fmt::format("  Config path: {}", g_config.config_path));
    logger.info(fmt::format("  Max neighbors: {}", g_config.max_nbr_size));
    logger.info(fmt::format("  Reserved neighbors: {}", g_config.reserved_nbr_size));
    logger.info(fmt::format("  Scale coeffs: {}", g_config.scale_coeffs));
    logger.info(fmt::format("  Shifted coeffs: {}", g_config.shifted_coeffs));
    logger.info(fmt::format("  Outer iterations: {}", g_config.num_outer_iters));
    logger.info(fmt::format("  Inner iterations: {}", g_config.num_inner_iters));
    logger.info(fmt::format("  Benchmark repetitions: {}", g_config.repetitions));

    DataProvider::instance().init();

    // Register benchmark
    benchmark::RegisterBenchmark("BM_ConvGraphFactory", BM_ConvGraphFactory)
        ->Unit(benchmark::kMillisecond)
        ->Repetitions(g_config.repetitions)
        ->ReportAggregatesOnly(false);

    // Initialize and run Google Benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
