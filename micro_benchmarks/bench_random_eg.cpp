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
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <memory>

using namespace artea;
using namespace artea::cpu;

struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
    layer_config_t layer_config{32, 32};
    conv_graph::pruning_config_t pruning_config{1.0, 0.0};
    conv_graph::propagate_config_t propagate_config{4, 14};
    int64_t iterations;
};

BenchConfig g_config;

class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init() {
        auto dataset = std::make_unique<vector_dataset_t>(
            g_config.config_path,
            g_config.dataset_name
        );

        dim_ = dataset->get_base_vecs().get_vec_dim();
        num_base_vecs_ = dataset->get_base_vecs().get_num_vecs();

        // Move base vectors from dataset
        base_vecs_ = std::move(dataset->get_base_vecs());

        // Initialize distance function
        dist_func_ = std::make_unique<dist_func_t>(dim_);
    }

    vec_dim_t get_dim() const { return dim_; }
    vec_num_t get_num_base_vecs() const { return num_base_vecs_; }
    const vector_array_t& get_base_vecs() const { return base_vecs_; }
    const dist_func_t& get_dist_func() const { return *dist_func_; }

private:
    vec_dim_t dim_;
    vec_num_t num_base_vecs_;
    vector_array_t base_vecs_;
    std::unique_ptr<dist_func_t> dist_func_;
};

// Benchmark for RandomEG
static void BM_RandomEG(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& base_vecs = provider.get_base_vecs();
    const auto& dist_func = provider.get_dist_func();
    const vec_num_t num_vertices = provider.get_num_base_vecs();

    // Create RandomEG instance
    random_eg_t random_eg(dist_func);

    for (auto _ : state) {
        // Create a new flat_graph (included in timing)
        flat_graph_t flat_graph(
            base_vecs,
            g_config.layer_config,
            g_config.pruning_config,
            g_config.propagate_config
        );

        // Perform the random edge generation
        random_eg.generate(flat_graph, g_config.layer_config.max_nbr_size());

        // Prevent optimization from removing the work
        benchmark::DoNotOptimize(flat_graph);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vertices);
    state.SetLabel(fmt::format("vertices={}, max_nbrs={}", num_vertices, g_config.layer_config.max_nbr_size()));
}

BENCHMARK(BM_RandomEG)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(g_config.iterations);

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_random_eg");
    program.add_description("Benchmark for RandomEG (Random Edge Generator)");

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
        .help("Max neighbor array size for the graph");

    // Benchmark control
    program.add_argument("-i", "--iterations")
        .default_value(int64_t(10))
        .scan<'i', int64_t>()
        .help("Number of iterations for benchmarks");

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
    vec_num_t max_nbr_size = static_cast<vec_num_t>(program.get<int>("--max-nbrs"));
    g_config.layer_config = layer_config_t(max_nbr_size, max_nbr_size);
    g_config.iterations = program.get<int64_t>("--iterations");

    ARTEA_INFO(fmt::format("Benchmark Configuration:"));
    ARTEA_INFO(fmt::format("  Dataset: {}", g_config.dataset_name));
    ARTEA_INFO(fmt::format("  Config path: {}", g_config.config_path));
    ARTEA_INFO(fmt::format("  Max neighbors: {}", g_config.layer_config.max_nbr_size()));
    ARTEA_INFO(fmt::format("  Iterations: {}", g_config.iterations));

    DataProvider::instance().init();

    ARTEA_INFO(fmt::format("Dataset loaded:"));
    ARTEA_INFO(fmt::format("  Dimension: {}", DataProvider::instance().get_dim()));
    ARTEA_INFO(fmt::format("  Base vectors: {}", DataProvider::instance().get_num_base_vecs()));

    // Initialize and run Google Benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}