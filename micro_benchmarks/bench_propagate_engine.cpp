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
#include <vector>

using namespace artea::cpu;
using namespace artea;

struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
    layer_config_t layer_config{16, 32};
    conv_graph::pruning_config_t pruning_config{1.0, 0.0};
    conv_graph::propagate_config_t propagate_config{4, 14, 0.6};
    vec_num_t rand_gen_size;
    iter_t num_iters;
    ratio_t scale_coeffs;
    ratio_t shifted_coeffs;
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

        // Initialize flat graph with random edges
        ARTEA_INFO("Initializing descent graph with random edges...");
        graph_index_ = std::make_unique<conv_graph::index_t>(
            base_vecs_,
            g_config.layer_config,
            g_config.pruning_config,
            g_config.propagate_config
        );

        random_eg_t random_eg(*dist_func_);
        random_eg.generate(graph_index_->get_refining_graph(), static_cast<vec_num_t>(
            g_config.layer_config.max_nbr_size() * g_config.propagate_config.prefill_ratio()
        ));
        ARTEA_INFO("Descent graph initialization complete.");

        // Save initial graph state for benchmark reset
        save_initial_state();
    }

    vec_dim_t get_dim() const { return dim_; }
    vec_num_t get_num_base_vecs() const { return num_base_vecs_; }
    const vector_array_t& get_base_vecs() const { return base_vecs_; }
    const dist_func_t& get_dist_func() const { return *dist_func_; }
    conv_graph::index_t& get_graph_index() const { return *graph_index_; }

    // Reset graph to initial state
    void reset_graph() {
        auto& nbrs_arr = graph_index_->get_nbrs_arr();
        for (size_t i = 0; i < nbrs_arr.size(); ++i) {
            nbrs_arr[i] = initial_nbrs_[i];
        }
    }

private:
    void save_initial_state() {
        const auto& nbrs_arr = graph_index_->get_nbrs_arr();
        initial_nbrs_.resize(nbrs_arr.size());
        for (size_t i = 0; i < nbrs_arr.size(); ++i) {
            initial_nbrs_[i] = nbrs_arr[i];
        }
    }

    vec_dim_t dim_;
    vec_num_t num_base_vecs_;
    vector_array_t base_vecs_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<conv_graph::index_t> graph_index_;
    std::vector<nbr_arr_t> initial_nbrs_;
};

// Benchmark for PropagateEngine with TriangleUpdater (with selective scheduling)
static void BM_TriangleUpdater(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& dist_func = provider.get_dist_func();
    conv_graph::index_t& graph_index = provider.get_graph_index();
    const vec_num_t num_vertices = provider.get_num_base_vecs();

    // Set max_nbr_size on the flat graph
    graph_index.layer_config().max_nbr_size(g_config.layer_config.max_nbr_size());

    // Create PropagateEngine instance
    propagate_engine_t propagate_engine(dist_func);
    propagate_engine.set_graph(graph_index.get_refining_graph());

    // Create TriangleUpdater using the factory method
    auto triangle_updater = propagate_engine.make_updater<triangle_updater_t>(
        g_config.scale_coeffs,
        g_config.shifted_coeffs
    );

    for (auto _ : state) {
        // Reset graph to initial state before each benchmark iteration
        state.PauseTiming();
        provider.reset_graph();
        state.ResumeTiming();

        // Run the propagation for specified iterations
        propagate_engine.run(g_config.num_iters, triangle_updater);

        // Prevent optimization from removing the work
        benchmark::DoNotOptimize(graph_index);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vertices * g_config.num_iters);
    state.SetLabel(fmt::format(
        "vertices={}, iters={}, max_nbrs={}",
        num_vertices,
        g_config.num_iters,
        g_config.layer_config.max_nbr_size()
    ));
}

// Benchmark for PropagateEngine with TriangleUpdater (without selective scheduling)
static void BM_TriangleUpdater_NoSS(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& dist_func = provider.get_dist_func();
    conv_graph::index_t& graph_index = provider.get_graph_index();
    const vec_num_t num_vertices = provider.get_num_base_vecs();

    // Set max_nbr_size on the flat graph
    graph_index.layer_config().max_nbr_size(g_config.layer_config.max_nbr_size());

    // Create PropagateEngine instance without selective scheduling
    propagate_engine_t propagate_engine(dist_func);
    propagate_engine.set_graph(graph_index.get_refining_graph());

    // Create TriangleUpdater using the factory method
    auto triangle_updater = propagate_engine.make_updater<triangle_updater_t>(
        g_config.scale_coeffs,
        g_config.shifted_coeffs
    );

    for (auto _ : state) {
        // Reset graph to initial state before each benchmark iteration
        state.PauseTiming();
        provider.reset_graph();
        state.ResumeTiming();

        // Run the propagation for specified iterations
        propagate_engine.run(g_config.num_iters, triangle_updater);

        // Prevent optimization from removing the work
        benchmark::DoNotOptimize(graph_index);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vertices * g_config.num_iters);
    state.SetLabel(fmt::format(
        "vertices={}, iters={}, max_nbrs={}",
        num_vertices,
        g_config.num_iters,
        g_config.layer_config.max_nbr_size()
    ));
}

// Benchmark for PropagateEngine with ReverseUpdater
static void BM_ReverseUpdater(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& dist_func = provider.get_dist_func();
    conv_graph::index_t& graph_index = provider.get_graph_index();
    const vec_num_t num_vertices = provider.get_num_base_vecs();

    // Create PropagateEngine instance
    propagate_engine_t propagate_engine(dist_func);
    propagate_engine.set_graph(graph_index.get_refining_graph());

    // Create ReverseUpdater using the factory method
    auto reverse_updater = propagate_engine.make_updater<reverse_updater_t>();

    for (auto _ : state) {
        // Reset graph to initial state before each benchmark iteration
        state.PauseTiming();
        provider.reset_graph();
        state.ResumeTiming();

        // Run the propagation for specified iterations
        propagate_engine.run(g_config.num_iters, reverse_updater);

        // Prevent optimization from removing the work
        benchmark::DoNotOptimize(graph_index);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vertices * g_config.num_iters);
    state.SetLabel(fmt::format(
        "vertices={}, iters={}, init_nbrs={}",
        num_vertices,
        g_config.num_iters,
        g_config.layer_config.max_nbr_size()
    ));
}

// Benchmark for PropagateEngine with ReverseUpdater (without selective scheduling)
static void BM_ReverseUpdater_NoSS(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& dist_func = provider.get_dist_func();
    conv_graph::index_t& graph_index = provider.get_graph_index();
    const vec_num_t num_vertices = provider.get_num_base_vecs();

    // Create PropagateEngine instance without selective scheduling
    propagate_engine_t propagate_engine(dist_func);
    propagate_engine.set_graph(graph_index.get_refining_graph());

    // Create ReverseUpdater using the factory method
    auto reverse_updater = propagate_engine.make_updater<reverse_updater_t>();

    for (auto _ : state) {
        // Reset graph to initial state before each benchmark iteration
        state.PauseTiming();
        provider.reset_graph();
        state.ResumeTiming();

        // Run the propagation for specified iterations
        propagate_engine.run(g_config.num_iters, reverse_updater);

        // Prevent optimization from removing the work
        benchmark::DoNotOptimize(graph_index);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vertices * g_config.num_iters);
    state.SetLabel(fmt::format(
        "vertices={}, iters={}, init_nbrs={}",
        num_vertices,
        g_config.num_iters,
        g_config.layer_config.max_nbr_size()
    ));
}

// Benchmark for PropagateEngine with RandomUpdater
static void BM_RandomUpdater(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& dist_func = provider.get_dist_func();
    conv_graph::index_t& graph_index = provider.get_graph_index();
    const vec_num_t num_vertices = provider.get_num_base_vecs();

    // Create PropagateEngine instance
    propagate_engine_t propagate_engine(dist_func);
    propagate_engine.set_graph(graph_index.get_refining_graph());

    // Create RandomUpdater using the factory method
    auto random_updater = propagate_engine.make_updater<random_updater_t>(g_config.rand_gen_size);

    for (auto _ : state) {
        // Reset graph to initial state before each benchmark iteration
        state.PauseTiming();
        provider.reset_graph();
        state.ResumeTiming();

        // Run the propagation for specified iterations
        propagate_engine.run(g_config.num_iters, random_updater);

        // Prevent optimization from removing the work
        benchmark::DoNotOptimize(graph_index);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vertices * g_config.num_iters);
    state.SetLabel(fmt::format(
        "vertices={}, iters={}, rand_gen_size={}",
        num_vertices,
        g_config.num_iters,
        g_config.rand_gen_size
    ));
}

// Benchmark for PropagateEngine with RandomUpdater (without selective scheduling)
static void BM_RandomUpdater_NoSS(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& dist_func = provider.get_dist_func();
    conv_graph::index_t& graph_index = provider.get_graph_index();
    const vec_num_t num_vertices = provider.get_num_base_vecs();

    // Create PropagateEngine instance without selective scheduling
    propagate_engine_t propagate_engine(dist_func);
    propagate_engine.set_graph(graph_index.get_refining_graph());

    // Create RandomUpdater using the factory method
    auto random_updater = propagate_engine.make_updater<random_updater_t>(g_config.rand_gen_size);

    for (auto _ : state) {
        // Reset graph to initial state before each benchmark iteration
        state.PauseTiming();
        provider.reset_graph();
        state.ResumeTiming();

        // Run the propagation for specified iterations
        propagate_engine.run(g_config.num_iters, random_updater);

        // Prevent optimization from removing the work
        benchmark::DoNotOptimize(graph_index);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vertices * g_config.num_iters);
    state.SetLabel(fmt::format(
        "vertices={}, iters={}, rand_gen_size={}",
        num_vertices,
        g_config.num_iters,
        g_config.rand_gen_size
    ));
}

// Forward declaration - benchmark will be registered dynamically in main()
// BENCHMARK(BM_PropagateEngine)
//     ->Unit(benchmark::kMillisecond)
//     ->Repetitions(g_config.iterations);

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_propagate_engine");
    program.add_description("Benchmark for PropagateEngine with TriangleUpdater, ReverseUpdater, and RandomUpdater");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    // Algorithm parameters
    program.add_argument("--init-nbrs")
        .default_value(32)
        .scan<'i', int>()
        .help("Number of random neighbors to generate for initial graph");

    program.add_argument("--reserved-nbrs")
        .default_value(32)
        .scan<'i', int>()
        .help("Reserved neighbor array size for the graph");

    program.add_argument("--max-nbrs")
        .default_value(16)
        .scan<'i', int>()
        .help("Maximum neighbor size after pruning");

    program.add_argument("--rand-gen-size")
        .default_value(10)
        .scan<'i', int>()
        .help("Number of random neighbors to generate per vertex (for RandomUpdater)");

    program.add_argument("--num-iters")
        .default_value(10)
        .scan<'i', int>()
        .help("Number of propagation iterations to run");

    program.add_argument("--scale-coeffs")
        .default_value(1.0)
        .scan<'g', double>()
        .help("Scale coefficient for triangle inequality pruning");

    program.add_argument("--shifted-coeffs")
        .default_value(0.0)
        .scan<'g', double>()
        .help("Shifted coefficient for triangle inequality pruning");

    program.add_argument("--prefill-ratio")
        .default_value(0.6f)
        .scan<'g', float>()
        .help("Prefill ratio for initial random graph (init_nbr_size = max_nbr_size * prefill_ratio)");

    // Benchmark control
    program.add_argument("-r", "--repetitions")
        .default_value(int64_t(5))
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
    g_config.layer_config = layer_config_t(
        static_cast<vec_num_t>(program.get<int>("--max-nbrs")),
        static_cast<vec_num_t>(program.get<int>("--reserved-nbrs"))
    );
    g_config.rand_gen_size = static_cast<vec_num_t>(program.get<int>("--rand-gen-size"));
    g_config.num_iters = static_cast<iter_t>(program.get<int>("--num-iters"));
    g_config.scale_coeffs = static_cast<ratio_t>(program.get<double>("--scale-coeffs"));
    g_config.shifted_coeffs = static_cast<ratio_t>(program.get<double>("--shifted-coeffs"));
    g_config.propagate_config.prefill_ratio(program.get<float>("--prefill-ratio"));
    g_config.repetitions = program.get<int64_t>("--repetitions");

    ARTEA_INFO(fmt::format("Benchmark Configuration:"));
    ARTEA_INFO(fmt::format("  Dataset: {}", g_config.dataset_name));
    ARTEA_INFO(fmt::format("  Config path: {}", g_config.config_path));
    ARTEA_INFO(fmt::format("  Reserved neighbors: {}", g_config.layer_config.reserved_nbr_size()));
    ARTEA_INFO(fmt::format("  Max neighbors: {}", g_config.layer_config.max_nbr_size()));
    ARTEA_INFO(fmt::format("  Random gen size: {}", g_config.rand_gen_size));
    ARTEA_INFO(fmt::format("  Propagation iterations: {}", g_config.num_iters));
    ARTEA_INFO(fmt::format("  Scale coeffs: {}", g_config.scale_coeffs));
    ARTEA_INFO(fmt::format("  Shifted coeffs: {}", g_config.shifted_coeffs));
    ARTEA_INFO(fmt::format("  Benchmark repetitions: {}", g_config.repetitions));

    DataProvider::instance().init();

    ARTEA_INFO(fmt::format("Dataset loaded:"));
    ARTEA_INFO(fmt::format("  Dimension: {}", DataProvider::instance().get_dim()));
    ARTEA_INFO(fmt::format("  Base vectors: {}", DataProvider::instance().get_num_base_vecs()));

    // Register benchmarks
    benchmark::RegisterBenchmark("BM_TriangleUpdater_NoSS", BM_TriangleUpdater_NoSS)
        ->Unit(benchmark::kMillisecond)
        ->Repetitions(g_config.repetitions)
        ->ReportAggregatesOnly(false);

    benchmark::RegisterBenchmark("BM_TriangleUpdater", BM_TriangleUpdater)
        ->Unit(benchmark::kMillisecond)
        ->Repetitions(g_config.repetitions)
        ->ReportAggregatesOnly(false);

    benchmark::RegisterBenchmark("BM_ReverseUpdater_NoSS", BM_ReverseUpdater_NoSS)
        ->Unit(benchmark::kMillisecond)
        ->Repetitions(g_config.repetitions)
        ->ReportAggregatesOnly(false);

    benchmark::RegisterBenchmark("BM_ReverseUpdater", BM_ReverseUpdater)
        ->Unit(benchmark::kMillisecond)
        ->Repetitions(g_config.repetitions)
        ->ReportAggregatesOnly(false);

    benchmark::RegisterBenchmark("BM_RandomUpdater_NoSS", BM_RandomUpdater_NoSS)
        ->Unit(benchmark::kMillisecond)
        ->Repetitions(g_config.repetitions)
        ->ReportAggregatesOnly(false);

    benchmark::RegisterBenchmark("BM_RandomUpdater", BM_RandomUpdater)
        ->Unit(benchmark::kMillisecond)
        ->Repetitions(g_config.repetitions)
        ->ReportAggregatesOnly(false);

    // Initialize and run Google Benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
