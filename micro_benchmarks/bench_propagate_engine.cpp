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
#include <memory>
#include <vector>

using namespace artea;
using namespace artea::cpu;

using base_traits_t = BaseTraits<uint32_t, float, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using buffer_traits_t = BufferTraits<base_traits_t, BufferPolicyT::LOCKED_BUFFER_WITH_MUTEX, /* BufCapacity = */32>;
using index_traits_t = IndexTraits<base_traits_t>;
using edge_generator_traits_t = EdgeGeneratorTraits<computer_traits_t, buffer_traits_t, index_traits_t>;
using vertex_generator_traits_t = VertexGeneratorTraits<computer_traits_t>;

using constructor_traits_t = ConstructorTraits<
    vertex_generator_traits_t,
    edge_generator_traits_t,
    index_traits_t,
    false  // selective_schedule
>;

using vec_num_t = typename base_traits_t::vec_num_t;
using vec_dim_t = typename base_traits_t::vec_dim_t;
using iter_t = typename base_traits_t::iter_t;
using ratio_t = typename base_traits_t::ratio_t;
using nbr_arr_t = typename base_traits_t::nbr_arr_t;
using dist_func_t = typename computer_traits_t::dist_func_t;
using vector_array_t = typename computer_traits_t::vector_array_t;
using vector_dataset_t = typename computer_traits_t::vector_dataset_t;
using flat_graph_t = typename index_traits_t::flat_graph_t;
using random_eg_t = typename edge_generator_traits_t::random_eg_t;
using triangle_updater_t = typename edge_generator_traits_t::triangle_updater_t;
using reverse_updater_t = typename edge_generator_traits_t::reverse_updater_t;
using random_updater_t = typename edge_generator_traits_t::random_updater_t;
using propagate_engine_t = typename constructor_traits_t::propagate_engine_t;

struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
    vec_num_t init_nbr_size;
    vec_num_t reserved_nbr_size;
    vec_num_t max_nbr_size;
    vec_num_t rand_gen_size;
    iter_t num_iters;
    ratio_t scale_coeffs;
    ratio_t shifted_coeffs;
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

        // Initialize flat graph with random edges
        logger.info("Initializing flat graph with random edges...");
        flat_graph_ = std::make_unique<flat_graph_t>(
            base_vecs_,
            num_base_vecs_,
            g_config.max_nbr_size,
            g_config.reserved_nbr_size
        );

        random_eg_t random_eg(*dist_func_);
        random_eg.generate(*flat_graph_, g_config.init_nbr_size);
        logger.info("Flat graph initialization complete.");

        // Save initial graph state for benchmark reset
        save_initial_state();
    }

    vec_dim_t get_dim() const { return dim_; }
    vec_num_t get_num_base_vecs() const { return num_base_vecs_; }
    const vector_array_t& get_base_vecs() const { return base_vecs_; }
    const dist_func_t& get_dist_func() const { return *dist_func_; }
    flat_graph_t& get_flat_graph() const { return *flat_graph_; }

    // Reset graph to initial state
    void reset_graph() {
        auto& nbrs_arr = flat_graph_->get_nbrs_arr();
        for (size_t i = 0; i < nbrs_arr.size(); ++i) {
            nbrs_arr[i] = initial_nbrs_[i];
        }
    }

private:
    void save_initial_state() {
        const auto& nbrs_arr = flat_graph_->get_nbrs_arr();
        initial_nbrs_.resize(nbrs_arr.size());
        for (size_t i = 0; i < nbrs_arr.size(); ++i) {
            initial_nbrs_[i] = nbrs_arr[i];
        }
    }

    vec_dim_t dim_;
    vec_num_t num_base_vecs_;
    vector_array_t base_vecs_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<flat_graph_t> flat_graph_;
    std::vector<nbr_arr_t> initial_nbrs_;
};

// Benchmark for PropagateEngine with TriangleUpdater
static void BM_PropagateEngine(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& dist_func = provider.get_dist_func();
    flat_graph_t& flat_graph = provider.get_flat_graph();
    const vec_num_t num_vertices = provider.get_num_base_vecs();

    // Create PropagateEngine instance
    propagate_engine_t propagate_engine(num_vertices);
    propagate_engine.set_graph(flat_graph);

    // Set max_nbr_size on the flat graph
    flat_graph.set_max_nbr_size(g_config.max_nbr_size);

    // Create TriangleUpdater using the factory method
    auto triangle_updater = propagate_engine.make_updater<triangle_updater_t>(
        dist_func,
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
        benchmark::DoNotOptimize(flat_graph);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vertices * g_config.num_iters);
    state.SetLabel(fmt::format(
        "vertices={}, iters={}, max_nbrs={}",
        num_vertices,
        g_config.num_iters,
        g_config.max_nbr_size
    ));
}

// Benchmark for PropagateEngine with ReverseUpdater
static void BM_ReverseUpdater(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& dist_func = provider.get_dist_func();
    flat_graph_t& flat_graph = provider.get_flat_graph();
    const vec_num_t num_vertices = provider.get_num_base_vecs();

    // Create PropagateEngine instance
    propagate_engine_t propagate_engine(num_vertices);
    propagate_engine.set_graph(flat_graph);

    // Create ReverseUpdater using the factory method
    auto reverse_updater = propagate_engine.make_updater<reverse_updater_t>(dist_func);

    for (auto _ : state) {
        // Reset graph to initial state before each benchmark iteration
        state.PauseTiming();
        provider.reset_graph();
        state.ResumeTiming();

        // Run the propagation for specified iterations
        propagate_engine.run(g_config.num_iters, reverse_updater);

        // Prevent optimization from removing the work
        benchmark::DoNotOptimize(flat_graph);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vertices * g_config.num_iters);
    state.SetLabel(fmt::format(
        "vertices={}, iters={}, init_nbrs={}",
        num_vertices,
        g_config.num_iters,
        g_config.init_nbr_size
    ));
}

// Benchmark for PropagateEngine with RandomUpdater
static void BM_RandomUpdater(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& dist_func = provider.get_dist_func();
    flat_graph_t& flat_graph = provider.get_flat_graph();
    const vec_num_t num_vertices = provider.get_num_base_vecs();

    // Create PropagateEngine instance
    propagate_engine_t propagate_engine(num_vertices);
    propagate_engine.set_graph(flat_graph);

    // Create RandomUpdater using the factory method
    auto random_updater = propagate_engine.make_updater<random_updater_t>(
        dist_func,
        g_config.rand_gen_size
    );

    for (auto _ : state) {
        // Reset graph to initial state before each benchmark iteration
        state.PauseTiming();
        provider.reset_graph();
        state.ResumeTiming();

        // Run the propagation for specified iterations
        propagate_engine.run(g_config.num_iters, random_updater);

        // Prevent optimization from removing the work
        benchmark::DoNotOptimize(flat_graph);
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
        .default_value(std::string("./datasets.json"))
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
        .default_value(5)
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
    g_config.init_nbr_size = static_cast<vec_num_t>(program.get<int>("--init-nbrs"));
    g_config.reserved_nbr_size = static_cast<vec_num_t>(program.get<int>("--reserved-nbrs"));
    g_config.max_nbr_size = static_cast<vec_num_t>(program.get<int>("--max-nbrs"));
    g_config.rand_gen_size = static_cast<vec_num_t>(program.get<int>("--rand-gen-size"));
    g_config.num_iters = static_cast<iter_t>(program.get<int>("--num-iters"));
    g_config.scale_coeffs = static_cast<ratio_t>(program.get<double>("--scale-coeffs"));
    g_config.shifted_coeffs = static_cast<ratio_t>(program.get<double>("--shifted-coeffs"));
    g_config.iterations = program.get<int64_t>("--iterations");

    logger.info(fmt::format("Benchmark Configuration:"));
    logger.info(fmt::format("  Dataset: {}", g_config.dataset_name));
    logger.info(fmt::format("  Config path: {}", g_config.config_path));
    logger.info(fmt::format("  Init neighbors: {}", g_config.init_nbr_size));
    logger.info(fmt::format("  Reserved neighbors: {}", g_config.reserved_nbr_size));
    logger.info(fmt::format("  Max neighbors: {}", g_config.max_nbr_size));
    logger.info(fmt::format("  Random gen size: {}", g_config.rand_gen_size));
    logger.info(fmt::format("  Propagation iterations: {}", g_config.num_iters));
    logger.info(fmt::format("  Scale coeffs: {}", g_config.scale_coeffs));
    logger.info(fmt::format("  Shifted coeffs: {}", g_config.shifted_coeffs));
    logger.info(fmt::format("  Benchmark iterations: {}", g_config.iterations));

    DataProvider::instance().init();

    logger.info(fmt::format("Dataset loaded:"));
    logger.info(fmt::format("  Dimension: {}", DataProvider::instance().get_dim()));
    logger.info(fmt::format("  Base vectors: {}", DataProvider::instance().get_num_base_vecs()));

    // Dynamically register benchmark with runtime-parsed repetitions
    benchmark::RegisterBenchmark("BM_TriangleUpdater", BM_PropagateEngine)
        ->Unit(benchmark::kMillisecond)
        ->Repetitions(g_config.iterations)
        ->ReportAggregatesOnly(false);  // Show individual repetition results

    benchmark::RegisterBenchmark("BM_ReverseUpdater", BM_ReverseUpdater)
        ->Unit(benchmark::kMillisecond)
        ->Repetitions(g_config.iterations)
        ->ReportAggregatesOnly(false);

    benchmark::RegisterBenchmark("BM_RandomUpdater", BM_RandomUpdater)
        ->Unit(benchmark::kMillisecond)
        ->Repetitions(g_config.iterations)
        ->ReportAggregatesOnly(false);

    // Initialize and run Google Benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
