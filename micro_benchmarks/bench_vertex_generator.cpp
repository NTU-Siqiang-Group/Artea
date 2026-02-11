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
#include <cstring>

using namespace artea;
using namespace artea::cpu;

using base_traits_t = BaseTraits<uint32_t, float, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using vg_traits_t = VertexGeneratorTraits<computer_traits_t>;
using dist_func_t = typename vg_traits_t::dist_func_t;
using vector_array_t = typename vg_traits_t::vector_array_t;
using mb_greedy_vg_t = typename vg_traits_t::mb_greedy_vg_t;
using lb_greedy_vg_t = typename vg_traits_t::lb_greedy_vg_t;
using random_vg_t = typename vg_traits_t::random_vg_t;

struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
    float min_radius;
    uint32_t max_result_size;
    uint32_t small_batch_size;
    uint32_t large_batch_size;
    uint32_t term_thresh;
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
        auto dataset = std::make_unique<typename base_traits_t::vector_dataset_t>(
            g_config.config_path,
            g_config.dataset_name
        );

        dim_ = dataset->get_base_vecs().get_vec_dim();
        num_base_vecs_ = dataset->get_base_vecs().get_num_vecs();

        // Use all base vectors from dataset
        base_vecs_ = std::move(dataset->get_base_vecs());

        // Initialize distance function
        dist_func_ = std::make_unique<dist_func_t>(dim_);
    }

    uint32_t get_dim() const { return dim_; }
    uint32_t get_num_base_vecs() const { return num_base_vecs_; }
    const vector_array_t& get_base_vecs() const { return base_vecs_; }
    const dist_func_t& get_dist_func() const { return *dist_func_; }

private:
    uint32_t dim_;
    uint32_t num_base_vecs_;
    vector_array_t base_vecs_;
    std::unique_ptr<dist_func_t> dist_func_;
};

// Benchmark for RandomVG
static void BM_RandomVG(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    random_vg_t generator;

    for (auto _ : state) {
        auto result = generator.generate(
            provider.get_base_vecs(),
            g_config.max_result_size
        );
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    // Report the actual result size in the last iteration
    auto result = generator.generate(
        provider.get_base_vecs(),
        g_config.max_result_size
    );
    state.counters["result_size"] = benchmark::Counter(
        static_cast<double>(result.size())
    );
    state.counters["approx_rnet_ratio(%)"] = benchmark::Counter(
        100.0 * result.size() / provider.get_num_base_vecs()
    );
}
BENCHMARK(BM_RandomVG)
    ->Name("RandomVG")
    ->Unit(benchmark::kMillisecond);

// Benchmark for MBGreedyVG
static void BM_MBGreedyVG(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    mb_greedy_vg_t generator(provider.get_dist_func());

    for (auto _ : state) {
        auto result = generator.generate(
            provider.get_base_vecs(),
            g_config.min_radius,
            g_config.max_result_size,
            g_config.small_batch_size
        );
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    // Report the actual result size in the last iteration
    auto result = generator.generate(
        provider.get_base_vecs(),
        g_config.min_radius,
        g_config.max_result_size,
        g_config.small_batch_size
    );
    state.counters["result_size"] = benchmark::Counter(
        static_cast<double>(result.size())
    );
    state.counters["approx_rnet_ratio(%)"] = benchmark::Counter(
        100.0 * result.size() / provider.get_num_base_vecs()
    );
}
BENCHMARK(BM_MBGreedyVG)
    ->Name("MBGreedyVG")
    ->Unit(benchmark::kMillisecond);

// Benchmark for LBGreedyVG
static void BM_LBGreedyVG(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    lb_greedy_vg_t generator(provider.get_dist_func());

    for (auto _ : state) {
        auto result = generator.generate(
            provider.get_base_vecs(),
            g_config.min_radius,
            g_config.max_result_size,
            g_config.large_batch_size,
            g_config.term_thresh
        );
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    // Report the actual result size in the last iteration
    auto result = generator.generate(
        provider.get_base_vecs(),
        g_config.min_radius,
        g_config.max_result_size,
        g_config.large_batch_size,
        g_config.term_thresh
    );
    state.counters["result_size"] = benchmark::Counter(
        static_cast<double>(result.size())
    );
    state.counters["approx_rnet_ratio(%)"] = benchmark::Counter(
        100.0 * result.size() / provider.get_num_base_vecs()
    );
}
BENCHMARK(BM_LBGreedyVG)
    ->Name("LBGreedyVG")
    ->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_vertex_generator");
    program.add_description("Benchmark for vertex generators (RandomVG, MBGreedyVG, LBGreedyVG)");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(std::string("./datasets.json"))
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    // Algorithm parameters
    program.add_argument("-r", "--min-radius")
        .default_value(100.0f)
        .help("Minimum radius for approximate r-net construction")
        .scan<'g', float>();

    program.add_argument("-m", "--max-result-size")
        .default_value(10000u)
        .help("Maximum number of vertices in the result")
        .scan<'u', uint32_t>();

    // Batch-specific parameters
    program.add_argument("--small-batch-size")
        .default_value(64u)
        .help("Batch size for MBGreedyVG")
        .scan<'u', uint32_t>();

    program.add_argument("--large-batch-size")
        .default_value(512u)
        .help("Batch size for LBGreedyVG")
        .scan<'u', uint32_t>();

    program.add_argument("--term-thresh")
        .default_value(17u)
        .help("Termination threshold for LBGreedyVG")
        .scan<'u', uint32_t>();

    // Benchmark control
    program.add_argument("-i", "--iterations")
        .default_value(int64_t(5))
        .help("Number of times to run each benchmark")
        .scan<'d', int64_t>();

    program.add_argument("-h", "--help")
        .default_value(false)
        .implicit_value(true)
        .help("Show this help message and exit");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Check for help flag
    if (program.get<bool>("--help")) {
        std::cout << program << std::endl;
        std::cout << "\nExample usage:" << std::endl;
        std::cout << "  " << argv[0] << " -d sift-1m -r 100 -m 10000 -i 5" << std::endl;
        std::cout << "  " << argv[0] << " --dataset deep-1m --min-radius 50 --iterations 10" << std::endl;
        return 0;
    }

    // Load configuration
    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.min_radius = program.get<float>("--min-radius");
    g_config.max_result_size = program.get<uint32_t>("--max-result-size");
    g_config.small_batch_size = program.get<uint32_t>("--small-batch-size");
    g_config.large_batch_size = program.get<uint32_t>("--large-batch-size");
    g_config.term_thresh = program.get<uint32_t>("--term-thresh");
    g_config.iterations = program.get<int64_t>("--iterations");

    // Initialize data
    std::cout << "Loading dataset..." << std::endl;
    DataProvider::instance().init();

    // Print configuration
    std::cout << "\n=== Benchmark Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Num base vectors: " << DataProvider::instance().get_num_base_vecs() << std::endl;
    std::cout << "Vector dimension: " << DataProvider::instance().get_dim() << std::endl;
    std::cout << "Min radius: " << g_config.min_radius << std::endl;
    std::cout << "Max result size: " << g_config.max_result_size << std::endl;
    std::cout << "Small batch size: " << g_config.small_batch_size << std::endl;
    std::cout << "Large batch size: " << g_config.large_batch_size << std::endl;
    std::cout << "Term thresh: " << g_config.term_thresh << std::endl;
    std::cout << "Iterations: " << g_config.iterations << std::endl;
    std::cout << "================================\n" << std::endl;

    // Run benchmarks with repetitions
    // Pass --benchmark_repetitions to Google Benchmark
    std::string repetitions_flag = "--benchmark_repetitions=" + std::to_string(g_config.iterations);
    int benchmark_argc = 2;
    char* benchmark_argv[] = {
        argv[0],
        const_cast<char*>(repetitions_flag.c_str()),
        nullptr
    };
    ::benchmark::Initialize(&benchmark_argc, benchmark_argv);
    ::benchmark::RunSpecifiedBenchmarks();

    return 0;
}