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

using base_traits_t = BaseTraits<uint32_t, float>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using vg_traits_t = VertexGeneratorTraits<computer_traits_t>;
using dist_func_t = typename vg_traits_t::dist_func_t;
using vector_array_t = typename vg_traits_t::vector_array_t;
using lb_greedy_vg_t = typename vg_traits_t::lb_greedy_vg_t;

struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
    float min_radius;
    uint32_t max_result_size;

    // Mode 1: Manual parameters
    bool use_manual_mode;
    uint32_t batch_size;
    uint32_t term_thresh;

    // Mode 2: Auto-compute parameters
    float coverage_ratio;
    float confidence;

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

        // Always shuffle dataset
        ARTEA_INFO("Shuffling dataset...");
        dataset->shuffle_in_place();

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

// Benchmark for LBGreedyVG with auto-computed parameters
static void BM_LBGreedyVG(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    lb_greedy_vg_t generator(provider.get_dist_func());

    for (auto _ : state) {
        auto result = generator.generate(
            provider.get_base_vecs(),
            g_config.min_radius,
            g_config.max_result_size,
            g_config.coverage_ratio,
            g_config.confidence,
            g_config.batch_size
        );
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }

    // Report the actual result size in the last iteration
    auto result = generator.generate(
        provider.get_base_vecs(),
        g_config.min_radius,
        g_config.max_result_size,
        g_config.coverage_ratio,
        g_config.confidence,
        g_config.batch_size
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
    argparse::ArgumentParser program("bench_lb_greedy_vg");
    program.add_description("Benchmark for LBGreedyVG (Large-Batch Greedy Vertex Generator)");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    // Algorithm parameters
    program.add_argument("-r", "--min-radius")
        .default_value(90000.0f)
        .help("Minimum radius (squared) for approximate r-net construction. "
              "Use DistanceProber with quantile=0.05 for 95% coverage target.")
        .scan<'g', float>();

    program.add_argument("-m", "--max-result-size")
        .default_value(100000u)
        .help("Maximum number of vertices in the result. "
              "For 95% coverage, set to at least 5% of dataset size.")
        .scan<'u', uint32_t>();

    // Auto-compute parameters
    program.add_argument("--coverage-ratio")
        .default_value(0.95f)
        .scan<'g', float>()
        .help("Target coverage ratio (e.g., 0.95 for 95% coverage)");

    program.add_argument("--confidence")
        .default_value(0.96f)
        .scan<'g', float>()
        .help("Confidence level (e.g., 0.96 for 96% confidence)");

    program.add_argument("-b", "--batch-size")
        .default_value(512u)
        .scan<'u', uint32_t>()
        .help("Batch size for processing (default: 512)");

    // Benchmark control
    program.add_argument("-i", "--iterations")
        .default_value(int64_t(5))
        .help("Number of times to run each benchmark")
        .scan<'d', int64_t>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Load configuration
    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.min_radius = program.get<float>("--min-radius");
    g_config.max_result_size = program.get<uint32_t>("--max-result-size");
    g_config.coverage_ratio = program.get<float>("--coverage-ratio");
    g_config.confidence = program.get<float>("--confidence");
    g_config.batch_size = program.get<uint32_t>("--batch-size");
    g_config.iterations = program.get<int64_t>("--iterations");

    // Initialize data
    std::cout << "Loading dataset..." << std::endl;
    DataProvider::instance().init();

    // Compute term_thresh
    uint32_t computed_term_thresh = lb_greedy_vg_t::compute_term_thresh(
        g_config.coverage_ratio, g_config.confidence, g_config.batch_size
    );

    // Print configuration
    std::cout << "\n=== Benchmark Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Num base vectors: " << DataProvider::instance().get_num_base_vecs() << std::endl;
    std::cout << "Vector dimension: " << DataProvider::instance().get_dim() << std::endl;
    std::cout << "Min radius: " << g_config.min_radius << std::endl;
    std::cout << "Max result size: " << g_config.max_result_size << std::endl;
    std::cout << "Coverage ratio: " << g_config.coverage_ratio << std::endl;
    std::cout << "Confidence: " << g_config.confidence << std::endl;
    std::cout << "Batch size: " << g_config.batch_size << std::endl;
    std::cout << "Computed term thresh: " << computed_term_thresh << std::endl;
    std::cout << "Iterations: " << g_config.iterations << std::endl;
    std::cout << "================================\n" << std::endl;

    // Run benchmarks with repetitions
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