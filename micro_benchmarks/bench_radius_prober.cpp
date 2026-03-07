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
using vec_num_t = typename base_traits_t::vec_num_t;
using dist_func_t = typename computer_traits_t::dist_func_t;
using vector_array_t = typename computer_traits_t::vector_array_t;
using vector_dataset_t = typename computer_traits_t::vector_dataset_t;
using radius_prober_t = typename computer_traits_t::radius_prober_t;

struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
    int num_samples;
    float quantile;
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

        // Use all base vectors from dataset
        base_vecs_ = std::move(dataset->get_base_vecs());

        // Initialize distance function
        dist_func_ = std::make_unique<dist_func_t>(dim_);
    }

    vec_num_t get_dim() const { return dim_; }
    vec_num_t get_num_base_vecs() const { return num_base_vecs_; }
    const vector_array_t& get_base_vecs() const { return base_vecs_; }
    const dist_func_t& get_dist_func() const { return *dist_func_; }

private:
    vec_num_t dim_;
    vec_num_t num_base_vecs_;
    vector_array_t base_vecs_;
    std::unique_ptr<dist_func_t> dist_func_;
};

// Benchmark for RadiusProber
static void BM_RadiusProber(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& base_vecs = provider.get_base_vecs();
    const auto& dist_func = provider.get_dist_func();

    vec_num_t num_vecs = std::min(g_config.num_samples, static_cast<int>(base_vecs.get_num_vecs()));

    radius_prober_t prober(dist_func);

    for (auto _ : state) {
        auto result = prober.probe(base_vecs, g_config.quantile, num_vecs);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_RadiusProber)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(g_config.iterations);

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_radius_prober");
    program.add_description("Benchmark for RadiusProber utility");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    // Algorithm parameters
    program.add_argument("-n", "--num-samples")
        .default_value(1000)
        .scan<'i', int>()
        .help("Number of vectors to sample from the dataset");

    program.add_argument("-q", "--quantile")
        .default_value(0.001f)
        .scan<'g', float>()
        .help("Target quantile (e.g., 0.001 for 0.1%, 0.01 for 1%, 0.05 for 5%)");

    // Benchmark control
    program.add_argument("-i", "--iterations")
        .default_value(int64_t(10))
        .scan<'i', int64_t>()
        .help("Maximum number of iterations for benchmarks");

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
    g_config.num_samples = program.get<int>("--num-samples");
    g_config.quantile = program.get<float>("--quantile");
    g_config.iterations = program.get<int64_t>("--iterations");

    logger.info(fmt::format("Benchmark Configuration:"));
    logger.info(fmt::format("  Dataset: {}", g_config.dataset_name));
    logger.info(fmt::format("  Config path: {}", g_config.config_path));
    logger.info(fmt::format("  Num samples: {}", g_config.num_samples));
    logger.info(fmt::format("  Quantile: {:.4f}", g_config.quantile));
    logger.info(fmt::format("  Iterations: {}", g_config.iterations));

    DataProvider::instance().init();

    logger.info(fmt::format("Dataset loaded:"));
    logger.info(fmt::format("  Dimension: {}", DataProvider::instance().get_dim()));
    logger.info(fmt::format("  Base vectors: {}", DataProvider::instance().get_num_base_vecs()));

    // Initialize and run Google Benchmark
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
