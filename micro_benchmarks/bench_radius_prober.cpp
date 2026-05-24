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
using vec_num_t = typename base_traits_t::vec_num_t;
using simd_dispatcher_t = SIMDDistanceDispatcher<computer_traits_t, 1>;
using vector_array_t = typename computer_traits_t::vector_array_t;
using vector_dataset_t = typename computer_traits_t::vector_dataset_t;
template <typename DistFuncT>
using distance_prober_t = typename computer_traits_t::template distance_prober_t<DistFuncT>;

struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
    int num_dists_sampled;
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
        dispatcher_ = std::make_unique<simd_dispatcher_t>(dim_);
    }

    vec_num_t get_dim() const { return dim_; }
    vec_num_t get_num_base_vecs() const { return num_base_vecs_; }
    const vector_array_t& get_base_vecs() const { return base_vecs_; }
    const simd_dispatcher_t& get_dispatcher() const { return *dispatcher_; }

private:
    vec_num_t dim_;
    vec_num_t num_base_vecs_;
    vector_array_t base_vecs_;
    std::unique_ptr<simd_dispatcher_t> dispatcher_;
};

// Benchmark for DistanceProber
static void BM_DistanceProber(benchmark::State& state) {
    auto& provider = DataProvider::instance();
    const auto& base_vecs = provider.get_base_vecs();
    auto& dispatcher      = provider.get_dispatcher();

    vec_num_t num_distances = g_config.num_dists_sampled;

    dispatcher.dispatch([&](const auto& dist_func) {
        using DistFunc = std::decay_t<decltype(dist_func)>;
        distance_prober_t<DistFunc> prober(dist_func);
        for (auto _ : state) {
            auto result = prober.probe(base_vecs, g_config.quantile, num_distances);
            benchmark::DoNotOptimize(result);
        }
        state.SetItemsProcessed(state.iterations());
    });
}

BENCHMARK(BM_DistanceProber)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(g_config.iterations);

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_radius_prober");
    program.add_description("Benchmark for DistanceProber utility");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(artea::default_dataset_config_path())
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    // Algorithm parameters
    program.add_argument("-m", "--num-dists")
        .default_value(100000)
        .scan<'i', int>()
        .help("Number of independent distance samples");

    program.add_argument("-q", "--quantile")
        .default_value(0.001f)
        .scan<'g', float>()
        .help("Target quantile (e.g., 0.0005 for 0.05%, 0.001 for 0.1%, 0.01 for 1%)");

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
    g_config.num_dists_sampled = program.get<int>("--num-dists");
    g_config.quantile = program.get<float>("--quantile");
    g_config.iterations = program.get<int64_t>("--iterations");

    ARTEA_INFO(fmt::format("Benchmark Configuration:"));
    ARTEA_INFO(fmt::format("  Dataset: {}", g_config.dataset_name));
    ARTEA_INFO(fmt::format("  Config path: {}", g_config.config_path));
    ARTEA_INFO(fmt::format("  Num distance samples: {}", g_config.num_dists_sampled));
    ARTEA_INFO(fmt::format("  Quantile: {:.4f}", g_config.quantile));
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
