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
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <artea/cpu/vertex_generator/random_vg.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <filesystem>
#include <memory>

using namespace artea;
using namespace artea::cpu;

// Global configuration
struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
} g_config;

// Global dataset
std::unique_ptr<vector_dataset_t> g_dataset = nullptr;
const vector_array_t* g_vecs_data = nullptr;
uint32_t g_num_vecs = 0;
uint32_t g_vec_dim = 0;

// Benchmark: Small sample size (1% of dataset)
static void BM_RandomVG_SmallSample(benchmark::State& state) {
    random_vg_t random_vg;
    uint32_t result_size = std::max(100u, g_num_vecs / 100);

    for (auto _ : state) {
        auto result = random_vg.generate(*g_vecs_data, result_size);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * result_size);
    state.counters["result_size"] = result_size;
    state.counters["num_vecs"] = g_num_vecs;
    state.counters["vec_dim"] = g_vec_dim;
}
BENCHMARK(BM_RandomVG_SmallSample)->Unit(benchmark::kMillisecond);

// Benchmark: Medium sample size (10% of dataset)
static void BM_RandomVG_MediumSample(benchmark::State& state) {
    random_vg_t random_vg;
    uint32_t result_size = g_num_vecs / 10;

    for (auto _ : state) {
        auto result = random_vg.generate(*g_vecs_data, result_size);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * result_size);
    state.counters["result_size"] = result_size;
    state.counters["num_vecs"] = g_num_vecs;
    state.counters["vec_dim"] = g_vec_dim;
}
BENCHMARK(BM_RandomVG_MediumSample)->Unit(benchmark::kMillisecond);

// Benchmark: Large sample size (50% of dataset)
static void BM_RandomVG_LargeSample(benchmark::State& state) {
    random_vg_t random_vg;
    uint32_t result_size = g_num_vecs / 2;

    for (auto _ : state) {
        auto result = random_vg.generate(*g_vecs_data, result_size);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * result_size);
    state.counters["result_size"] = result_size;
    state.counters["num_vecs"] = g_num_vecs;
    state.counters["vec_dim"] = g_vec_dim;
}
BENCHMARK(BM_RandomVG_LargeSample)->Unit(benchmark::kMillisecond);

// Benchmark: Very large sample size (90% of dataset)
static void BM_RandomVG_VeryLargeSample(benchmark::State& state) {
    random_vg_t random_vg;
    uint32_t result_size = static_cast<uint32_t>(g_num_vecs * 0.9);

    for (auto _ : state) {
        auto result = random_vg.generate(*g_vecs_data, result_size);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * result_size);
    state.counters["result_size"] = result_size;
    state.counters["num_vecs"] = g_num_vecs;
    state.counters["vec_dim"] = g_vec_dim;
}
BENCHMARK(BM_RandomVG_VeryLargeSample)->Unit(benchmark::kMillisecond);

// Benchmark: Varying sample sizes
static void BM_RandomVG_VaryingSizes(benchmark::State& state) {
    random_vg_t random_vg;
    uint32_t result_size = state.range(0);

    for (auto _ : state) {
        auto result = random_vg.generate(*g_vecs_data, result_size);
        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * result_size);
    state.counters["result_size"] = result_size;
    state.counters["num_vecs"] = g_num_vecs;
    state.counters["vec_dim"] = g_vec_dim;
}
BENCHMARK(BM_RandomVG_VaryingSizes)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1000)
    ->Arg(5000)
    ->Arg(10000)
    ->Arg(50000)
    ->Arg(100000);

// Benchmark: ID generation only (measure overhead of sorting)
static void BM_RandomVG_IDGeneration(benchmark::State& state) {
    uint32_t result_size = state.range(0);

    for (auto _ : state) {
        random_seq_nr_t random_seq_nr;
        auto random_ids = random_seq_nr.generate(result_size, g_num_vecs);

        std::vector<uint32_t> vec_ids;
        vec_ids.insert(vec_ids.end(), random_ids.begin(), random_ids.end());
        std::sort(vec_ids.begin(), vec_ids.end());

        benchmark::DoNotOptimize(vec_ids);
    }

    state.SetItemsProcessed(state.iterations() * result_size);
    state.counters["result_size"] = result_size;
}
BENCHMARK(BM_RandomVG_IDGeneration)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(50000)
    ->Arg(100000);

// Benchmark: Vector extraction only (measure extract_subset performance)
static void BM_RandomVG_VectorExtraction(benchmark::State& state) {
    uint32_t result_size = state.range(0);

    // Pre-generate sorted IDs
    random_seq_nr_t random_seq_nr;
    auto random_ids = random_seq_nr.generate(result_size, g_num_vecs);
    std::vector<uint32_t> vec_ids;
    vec_ids.insert(vec_ids.end(), random_ids.begin(), random_ids.end());
    std::sort(vec_ids.begin(), vec_ids.end());

    for (auto _ : state) {
        auto extracted = g_vecs_data->extract_subset(vec_ids);
        benchmark::DoNotOptimize(extracted);
    }

    state.SetItemsProcessed(state.iterations() * result_size);
    state.counters["result_size"] = result_size;
    state.counters["vec_dim"] = g_vec_dim;
}
BENCHMARK(BM_RandomVG_VectorExtraction)
    ->Unit(benchmark::kMillisecond)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(50000)
    ->Arg(100000);

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_random_vg");
    program.add_argument("-c", "--config")
        .help("Path to the dataset configuration JSON file")
        .default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset")
        .help("Name of the dataset to use")
        .default_value(std::string("sift-1m"));

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");

    // Load dataset
    if (!std::filesystem::exists(g_config.config_path)) {
        std::cerr << "Config file not found: " << g_config.config_path << std::endl;
        return 1;
    }

    ARTEA_INFO(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
    g_dataset = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);

    g_vecs_data = &g_dataset->get_base_vecs();
    g_num_vecs = g_vecs_data->get_num_vecs();
    g_vec_dim = g_vecs_data->get_vec_dim();

    ARTEA_INFO(fmt::format("Dataset loaded: {} vectors, {} dimensions", g_num_vecs, g_vec_dim));

    // Run benchmarks
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    return 0;
}
