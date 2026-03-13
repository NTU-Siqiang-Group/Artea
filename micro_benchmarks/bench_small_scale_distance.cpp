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

/*
 * @FilePath: /Artea/micro_benchmark/playground/bench_small_scale_distance.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2026-03-13
 * @Description: Benchmark comparing serial vs TBB parallel distance computation
 *               for small-scale vector arrays (32/48/64/96/128 vectors) with
 *               random sparse memory access patterns.
 */

#include <benchmark/benchmark.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <vector>
#include <random>
#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>

using namespace artea;
using namespace artea::cpu;

// Type definitions
using vec_num_t = uint32_t;
using vec_ele_t = float;
using vec_id_t = uint32_t;
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using vector_array_t = typename base_traits_t::vector_array_t;
using vector_dataset_t = typename base_traits_t::vector_dataset_t;
using dist_func_t = typename computer_traits_t::dist_func_t;
using distance_t = typename base_traits_t::distance_t;
using random_seq_t = typename base_traits_t::random_seq_t;

// Global configuration
struct BenchConfig {
    std::string config_path;
    std::string dataset_name;
} g_bench_config;

// Global data provider (singleton pattern)
class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init() {
        auto dataset = std::make_unique<vector_dataset_t>(
            g_bench_config.config_path,
            g_bench_config.dataset_name
        );
        base_vecs_ = &dataset->get_base_vecs();
        dim_ = base_vecs_->get_vec_dim();
        query_vec_.resize(dim_);
        std::memcpy(query_vec_.data(), dataset->get_query_vecs().get(0), dim_ * sizeof(vec_ele_t));
        dataset_ = std::move(dataset);
    }

    const vector_array_t& get_base_vecs() const { return *base_vecs_; }
    const vec_ele_t* get_query_vec() const { return query_vec_.data(); }
    vec_num_t get_dim() const { return dim_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    const vector_array_t* base_vecs_ = nullptr;
    vec_num_t dim_ = 0;
    std::vector<vec_ele_t> query_vec_;
};

// Serial distance computation
static void BM_Serial_Distance(benchmark::State& state) {
    const vec_num_t num_vecs = state.range(0);
    const vec_num_t dim = state.range(1);

    auto& provider = DataProvider::instance();
    const auto& base_vecs = provider.get_base_vecs();
    const vec_ele_t* query_vec = provider.get_query_vec();
    dist_func_t dist_func(dim);
    std::vector<distance_t> distances(num_vecs);

    // Generate random IDs outside timing region
    random_seq_t random_seq(base_vecs.get_num_vecs());
    std::vector<vec_id_t> random_ids(num_vecs);
    random_seq.generate(random_ids, num_vecs);

    for (auto _ : state) {
        for (vec_num_t i = 0; i < num_vecs; ++i) {
            distances[i] = dist_func(query_vec, base_vecs.get(random_ids[i]));
        }
        benchmark::DoNotOptimize(distances.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vecs);
    state.SetBytesProcessed(state.iterations() * num_vecs * dim * sizeof(vec_ele_t) * 2);
}

// TBB parallel_for with static partitioner
static void BM_TBB_Static_Distance(benchmark::State& state) {
    const vec_num_t num_vecs = state.range(0);
    const vec_num_t dim = state.range(1);

    auto& provider = DataProvider::instance();
    const auto& base_vecs = provider.get_base_vecs();
    const vec_ele_t* query_vec = provider.get_query_vec();
    dist_func_t dist_func(dim);
    std::vector<distance_t> distances(num_vecs);

    // Generate random IDs outside timing region
    random_seq_t random_seq(base_vecs.get_num_vecs());
    std::vector<vec_id_t> random_ids(num_vecs);
    random_seq.generate(random_ids, num_vecs);

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<vec_num_t>(0, num_vecs),
            [&](const tbb::blocked_range<vec_num_t>& r) {
                for (vec_num_t i = r.begin(); i != r.end(); ++i) {
                    distances[i] = dist_func(query_vec, base_vecs.get(random_ids[i]));
                }
            },
            tbb::static_partitioner()
        );
        benchmark::DoNotOptimize(distances.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vecs);
    state.SetBytesProcessed(state.iterations() * num_vecs * dim * sizeof(vec_ele_t) * 2);
}

// TBB parallel_for with auto partitioner (for comparison)
static void BM_TBB_Auto_Distance(benchmark::State& state) {
    const vec_num_t num_vecs = state.range(0);
    const vec_num_t dim = state.range(1);

    auto& provider = DataProvider::instance();
    const auto& base_vecs = provider.get_base_vecs();
    const vec_ele_t* query_vec = provider.get_query_vec();
    dist_func_t dist_func(dim);
    std::vector<distance_t> distances(num_vecs);

    // Generate random IDs outside timing region
    random_seq_t random_seq(base_vecs.get_num_vecs());
    std::vector<vec_id_t> random_ids(num_vecs);
    random_seq.generate(random_ids, num_vecs);

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<vec_num_t>(0, num_vecs),
            [&](const tbb::blocked_range<vec_num_t>& r) {
                for (vec_num_t i = r.begin(); i != r.end(); ++i) {
                    distances[i] = dist_func(query_vec, base_vecs.get(random_ids[i]));
                }
            },
            tbb::auto_partitioner()
        );
        benchmark::DoNotOptimize(distances.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * num_vecs);
    state.SetBytesProcessed(state.iterations() * num_vecs * dim * sizeof(vec_ele_t) * 2);
}

// Register benchmarks for different vector counts and dimensions
// Format: ->Args({num_vecs, dim})

// 32 vectors
BENCHMARK(BM_Serial_Distance)->Args({32, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Static_Distance)->Args({32, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Auto_Distance)->Args({32, 128})->Unit(benchmark::kMicrosecond);

// 48 vectors
BENCHMARK(BM_Serial_Distance)->Args({48, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Static_Distance)->Args({48, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Auto_Distance)->Args({48, 128})->Unit(benchmark::kMicrosecond);

// 64 vectors
BENCHMARK(BM_Serial_Distance)->Args({64, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Static_Distance)->Args({64, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Auto_Distance)->Args({64, 128})->Unit(benchmark::kMicrosecond);

// 96 vectors
BENCHMARK(BM_Serial_Distance)->Args({96, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Static_Distance)->Args({96, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Auto_Distance)->Args({96, 128})->Unit(benchmark::kMicrosecond);

// 128 vectors
BENCHMARK(BM_Serial_Distance)->Args({128, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Static_Distance)->Args({128, 128})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Auto_Distance)->Args({128, 128})->Unit(benchmark::kMicrosecond);

// Additional dimension tests for 64 vectors (common case)
BENCHMARK(BM_Serial_Distance)->Args({64, 64})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Static_Distance)->Args({64, 64})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Auto_Distance)->Args({64, 64})->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Serial_Distance)->Args({64, 256})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Static_Distance)->Args({64, 256})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Auto_Distance)->Args({64, 256})->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Serial_Distance)->Args({64, 512})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Static_Distance)->Args({64, 512})->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_TBB_Auto_Distance)->Args({64, 512})->Unit(benchmark::kMicrosecond);

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("bench_small_scale_distance");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    try { program.parse_args(argc, argv); } catch (...) { return 1; }
    g_bench_config.config_path = program.get<std::string>("--config");
    g_bench_config.dataset_name = program.get<std::string>("--dataset");

    // Initialize data provider (load dataset once)
    DataProvider::instance().init();

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
