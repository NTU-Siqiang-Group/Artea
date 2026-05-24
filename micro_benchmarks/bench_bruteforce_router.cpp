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
 * @FilePath: /Artea/micro_benchmarks/bench_bruteforce_router.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Single-query micro-benchmark for BruteforceRouter,
 *               comparing plain L2 (BruteforceRouter::query) against
 *               FastL2 (BruteforceRouter::query_fast). Each iteration
 *               runs one query; queries cycle through the query set
 *               (idx % num_queries) so Google Benchmark can run as
 *               many iterations as needed for the measurement to
 *               converge. Default dataset: crawl.
 */

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
} g_config;

// Singleton DataProvider: loads the dataset and primes the FastL2
// per-base ||p||^2 cache once for the whole process.
class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init() {
        _dataset = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        _dist_func = std::make_unique<dist_func_t>(_dataset->get_base_vecs().get_vec_dim());
        // FastL2 needs the precomputed per-base norms — pay for them once.
        _dataset->enable_fast_L2();
    }

    const vector_dataset_t& dataset() const { return *_dataset; }
    const dist_func_t& dist_func() const { return *_dist_func; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> _dataset;
    std::unique_ptr<dist_func_t> _dist_func;
};

// Bruteforce visits every base vector regardless of topk, so the
// distance-kernel timing dominates and is essentially flat in topk.
// Benchmark top-1 only.
static constexpr uint32_t TOPK = 1;

// ----------------------------------------------------------------------
// Benchmark: plain L2 — single query per iteration, cycling.
// ----------------------------------------------------------------------
static void BM_BruteforceL2(benchmark::State& state) {
    auto& p = DataProvider::instance();
    const auto& base_vecs  = p.dataset().get_base_vecs();
    const auto& query_vecs = p.dataset().get_query_vecs();
    const auto& dist_func  = p.dist_func();
    const uint32_t num_queries = query_vecs.get_num_vecs();

    bruteforce_router_t router(base_vecs, dist_func, TOPK);
    router.initialize();

    uint64_t idx = 0;
    for (auto _ : state) {
        const vec_ele_t* q = query_vecs.get(idx % num_queries);
        auto result = router.query(q);
        benchmark::DoNotOptimize(result);
        ++idx;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BruteforceL2)
    ->Name("Bruteforce_L2_top1")
    ->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------------
// Benchmark: FastL2 — single query per iteration, cycling.
// Uses BruteforceRouter::query_fast with the precomputed base norms
// (-2 * <p,q> + ||p||^2 — ranking-equivalent to L2 within one query).
// ----------------------------------------------------------------------
static void BM_BruteforceFastL2(benchmark::State& state) {
    auto& p = DataProvider::instance();
    const auto& base_vecs  = p.dataset().get_base_vecs();
    const auto& query_vecs = p.dataset().get_query_vecs();
    const auto& base_norms = p.dataset().get_base_norms();
    const auto& dist_func  = p.dist_func();
    const uint32_t num_queries = query_vecs.get_num_vecs();

    bruteforce_router_t router(base_vecs, dist_func, TOPK);
    router.initialize();

    uint64_t idx = 0;
    for (auto _ : state) {
        const vec_ele_t* q = query_vecs.get(idx % num_queries);
        auto result = router.query_fast(q, base_norms);
        benchmark::DoNotOptimize(result);
        ++idx;
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_BruteforceFastL2)
    ->Name("Bruteforce_FastL2_top1")
    ->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------------
// Cache-resident variants: extract_subset(0, N) shrinks the base so
// the working set fits in some level of cache. As N grows from L1 → L2
// → L3 → DRAM you should see FastL2's ALU saving (~25–35%) progressively
// hidden by memory traffic. Pick sizes that bracket your L3.
//
// extract_subset(0, N) preserves the first N vectors' order, so the
// router can index the full dataset.get_base_norms()[vid] for vid<N —
// no need to recompute a smaller norms cache.
// ----------------------------------------------------------------------

static void BM_BruteforceL2_SmallBase(benchmark::State& state) {
    auto& p = DataProvider::instance();
    const auto& full_base  = p.dataset().get_base_vecs();
    const auto& query_vecs = p.dataset().get_query_vecs();
    const auto& dist_func  = p.dist_func();
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    const uint32_t num_queries = query_vecs.get_num_vecs();

    auto small_base = full_base.extract_subset(0, N);
    bruteforce_router_t router(small_base, dist_func, TOPK);
    router.initialize();

    uint64_t idx = 0;
    for (auto _ : state) {
        const vec_ele_t* q = query_vecs.get(idx % num_queries);
        auto result = router.query(q);
        benchmark::DoNotOptimize(result);
        ++idx;
    }
    state.SetItemsProcessed(state.iterations());
    // Bytes-per-iteration: N base_vecs * dim * sizeof(float)
    state.SetBytesProcessed(
        int64_t(state.iterations()) * N * full_base.get_vec_dim() * sizeof(float));
}
BENCHMARK(BM_BruteforceL2_SmallBase)
    ->Name("Bruteforce_L2_top1_smallbase")
    ->Arg(1024)->Arg(8192)->Arg(65536)->Arg(524288)
    ->ArgName("base_size")
    ->Unit(benchmark::kMicrosecond);

static void BM_BruteforceFastL2_SmallBase(benchmark::State& state) {
    auto& p = DataProvider::instance();
    const auto& full_base  = p.dataset().get_base_vecs();
    const auto& query_vecs = p.dataset().get_query_vecs();
    const auto& base_norms = p.dataset().get_base_norms();
    const auto& dist_func  = p.dist_func();
    const uint32_t N = static_cast<uint32_t>(state.range(0));
    const uint32_t num_queries = query_vecs.get_num_vecs();

    auto small_base = full_base.extract_subset(0, N);
    bruteforce_router_t router(small_base, dist_func, TOPK);
    router.initialize();

    uint64_t idx = 0;
    for (auto _ : state) {
        const vec_ele_t* q = query_vecs.get(idx % num_queries);
        // base_norms[0..N) matches small_base's vid order because
        // extract_subset(0, N) preserves the first N vectors.
        auto result = router.query_fast(q, base_norms);
        benchmark::DoNotOptimize(result);
        ++idx;
    }
    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(
        int64_t(state.iterations()) * N * full_base.get_vec_dim() * sizeof(float));
}
BENCHMARK(BM_BruteforceFastL2_SmallBase)
    ->Name("Bruteforce_FastL2_top1_smallbase")
    ->Arg(1024)->Arg(8192)->Arg(65536)->Arg(524288)
    ->ArgName("base_size")
    ->Unit(benchmark::kMicrosecond);

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_bruteforce_router");
    program.add_argument("-c", "--config").default_value(artea::default_dataset_config_path());
    program.add_argument("-d", "--dataset").default_value(std::string("crawl"));

    // parse_known_args extracts -c/-d and returns everything else
    // (incl. --benchmark_*) for google benchmark to consume.
    std::vector<std::string> remaining;
    try {
        remaining = program.parse_known_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    g_config.config_path  = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    DataProvider::instance().init();

    // Rebuild (argc, argv) for benchmark::Initialize from the remaining args.
    std::vector<char*> bench_argv;
    bench_argv.reserve(remaining.size() + 1);
    bench_argv.push_back(argv[0]);
    for (auto& s : remaining) bench_argv.push_back(s.data());
    int bench_argc = static_cast<int>(bench_argv.size());

    ::benchmark::Initialize(&bench_argc, bench_argv.data());
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
