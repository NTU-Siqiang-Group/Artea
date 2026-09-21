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
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <fmt/format.h>
#include <experimental/simd>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstring>

using namespace artea;
using namespace artea::cpu;

// The SIMD distance functor is metric/dim-dependent now. Expose it as a
// <Metric, Dim, U> alias so the unroll factor U stays a benchmark template axis
// while (metric, padded-dim) — metric from --metric, padded dim from the
// loaded dataset — are resolved via infra_dispatch().
template <DistanceMetricsT Metric, vec_dim_t Dim, std::size_t U>
using artea_simd_dist_t = typename computer_traits_t<Metric, Dim>::template simd_dist_t<U>;

static constexpr uint32_t NUM_PAIRS = 65536;

struct TestConfig { std::string config_path, dataset_name, metric; } g_config;

class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }
    void init() {
        _dataset = std::make_unique<typename base_traits_t::vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        _dim = _dataset->get_base_vecs().get_vec_dim();
        _num_vecs = _dataset->get_base_vecs().get_num_vecs();

        // Resolve BOTH compile-time axes: metric from --metric, padded dim from the loaded dataset.
        _dataset_info = DatasetInfra{parse_metric(g_config.metric), _dim};

        random_seq_t rng_a;
        random_seq_t rng_b;
        _ids_a.resize(NUM_PAIRS);
        _ids_b.resize(NUM_PAIRS);
        rng_a.generate(_ids_a, _num_vecs, NUM_PAIRS);
        rng_b.generate(_ids_b, _num_vecs, NUM_PAIRS);
    }
    uint32_t get_dim() const { return _dim; }
    uint32_t get_num_vecs() const { return _num_vecs; }
    DatasetInfra get_dataset_info() const { return _dataset_info; }
    const float* get_vec(uint32_t id) const { return _dataset->get_base_vecs().get(id); }
    uint32_t get_id_a(uint32_t idx) const { return _ids_a[idx % NUM_PAIRS]; }
    uint32_t get_id_b(uint32_t idx) const { return _ids_b[idx % NUM_PAIRS]; }
private:
    std::unique_ptr<typename base_traits_t::vector_dataset_t> _dataset;
    uint32_t _dim;
    uint32_t _num_vecs;
    DatasetInfra _dataset_info{};
    std::vector<uint32_t> _ids_a;
    std::vector<uint32_t> _ids_b;
};

// ============================================================
// Single-thread benchmarks
// ============================================================

// 0. SimpleForLoop: scalar for-loop with #pragma simd hint
static float simple_L2sqr(const float* a, const float* b, uint32_t dim) {
    float result = 0.0f;
    #pragma omp simd reduction(+:result)
    for (uint32_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

static void BM_SimpleForLoop(benchmark::State& state) {
    auto& p = DataProvider::instance();
    uint32_t idx = 0;
    for (auto _ : state) {
        const float* q = p.get_vec(p.get_id_a(idx));
        const float* t = p.get_vec(p.get_id_b(idx));
        benchmark::DoNotOptimize(simple_L2sqr(q, t, p.get_dim()));
        ++idx;
    }
    state.SetItemsProcessed(state.iterations());
}

// 1. StdSimd (no tail processing, requires SIMD-aligned dim)
namespace stdx = std::experimental;
template <std::size_t U>
static float stdsimd_L2sqr(const float* a, const float* b, uint32_t dim) {
    using simd_t = stdx::native_simd<float>;
    constexpr std::size_t W = simd_t::size();
    constexpr std::size_t stride = W * U;
    simd_t sums[U] = {};
    std::size_t i = 0;
    for (; i + stride <= dim; i += stride) {
        for (std::size_t u = 0; u < U; ++u) {
            simd_t va(a + i + u * W, stdx::element_aligned);
            simd_t vb(b + i + u * W, stdx::element_aligned);
            simd_t diff = va - vb;
            sums[u] += diff * diff;
        }
    }
    for (; i + W <= dim; i += W) {
        simd_t va(a + i, stdx::element_aligned);
        simd_t vb(b + i, stdx::element_aligned);
        simd_t diff = va - vb;
        sums[0] += diff * diff;
    }
    for (std::size_t u = 1; u < U; ++u) sums[0] += sums[u];
    return stdx::reduce(sums[0]);
}

template <std::size_t U>
static void BM_StdSimd(benchmark::State& state) {
    auto& p = DataProvider::instance();
    uint32_t idx = 0;
    for (auto _ : state) {
        const float* q = p.get_vec(p.get_id_a(idx));
        const float* t = p.get_vec(p.get_id_b(idx));
        benchmark::DoNotOptimize(stdsimd_L2sqr<U>(q, t, p.get_dim()));
        ++idx;
    }
    state.SetItemsProcessed(state.iterations());
}

// 2. StdSimdTail (with scalar tail processing, supports any dim)
template <std::size_t U>
static float stdsimd_L2sqr_tail(const float* a, const float* b, uint32_t dim) {
    using simd_t = stdx::native_simd<float>;
    constexpr std::size_t W = simd_t::size();
    constexpr std::size_t stride = W * U;
    simd_t sums[U] = {};
    std::size_t i = 0;
    for (; i + stride <= dim; i += stride) {
        for (std::size_t u = 0; u < U; ++u) {
            simd_t va(a + i + u * W, stdx::element_aligned);
            simd_t vb(b + i + u * W, stdx::element_aligned);
            simd_t diff = va - vb;
            sums[u] += diff * diff;
        }
    }
    for (; i + W <= dim; i += W) {
        simd_t va(a + i, stdx::element_aligned);
        simd_t vb(b + i, stdx::element_aligned);
        simd_t diff = va - vb;
        sums[0] += diff * diff;
    }
    for (std::size_t u = 1; u < U; ++u) sums[0] += sums[u];
    float result = stdx::reduce(sums[0]);
    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

template <std::size_t U>
static void BM_StdSimdTail(benchmark::State& state) {
    auto& p = DataProvider::instance();
    uint32_t idx = 0;
    for (auto _ : state) {
        const float* q = p.get_vec(p.get_id_a(idx));
        const float* t = p.get_vec(p.get_id_b(idx));
        benchmark::DoNotOptimize(stdsimd_L2sqr_tail<U>(q, t, p.get_dim()));
        ++idx;
    }
    state.SetItemsProcessed(state.iterations());
}

// 3. Artea SIMDDistance (class wrapper, compile-time dim)
template <std::size_t U>
static void BM_Artea(benchmark::State& state) {
    auto& p = DataProvider::instance();
    infra_dispatch(p.get_dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        // Stateless functor: the dimension is a compile-time trait now.
        artea_simd_dist_t<Metric, Dim, U> func;
        uint32_t idx = 0;
        for (auto _ : state) {
            const float* q = p.get_vec(p.get_id_a(idx));
            const float* t = p.get_vec(p.get_id_b(idx));
            benchmark::DoNotOptimize(func(q, t));
            ++idx;
        }
    });
    state.SetItemsProcessed(state.iterations());
}

// ============================================================
// Parallel benchmarks (TBB)
// ============================================================

static constexpr uint32_t PARALLEL_BATCH = NUM_PAIRS;

static void BM_SimpleForLoop_Parallel(benchmark::State& state) {
    auto& p = DataProvider::instance();
    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, PARALLEL_BATCH),
            [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t i = r.begin(); i != r.end(); ++i) {
                    const float* q = p.get_vec(p.get_id_a(i));
                    const float* t = p.get_vec(p.get_id_b(i));
                    benchmark::DoNotOptimize(simple_L2sqr(q, t, p.get_dim()));
                }
            }
        );
    }
    state.SetItemsProcessed(state.iterations() * PARALLEL_BATCH);
}

template <std::size_t U>
static void BM_StdSimd_Parallel(benchmark::State& state) {
    auto& p = DataProvider::instance();
    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, PARALLEL_BATCH),
            [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t i = r.begin(); i != r.end(); ++i) {
                    const float* q = p.get_vec(p.get_id_a(i));
                    const float* t = p.get_vec(p.get_id_b(i));
                    benchmark::DoNotOptimize(stdsimd_L2sqr<U>(q, t, p.get_dim()));
                }
            }
        );
    }
    state.SetItemsProcessed(state.iterations() * PARALLEL_BATCH);
}

template <std::size_t U>
static void BM_StdSimdTail_Parallel(benchmark::State& state) {
    auto& p = DataProvider::instance();
    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, PARALLEL_BATCH),
            [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t i = r.begin(); i != r.end(); ++i) {
                    const float* q = p.get_vec(p.get_id_a(i));
                    const float* t = p.get_vec(p.get_id_b(i));
                    benchmark::DoNotOptimize(stdsimd_L2sqr_tail<U>(q, t, p.get_dim()));
                }
            }
        );
    }
    state.SetItemsProcessed(state.iterations() * PARALLEL_BATCH);
}

template <std::size_t U>
static void BM_Artea_Parallel(benchmark::State& state) {
    auto& p = DataProvider::instance();
    infra_dispatch(p.get_dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        for (auto _ : state) {
            tbb::parallel_for(
                tbb::blocked_range<uint32_t>(0, PARALLEL_BATCH),
                [&](const tbb::blocked_range<uint32_t>& r) {
                    // Stateless functor: the dimension is a compile-time trait now.
                    artea_simd_dist_t<Metric, Dim, U> func;
                    for (uint32_t i = r.begin(); i != r.end(); ++i) {
                        const float* q = p.get_vec(p.get_id_a(i));
                        const float* t = p.get_vec(p.get_id_b(i));
                        benchmark::DoNotOptimize(func(q, t));
                    }
                }
            );
        }
    });
    state.SetItemsProcessed(state.iterations() * PARALLEL_BATCH);
}

template <std::size_t U>
static void register_unrolled_benchmarks(DistanceMetricsT metric) {
    const auto name = metric_name(metric);
    benchmark::RegisterBenchmark(fmt::format("Artea_{}_U{}", name, U), BM_Artea<U>);
    benchmark::RegisterBenchmark(fmt::format("Par_Artea_{}_U{}", name, U), BM_Artea_Parallel<U>)
        ->UseRealTime();

    // These reference kernels only implement squared Euclidean distance.
    if (metric == DistanceMetricsT::EUCLIDEAN_SQR) {
        benchmark::RegisterBenchmark(fmt::format("StdSimd_{}_U{}", name, U), BM_StdSimd<U>);
        benchmark::RegisterBenchmark(fmt::format("StdSimdTail_{}_U{}", name, U), BM_StdSimdTail<U>);
        benchmark::RegisterBenchmark(fmt::format("Par_StdSimd_{}_U{}", name, U), BM_StdSimd_Parallel<U>)
            ->UseRealTime();
        benchmark::RegisterBenchmark(fmt::format("Par_StdSimdTail_{}_U{}", name, U), BM_StdSimdTail_Parallel<U>)
            ->UseRealTime();
    }
}

static void register_benchmarks(DistanceMetricsT metric) {
    if (metric == DistanceMetricsT::EUCLIDEAN_SQR) {
        benchmark::RegisterBenchmark("SimpleForLoop_euclidean_sqr", BM_SimpleForLoop);
        benchmark::RegisterBenchmark("Par_SimpleForLoop_euclidean_sqr", BM_SimpleForLoop_Parallel)
            ->UseRealTime();
    }
    register_unrolled_benchmarks<1>(metric);
    register_unrolled_benchmarks<2>(metric);
    register_unrolled_benchmarks<4>(metric);
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_simd_distance");
    program.add_argument("-c", "--config").default_value(artea::default_dataset_config_path());
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--metric").default_value(std::string("euclidean_sqr"))
        .help("Distance metric: 'euclidean' ('l2'), 'euclidean_sqr' ('l2_sqr'), 'inner_product', or 'cosine'");
    std::vector<std::string> benchmark_args;
    try {
        benchmark_args = program.parse_known_args(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n' << program;
        return 1;
    }
    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.metric = program.get<std::string>("--metric");
    // Forward only Google Benchmark flags after consuming the dataset/metric options.
    benchmark_args.insert(benchmark_args.begin(), argv[0]);
    std::vector<char*> benchmark_argv;
    for (auto& arg : benchmark_args) benchmark_argv.push_back(arg.data());
    int benchmark_argc = static_cast<int>(benchmark_argv.size());
    benchmark_argv.push_back(nullptr);
    ::benchmark::Initialize(&benchmark_argc, benchmark_argv.data());
    if (::benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_argv.data())) return 1;

    DataProvider::instance().init();
    register_benchmarks(DataProvider::instance().get_dataset_info().metric);
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
