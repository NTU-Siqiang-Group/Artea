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
#include <faiss/utils/distances.h>
#include <hnswlib/hnswlib.h>
#include <numkong/spatial.h>
#include <experimental/simd>
#include <memory>
#include <vector>
#include <cstring>

using namespace artea;
using namespace artea::cpu;

using base_traits_t = BaseTraits<uint32_t, float>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
template <std::size_t U> using artea_simd_dist_t = computer_traits_t::template simd_dist_t<U>;

struct TestConfig { std::string config_path, dataset_name; } g_config;

class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }
    void init() {
        auto dataset = std::make_unique<typename base_traits_t::vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dim_ = dataset->get_base_vecs().get_vec_dim();
        query_vec_.resize(dim_);
        target_vec_.resize(dim_); // For benchmark, we just need one hot pair usually, or could use array
        std::memcpy(query_vec_.data(), dataset->get_query_vecs().get(0), dim_ * sizeof(float));
        std::memcpy(target_vec_.data(), dataset->get_base_vecs().get(0), dim_ * sizeof(float));
    }
    uint32_t get_dim() const { return dim_; }
    float* get_q() { return query_vec_.data(); }
    float* get_t() { return target_vec_.data(); }
private:
    uint32_t dim_;
    std::vector<float> query_vec_, target_vec_;
};

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
    for (auto _ : state) {
        benchmark::DoNotOptimize(simple_L2sqr(p.get_q(), p.get_t(), p.get_dim()));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_SimpleForLoop)->Name("SimpleForLoop_L2");

// 1. StdSimd: std::experimental::simd L2 squared distance with unroll factor U
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
    // Reduce remaining full SIMD lanes
    for (; i + W <= dim; i += W) {
        simd_t va(a + i, stdx::element_aligned);
        simd_t vb(b + i, stdx::element_aligned);
        simd_t diff = va - vb;
        sums[0] += diff * diff;
    }
    // Merge accumulators and scalar tail
    for (std::size_t u = 1; u < U; ++u) sums[0] += sums[u];
    return stdx::reduce(sums[0]);
}

template <std::size_t U>
static void BM_StdSimd(benchmark::State& state) {
    auto& p = DataProvider::instance();
    for (auto _ : state) {
        benchmark::DoNotOptimize(stdsimd_L2sqr<U>(p.get_q(), p.get_t(), p.get_dim()));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_TEMPLATE(BM_StdSimd, 1)->Name("StdSimd_L2_U1");
BENCHMARK_TEMPLATE(BM_StdSimd, 2)->Name("StdSimd_L2_U2");
BENCHMARK_TEMPLATE(BM_StdSimd, 4)->Name("StdSimd_L2_U4");

// 1. Artea Benchmarks (1, 2, 4)
template <std::size_t U>
static void BM_Artea(benchmark::State& state) {
    auto& p = DataProvider::instance();
    artea_simd_dist_t<U> func(p.get_dim());
    for (auto _ : state) {
        benchmark::DoNotOptimize(func(p.get_q(), p.get_t()));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK_TEMPLATE(BM_Artea, 1)->Name("Artea_L2_U1");
BENCHMARK_TEMPLATE(BM_Artea, 2)->Name("Artea_L2_U2");
BENCHMARK_TEMPLATE(BM_Artea, 4)->Name("Artea_L2_U4");

// 2. Faiss
static void BM_Faiss(benchmark::State& state) {
    auto& p = DataProvider::instance();
    for (auto _ : state) {
        benchmark::DoNotOptimize(faiss::fvec_L2sqr(p.get_q(), p.get_t(), p.get_dim()));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Faiss)->Name("Faiss_L2_fvec");

// 3. HNSWLib
static void BM_HNSWLib(benchmark::State& state) {
    auto& p = DataProvider::instance();
    hnswlib::L2Space space(p.get_dim());
    auto f = space.get_dist_func();
    void* param = space.get_dist_func_param();
    for (auto _ : state) {
        benchmark::DoNotOptimize(f(p.get_q(), p.get_t(), param));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HNSWLib)->Name("HNSWLib_L2");

// 4. NumKong
static void BM_NumKong(benchmark::State& state) {
    auto& p = DataProvider::instance();
    for (auto _ : state) {
        nk_f64_t result = 0;
        nk_sqeuclidean_f32(p.get_q(), p.get_t(), p.get_dim(), &result);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_NumKong)->Name("NumKong_L2_sqeuclidean");

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_simd_distance");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    try { program.parse_args(argc, argv); } catch (...) { return 1; }
    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    DataProvider::instance().init();

#if NK_TARGET_SKYLAKE
    printf("[NumKong] Using Skylake (AVX-512)\n");
#elif NK_TARGET_HASWELL
    printf("[NumKong] Using Haswell (AVX2)\n");
#else
    printf("[NumKong] Using serial fallback\n");
#endif

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}