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

// Copyright 2026 Weitang Ye
// Correctness test for SIMD Linear Transform (Ax + b).

// Copyright 2026 Weitang Ye
// Performance benchmarks for SIMD Linear Transform (Ax + b).

#include <benchmark/benchmark.h>
#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/utils/simd_linear.hpp>
#include <faiss/utils/distances.h>

using namespace artea;
using namespace artea::cpu;

using base_traits_t = BaseTraits<uint32_t, float, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::DOT>;
template <std::size_t U> using artea_simd_linear_t = SIMDLinear<computer_traits_t, U>;

struct TestConfig { std::string config_path, dataset_name; } g_config;

class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }
    void init() {
        auto dataset = std::make_unique<typename base_traits_t::vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dim_ = dataset->get_base_vecs().get_vec_dim();
        a_.resize(dim_); x_.resize(dim_); bias_ = 4.25f;
        std::memcpy(a_.data(), dataset->get_query_vecs().get(0), dim_ * sizeof(float));
        std::memcpy(x_.data(), dataset->get_base_vecs().get(0), dim_ * sizeof(float));
    }
    uint32_t get_dim() const { return dim_; }
    float* get_a() { return a_.data(); }
    float* get_x() { return x_.data(); }
    float get_b() const { return bias_; }
private:
    uint32_t dim_; float bias_;
    std::vector<float> a_, x_;
};

// 1. Artea Fused Linear (1, 2, 4)
template <std::size_t U>
static void BM_Artea_Linear(benchmark::State& state) {
    auto& p = DataProvider::instance();
    artea_simd_linear_t<U> func(p.get_dim());
    for (auto _ : state) benchmark::DoNotOptimize(func(p.get_a(), p.get_x(), p.get_b()));
}
BENCHMARK_TEMPLATE(BM_Artea_Linear, 1)->Name("Artea_Linear_U1");
BENCHMARK_TEMPLATE(BM_Artea_Linear, 2)->Name("Artea_Linear_U2");
BENCHMARK_TEMPLATE(BM_Artea_Linear, 4)->Name("Artea_Linear_U4");

// 2. Reference: Faiss IP + Manual Bias Add (Non-fused)
// This benchmarks the cost of "Call library IP" + "Add float in C++"
static void BM_Ref_FaissIP_Add(benchmark::State& state) {
    auto& p = DataProvider::instance();
    float bias = p.get_b();
    for (auto _ : state) {
        float dot = faiss::fvec_inner_product(p.get_a(), p.get_x(), p.get_dim());
        benchmark::DoNotOptimize(dot + bias);
    }
}
BENCHMARK(BM_Ref_FaissIP_Add)->Name("Ref_FaissIP_Plus_Bias");

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_simd_linear");
    program.add_argument("-c", "--config").default_value(std::string("./datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    try { program.parse_args(argc, argv); } catch(...) { return 1; }
    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    DataProvider::instance().init();
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}