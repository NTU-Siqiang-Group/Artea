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
#include <artea/cpu/utils/simd_fma.hpp>
#include <faiss/utils/distances.h>
#include <hnswlib/hnswlib.h>

using namespace artea;
using namespace artea::cpu;

using base_traits_t = BaseTraits<uint32_t, float, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::DOT>;
template <std::size_t U> using artea_simd_fma_t = SIMDFMA<computer_traits_t, U>;

struct TestConfig { std::string config_path, dataset_name; } g_config;

class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }
    void init() {
        auto dataset = std::make_unique<typename base_traits_t::vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dim_ = dataset->get_base_vecs().get_vec_dim();
        q_.resize(dim_); t_.resize(dim_);
        std::memcpy(q_.data(), dataset->get_query_vecs().get(0), dim_ * sizeof(float));
        std::memcpy(t_.data(), dataset->get_base_vecs().get(0), dim_ * sizeof(float));
    }
    uint32_t get_dim() const { return dim_; }
    float* get_q() { return q_.data(); }
    float* get_t() { return t_.data(); }
private:
    uint32_t dim_;
    std::vector<float> q_, t_;
};

// 1. Artea FMA (1, 2, 4)
template <std::size_t U>
static void BM_Artea_FMA(benchmark::State& state) {
    auto& p = DataProvider::instance();
    artea_simd_fma_t<U> func(p.get_dim());
    for (auto _ : state) benchmark::DoNotOptimize(func(p.get_q(), p.get_t()));
}
BENCHMARK_TEMPLATE(BM_Artea_FMA, 1)->Name("Artea_FMA_U1");
BENCHMARK_TEMPLATE(BM_Artea_FMA, 2)->Name("Artea_FMA_U2");
BENCHMARK_TEMPLATE(BM_Artea_FMA, 4)->Name("Artea_FMA_U4");

// 2. Faiss IP
static void BM_Faiss_IP(benchmark::State& state) {
    auto& p = DataProvider::instance();
    for (auto _ : state) benchmark::DoNotOptimize(faiss::fvec_inner_product(p.get_q(), p.get_t(), p.get_dim()));
}
BENCHMARK(BM_Faiss_IP)->Name("Faiss_IP");

// 3. HNSWLib IP (Returns 1-dot)
static void BM_HNSWLib_IP(benchmark::State& state) {
    auto& p = DataProvider::instance();
    hnswlib::InnerProductSpace space(p.get_dim());
    auto f = space.get_dist_func();
    void* param = space.get_dist_func_param();
    for (auto _ : state) benchmark::DoNotOptimize(f(p.get_q(), p.get_t(), param));
}
BENCHMARK(BM_HNSWLib_IP)->Name("HNSWLib_IP");

int main(int argc, char** argv) {
    argparse::ArgumentParser program("bench_simd_fma");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    try { program.parse_args(argc, argv); } catch (...) { return 1; }
    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    DataProvider::instance().init();
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}