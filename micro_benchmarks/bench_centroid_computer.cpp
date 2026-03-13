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
 * @FilePath: /Artea/micro_benchmarks/bench_centroid_computer.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Benchmark comparing SIMD vs Serial centroid computation.
 */

#include <benchmark/benchmark.h>
#include <vector>
#include <random>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;
using namespace artea::cpu::default_context;

// Serial implementation for comparison
class SerialCentroidComputer {
public:
    static auto compute(const vector_array_t& vecs) -> std::vector<vec_ele_t> {
        const vertex_num_t num_vecs = vecs.get_num_vecs();
        const vec_dim_t vec_dim = vecs.get_vec_dim();

        if (num_vecs == 0) {
            return std::vector<vec_ele_t>(vec_dim, 0.0f);
        }

        std::vector<vec_ele_t> centroid(vec_dim, 0.0f);

        // Sum all vectors
        for (vertex_num_t i = 0; i < num_vecs; ++i) {
            const vec_ele_t* vec = vecs.get(i);
            for (vec_dim_t d = 0; d < vec_dim; ++d) {
                centroid[d] += vec[d];
            }
        }

        // Divide by number of vectors to get mean
        const vec_ele_t inv_num_vecs = static_cast<vec_ele_t>(1.0) / static_cast<vec_ele_t>(num_vecs);
        for (vec_dim_t d = 0; d < vec_dim; ++d) {
            centroid[d] *= inv_num_vecs;
        }

        return centroid;
    }
};

// Benchmark SIMD implementation
static void BM_CentroidComputer_SIMD(benchmark::State& state) {
    const vertex_num_t num_vecs = state.range(0);
    const vec_dim_t vec_dim = state.range(1);

    // Generate random vectors
    std::mt19937 rng(42);
    std::uniform_real_distribution<vec_ele_t> dist(-1.0f, 1.0f);

    vector_array_t vecs(num_vecs, vec_dim);
    for (vertex_num_t i = 0; i < num_vecs; ++i) {
        vec_ele_t* vec = vecs.get(i);
        for (vec_dim_t d = 0; d < vec_dim; ++d) {
            vec[d] = dist(rng);
        }
    }

    for (auto _ : state) {
        auto centroid = centroid_computer_t::compute(vecs);
        benchmark::DoNotOptimize(centroid);
    }

    state.SetItemsProcessed(state.iterations() * num_vecs * vec_dim);
}

// Benchmark Serial implementation
static void BM_CentroidComputer_Serial(benchmark::State& state) {
    const vertex_num_t num_vecs = state.range(0);
    const vec_dim_t vec_dim = state.range(1);

    // Generate random vectors
    std::mt19937 rng(42);
    std::uniform_real_distribution<vec_ele_t> dist(-1.0f, 1.0f);

    vector_array_t vecs(num_vecs, vec_dim);
    for (vertex_num_t i = 0; i < num_vecs; ++i) {
        vec_ele_t* vec = vecs.get(i);
        for (vec_dim_t d = 0; d < vec_dim; ++d) {
            vec[d] = dist(rng);
        }
    }

    for (auto _ : state) {
        auto centroid = SerialCentroidComputer::compute(vecs);
        benchmark::DoNotOptimize(centroid);
    }

    state.SetItemsProcessed(state.iterations() * num_vecs * vec_dim);
}

// Register benchmarks with different configurations
// Format: (num_vecs, vec_dim)

// Small dataset, various dimensions
BENCHMARK(BM_CentroidComputer_SIMD)->Args({100, 128});
BENCHMARK(BM_CentroidComputer_Serial)->Args({100, 128});

BENCHMARK(BM_CentroidComputer_SIMD)->Args({100, 256});
BENCHMARK(BM_CentroidComputer_Serial)->Args({100, 256});

BENCHMARK(BM_CentroidComputer_SIMD)->Args({100, 512});
BENCHMARK(BM_CentroidComputer_Serial)->Args({100, 512});

// Medium dataset
BENCHMARK(BM_CentroidComputer_SIMD)->Args({1000, 128});
BENCHMARK(BM_CentroidComputer_Serial)->Args({1000, 128});

BENCHMARK(BM_CentroidComputer_SIMD)->Args({1000, 256});
BENCHMARK(BM_CentroidComputer_Serial)->Args({1000, 256});

BENCHMARK(BM_CentroidComputer_SIMD)->Args({1000, 512});
BENCHMARK(BM_CentroidComputer_Serial)->Args({1000, 512});

// Large dataset
BENCHMARK(BM_CentroidComputer_SIMD)->Args({10000, 128});
BENCHMARK(BM_CentroidComputer_Serial)->Args({10000, 128});

BENCHMARK(BM_CentroidComputer_SIMD)->Args({10000, 256});
BENCHMARK(BM_CentroidComputer_Serial)->Args({10000, 256});

BENCHMARK(BM_CentroidComputer_SIMD)->Args({10000, 512});
BENCHMARK(BM_CentroidComputer_Serial)->Args({10000, 512});

// Very large dataset
BENCHMARK(BM_CentroidComputer_SIMD)->Args({100000, 128});
BENCHMARK(BM_CentroidComputer_Serial)->Args({100000, 128});

BENCHMARK(BM_CentroidComputer_SIMD)->Args({100000, 256});
BENCHMARK(BM_CentroidComputer_Serial)->Args({100000, 256});

BENCHMARK_MAIN();
