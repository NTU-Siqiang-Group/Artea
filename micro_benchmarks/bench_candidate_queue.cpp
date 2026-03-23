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
 * @FilePath: /Artea/micro_benchmarks/bench_candidate_queue.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Micro-benchmark for StdCandidateQueue, LinearCandidateQueue,
 *               and FHCandidateQueue — try_push & pop_best_unexplored.
 */

#include <benchmark/benchmark.h>
#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace artea;
using namespace artea::cpu;

// --- Global Configuration ---
struct BenchConfig {
    uint32_t seed;
    std::vector<int64_t> capacities;  // user-specified capacity list
} g_config;

// --- Pre-generated data provider ---
class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init(uint32_t seed, std::size_t max_entries) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<distance_t> dist(0.0f, 10000.0f);
        vertex_ids_.reserve(max_entries);
        distances_.reserve(max_entries);
        for (std::size_t i = 0; i < max_entries; ++i) {
            vertex_ids_.push_back(static_cast<vertex_id_t>(i));
            distances_.push_back(dist(rng));
        }
    }

    const std::vector<vertex_id_t>& vertex_ids() const { return vertex_ids_; }
    const std::vector<distance_t>& distances() const { return distances_; }

private:
    std::vector<vertex_id_t> vertex_ids_;
    std::vector<distance_t> distances_;
};

// ============================================================================
// Benchmark: try_push (fill queue from empty to capacity)
// ============================================================================

static void BM_TryPush_StdQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();

    for (auto _ : state) {
        std_candidate_queue_t q(K);
        for (std::size_t i = 0; i < K; ++i) {
            q.try_push(vertex_ids[i], distances[i]);
        }
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(K));
}

static void BM_TryPush_LinearQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();

    for (auto _ : state) {
        linear_candidate_queue_t q(K);
        for (std::size_t i = 0; i < K; ++i) {
            q.try_push(vertex_ids[i], distances[i]);
        }
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(K));
}

static void BM_TryPush_FHQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();

    for (auto _ : state) {
        fh_candidate_queue_t q(K);
        for (std::size_t i = 0; i < K; ++i) {
            q.try_push(vertex_ids[i], distances[i]);
        }
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(K));
}

// ============================================================================
// Benchmark: try_push with eviction (queue full, push 4x entries)
// ============================================================================

static void BM_TryPushEvict_StdQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const std::size_t N = K * 4;
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();

    for (auto _ : state) {
        std_candidate_queue_t q(K);
        for (std::size_t i = 0; i < N; ++i) {
            q.try_push(vertex_ids[i], distances[i]);
        }
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
}

static void BM_TryPushEvict_LinearQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const std::size_t N = K * 4;
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();

    for (auto _ : state) {
        linear_candidate_queue_t q(K);
        for (std::size_t i = 0; i < N; ++i) {
            q.try_push(vertex_ids[i], distances[i]);
        }
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
}

static void BM_TryPushEvict_FHQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const std::size_t N = K * 4;
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();

    for (auto _ : state) {
        fh_candidate_queue_t q(K);
        for (std::size_t i = 0; i < N; ++i) {
            q.try_push(vertex_ids[i], distances[i]);
        }
        benchmark::DoNotOptimize(q);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(N));
}

// ============================================================================
// Benchmark: pop_best_unexplored (drain a full queue)
// ============================================================================

static void BM_GetBest_StdQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();
    // Initialize with candidate entries
        std::vector<candidate_entry_t> init_data;
        init_data.reserve(K);
        for (std::size_t i = 0; i < K; ++i) {
            init_data.emplace_back(vertex_ids[i], distances[i]);
        }

    for (auto _ : state) {
        std_candidate_queue_t q(K);
        q.initialize(init_data);

        for (std::size_t i = 0; i < K; ++i) {
            auto e = q.pop_best_unexplored();
            benchmark::DoNotOptimize(e.first);
            benchmark::DoNotOptimize(e.second);
        }
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(K));
}

static void BM_GetBest_LinearQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();

    for (auto _ : state) {
        // Initialize with candidate entries
        std::vector<candidate_entry_t> init_data;
        init_data.reserve(K);
        for (std::size_t i = 0; i < K; ++i) {
            init_data.emplace_back(vertex_ids[i], distances[i]);
        }
        linear_candidate_queue_t q(K);
        q.initialize(init_data);

        for (std::size_t i = 0; i < K; ++i) {
            auto e = q.pop_best_unexplored();
            benchmark::DoNotOptimize(e.first);
            benchmark::DoNotOptimize(e.second);
        }
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(K));
}

static void BM_GetBest_FHQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();
    // Initialize with candidate entries
        std::vector<candidate_entry_t> init_data;
        init_data.reserve(K);
        for (std::size_t i = 0; i < K; ++i) {
            init_data.emplace_back(vertex_ids[i], distances[i]);
        }

    for (auto _ : state) {
        fh_candidate_queue_t q(K);
        q.initialize(init_data);

        for (std::size_t i = 0; i < K; ++i) {
            auto e = q.pop_best_unexplored();
            benchmark::DoNotOptimize(e.first);
            benchmark::DoNotOptimize(e.second);
        }
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(K));
}

// ============================================================================
// Benchmark: mixed push + explore (simulates real graph search pattern)
//   Fill queue to K, then alternate: explore one, push two new candidates.
// ============================================================================

static void BM_Mixed_StdQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();

    for (auto _ : state) {
        std_candidate_queue_t q(K);
        for (std::size_t i = 0; i < K; ++i) {
            q.try_push(vertex_ids[i], distances[i]);
        }
        std::size_t push_idx = K;
        while (!q.empty() && push_idx + 1 < vertex_ids.size()) {
            auto [best_id, best_dist] = q.pop_best_unexplored();
            benchmark::DoNotOptimize(best_id);
            benchmark::DoNotOptimize(best_dist);
            if (best_dist == base_traits_t::max_distance) break;
            q.try_push(vertex_ids[push_idx], distances[push_idx]);
            push_idx++;
            if (push_idx < vertex_ids.size()) {
                q.try_push(vertex_ids[push_idx], distances[push_idx]);
                push_idx++;
            }
        }
        benchmark::ClobberMemory();
    }
}

static void BM_Mixed_LinearQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();

    for (auto _ : state) {
        linear_candidate_queue_t q(K);
        for (std::size_t i = 0; i < K; ++i) {
            q.try_push(vertex_ids[i], distances[i]);
        }
        std::size_t push_idx = K;
        while (!q.empty() && push_idx + 1 < vertex_ids.size()) {
            auto [best_id, best_dist] = q.pop_best_unexplored();
            benchmark::DoNotOptimize(best_id);
            benchmark::DoNotOptimize(best_dist);
            q.try_push(vertex_ids[push_idx], distances[push_idx]);
            push_idx++;
            if (push_idx < vertex_ids.size()) {
                q.try_push(vertex_ids[push_idx], distances[push_idx]);
                push_idx++;
            }
        }
        benchmark::ClobberMemory();
    }
}

static void BM_Mixed_FHQueue(benchmark::State& state) {
    const std::size_t K = static_cast<std::size_t>(state.range(0));
    const auto& vertex_ids = DataProvider::instance().vertex_ids();
    const auto& distances = DataProvider::instance().distances();

    for (auto _ : state) {
        fh_candidate_queue_t q(K);
        for (std::size_t i = 0; i < K; ++i) {
            q.try_push(vertex_ids[i], distances[i]);
        }
        std::size_t push_idx = K;
        while (!q.empty() && push_idx + 1 < vertex_ids.size()) {
            auto [best_id, best_dist] = q.pop_best_unexplored();
            benchmark::DoNotOptimize(best_id);
            benchmark::DoNotOptimize(best_dist);
            if (push_idx < vertex_ids.size()) {
                q.try_push(vertex_ids[push_idx], distances[push_idx]);
                push_idx++;
            }
        }
        benchmark::ClobberMemory();
    }
}

// ============================================================================
// Register benchmarks — done in main() after g_config is populated
// ============================================================================

static void RegisterAllBenchmarks() {
    auto apply = [](::benchmark::Benchmark* b) {
        for (auto k : g_config.capacities) {
            b->Arg(k);
        }
        b->Unit(benchmark::kMicrosecond);
    };

    // try_push (fill)
    apply(benchmark::RegisterBenchmark("BM_TryPush_StdQueue", BM_TryPush_StdQueue));
    apply(benchmark::RegisterBenchmark("BM_TryPush_LinearQueue", BM_TryPush_LinearQueue));
    apply(benchmark::RegisterBenchmark("BM_TryPush_FHQueue", BM_TryPush_FHQueue));

    // try_push with eviction
    apply(benchmark::RegisterBenchmark("BM_TryPushEvict_StdQueue", BM_TryPushEvict_StdQueue));
    apply(benchmark::RegisterBenchmark("BM_TryPushEvict_LinearQueue", BM_TryPushEvict_LinearQueue));
    apply(benchmark::RegisterBenchmark("BM_TryPushEvict_FHQueue", BM_TryPushEvict_FHQueue));

    // pop_best_unexplored (drain)
    apply(benchmark::RegisterBenchmark("BM_GetBest_StdQueue", BM_GetBest_StdQueue));
    apply(benchmark::RegisterBenchmark("BM_GetBest_LinearQueue", BM_GetBest_LinearQueue));
    apply(benchmark::RegisterBenchmark("BM_GetBest_FHQueue", BM_GetBest_FHQueue));

    // mixed push + explore
    apply(benchmark::RegisterBenchmark("BM_Mixed_StdQueue", BM_Mixed_StdQueue));
    apply(benchmark::RegisterBenchmark("BM_Mixed_LinearQueue", BM_Mixed_LinearQueue));
    apply(benchmark::RegisterBenchmark("BM_Mixed_FHQueue", BM_Mixed_FHQueue));
}

// ============================================================================
// Main
// ============================================================================

static std::vector<int64_t> parse_capacity_list(const std::string& s) {
    std::vector<int64_t> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(std::stoll(token));
    }
    return result;
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    argparse::ArgumentParser program("bench_candidate_queue");
    program.add_description("Micro-benchmark for CandidateQueue implementations");

    program.add_argument("-s", "--seed")
        .default_value(uint32_t{42})
        .scan<'u', uint32_t>()
        .help("Random seed for data generation");

    program.add_argument("-k", "--capacities")
        .default_value(std::string("16,32,64,128,256,512,1024,2048,4096"))
        .help("Comma-separated list of queue capacities to benchmark");

    program.add_argument("-h", "--help")
        .default_value(false)
        .implicit_value(true)
        .help("Show this help message");

    try {
        program.parse_known_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    if (program.get<bool>("--help")) {
        std::cout << program;
        return 0;
    }

    g_config.seed = program.get<uint32_t>("--seed");
    g_config.capacities = parse_capacity_list(program.get<std::string>("--capacities"));

    // Compute max entries needed: largest capacity * 4 (for eviction benchmarks)
    int64_t max_cap = *std::max_element(g_config.capacities.begin(), g_config.capacities.end());
    std::size_t max_entries = static_cast<std::size_t>(max_cap) * 4;

    logger.info("==========================================================");
    logger.info("      CandidateQueue Micro-Benchmark");
    logger.info(fmt::format("      Seed: {}", g_config.seed));
    logger.info(fmt::format("      Capacities: {}", program.get<std::string>("--capacities")));
    logger.info(fmt::format("      Max pre-generated entries: {}", max_entries));
    logger.info("==========================================================");

    DataProvider::instance().init(g_config.seed, max_entries);

    RegisterAllBenchmarks();

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
