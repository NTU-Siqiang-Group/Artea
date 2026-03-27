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
 * @FilePath: /Artea/micro_benchmarks/bench_visited_table.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Micro-benchmark for ThreadLocalBitmap vs VersionTagTable
 *               — set, test, clear via VisitedTablePool (multi-threaded).
 */

#include <benchmark/benchmark.h>
#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/containers/thread_local_bitmap.hpp>
#include <artea/cpu/containers/version_tag_table.hpp>

#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <atomic>

using namespace artea;
using namespace artea::cpu;

// --- Type Definitions ---
using vec_num_t = uint32_t;
using vec_ele_t = float;
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using index_traits_t = IndexTraits<base_traits_t>;
using router_traits_t = RouterTraits<computer_traits_t, index_traits_t, false>;

using bitmap_pool_t = VisitedTablePool<router_traits_t, ThreadLocalBitmap>;
using version_tag_pool_t = VisitedTablePool<router_traits_t, VersionTagTable>;

// --- Global Configuration ---
struct BenchConfig {
    uint32_t seed;
    uint32_t num_queries;
    uint32_t visit_count;
    std::vector<int64_t> table_sizes;
} g_config;

// --- Pre-generated random visit indices per query ---
class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init(uint32_t seed, uint32_t num_queries, uint32_t visit_count, vec_num_t max_table_size) {
        num_queries_ = num_queries;
        visit_count_ = visit_count;

        indices_.resize(num_queries);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<vec_num_t> dist(0, max_table_size - 1);

        for (uint32_t q = 0; q < num_queries; ++q) {
            indices_[q].resize(visit_count);
            for (uint32_t i = 0; i < visit_count; ++i) {
                indices_[q][i] = dist(rng);
            }
        }
    }

    const std::vector<vec_num_t>& query_indices(uint32_t q) const {
        return indices_[q % num_queries_];
    }

    uint32_t num_queries() const { return num_queries_; }
    uint32_t visit_count() const { return visit_count_; }

private:
    std::vector<std::vector<vec_num_t>> indices_;
    uint32_t num_queries_ = 0;
    uint32_t visit_count_ = 0;
};

// ============================================================================
// 1. Clear — measures the cost of clearing the visited table between queries.
//    Bitmap: O(n) memset.  VersionTag: O(1) version bump.
// ============================================================================

static void BM_Clear_Bitmap(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    const uint32_t num_queries = g_config.num_queries;
    bitmap_pool_t pool(N);
    pool.warmup();

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, num_queries),
            [&](const tbb::blocked_range<uint32_t>& r) {
                auto& visited = pool.acquire();
                for (uint32_t q = r.begin(); q != r.end(); ++q) {
                    visited.clear();
                    benchmark::ClobberMemory();
                }
            }
        );
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(num_queries));
}

static void BM_Clear_VersionTag(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    const uint32_t num_queries = g_config.num_queries;
    version_tag_pool_t pool(N);
    pool.warmup();

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, num_queries),
            [&](const tbb::blocked_range<uint32_t>& r) {
                auto& visited = pool.acquire();
                for (uint32_t q = r.begin(); q != r.end(); ++q) {
                    visited.clear();
                    benchmark::ClobberMemory();
                }
            }
        );
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(num_queries));
}

// ============================================================================
// 2. Set — measures the cost of marking vertices as visited.
//    Each query sets visit_count random indices.
// ============================================================================

static void BM_Set_Bitmap(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    const auto& data = DataProvider::instance();
    const uint32_t num_queries = data.num_queries();
    const uint32_t visit_count = data.visit_count();
    bitmap_pool_t pool(N);
    pool.warmup();

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, num_queries),
            [&](const tbb::blocked_range<uint32_t>& r) {
                auto& visited = pool.acquire();
                for (uint32_t q = r.begin(); q != r.end(); ++q) {
                    visited.clear();
                    const auto& indices = data.query_indices(q);
                    for (uint32_t i = 0; i < visit_count; ++i) {
                        visited.set(indices[i]);
                    }
                    benchmark::ClobberMemory();
                }
            }
        );
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(num_queries) * static_cast<int64_t>(visit_count));
}

static void BM_Set_VersionTag(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    const auto& data = DataProvider::instance();
    const uint32_t num_queries = data.num_queries();
    const uint32_t visit_count = data.visit_count();
    version_tag_pool_t pool(N);
    pool.warmup();

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, num_queries),
            [&](const tbb::blocked_range<uint32_t>& r) {
                auto& visited = pool.acquire();
                for (uint32_t q = r.begin(); q != r.end(); ++q) {
                    visited.clear();
                    const auto& indices = data.query_indices(q);
                    for (uint32_t i = 0; i < visit_count; ++i) {
                        visited.set(indices[i]);
                    }
                    benchmark::ClobberMemory();
                }
            }
        );
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(num_queries) * static_cast<int64_t>(visit_count));
}

// ============================================================================
// 3. Test — measures the cost of checking visited status.
//    Each query sets visit_count indices, then tests 2x visit_count indices
//    (mix of hits and misses).
// ============================================================================

static void BM_Test_Bitmap(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    const auto& data = DataProvider::instance();
    const uint32_t num_queries = data.num_queries();
    const uint32_t visit_count = data.visit_count();
    bitmap_pool_t pool(N);
    pool.warmup();

    // Pre-generate test indices (shifted from set indices to get mix of hits/misses)
    std::vector<std::vector<vec_num_t>> test_indices(num_queries);
    std::mt19937 rng(g_config.seed + 1);
    std::uniform_int_distribution<vec_num_t> dist(0, N - 1);
    for (uint32_t q = 0; q < num_queries; ++q) {
        const auto& set_idx = data.query_indices(q);
        test_indices[q].resize(visit_count * 2);
        // First half: re-test set indices (guaranteed hits)
        for (uint32_t i = 0; i < visit_count; ++i) {
            test_indices[q][i] = set_idx[i];
        }
        // Second half: random indices (likely misses)
        for (uint32_t i = visit_count; i < visit_count * 2; ++i) {
            test_indices[q][i] = dist(rng);
        }
    }

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, num_queries),
            [&](const tbb::blocked_range<uint32_t>& r) {
                auto& visited = pool.acquire();
                for (uint32_t q = r.begin(); q != r.end(); ++q) {
                    visited.clear();
                    const auto& set_idx = data.query_indices(q);
                    for (uint32_t i = 0; i < visit_count; ++i) {
                        visited.set(set_idx[i]);
                    }
                    const auto& tst_idx = test_indices[q % num_queries];
                    bool sink = false;
                    for (uint32_t i = 0; i < visit_count * 2; ++i) {
                        sink ^= visited.test(tst_idx[i]);
                    }
                    benchmark::DoNotOptimize(sink);
                }
            }
        );
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(num_queries) * static_cast<int64_t>(visit_count * 2));
}

static void BM_Test_VersionTag(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    const auto& data = DataProvider::instance();
    const uint32_t num_queries = data.num_queries();
    const uint32_t visit_count = data.visit_count();
    version_tag_pool_t pool(N);
    pool.warmup();

    std::vector<std::vector<vec_num_t>> test_indices(num_queries);
    std::mt19937 rng(g_config.seed + 1);
    std::uniform_int_distribution<vec_num_t> dist(0, N - 1);
    for (uint32_t q = 0; q < num_queries; ++q) {
        const auto& set_idx = data.query_indices(q);
        test_indices[q].resize(visit_count * 2);
        for (uint32_t i = 0; i < visit_count; ++i) {
            test_indices[q][i] = set_idx[i];
        }
        for (uint32_t i = visit_count; i < visit_count * 2; ++i) {
            test_indices[q][i] = dist(rng);
        }
    }

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, num_queries),
            [&](const tbb::blocked_range<uint32_t>& r) {
                auto& visited = pool.acquire();
                for (uint32_t q = r.begin(); q != r.end(); ++q) {
                    visited.clear();
                    const auto& set_idx = data.query_indices(q);
                    for (uint32_t i = 0; i < visit_count; ++i) {
                        visited.set(set_idx[i]);
                    }
                    const auto& tst_idx = test_indices[q % num_queries];
                    bool sink = false;
                    for (uint32_t i = 0; i < visit_count * 2; ++i) {
                        sink ^= visited.test(tst_idx[i]);
                    }
                    benchmark::DoNotOptimize(sink);
                }
            }
        );
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(num_queries) * static_cast<int64_t>(visit_count * 2));
}

// ============================================================================
// 4. Mixed — simulates a realistic beam search pattern:
//    clear, then interleave set + test for visit_count vertices.
// ============================================================================

static void BM_Mixed_Bitmap(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    const auto& data = DataProvider::instance();
    const uint32_t num_queries = data.num_queries();
    const uint32_t visit_count = data.visit_count();
    bitmap_pool_t pool(N);
    pool.warmup();

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, num_queries),
            [&](const tbb::blocked_range<uint32_t>& r) {
                auto& visited = pool.acquire();
                for (uint32_t q = r.begin(); q != r.end(); ++q) {
                    visited.clear();
                    const auto& indices = data.query_indices(q);
                    uint32_t new_visits = 0;
                    for (uint32_t i = 0; i < visit_count; ++i) {
                        if (!visited.test(indices[i])) {
                            visited.set(indices[i]);
                            ++new_visits;
                        }
                    }
                    benchmark::DoNotOptimize(new_visits);
                }
            }
        );
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(num_queries) * static_cast<int64_t>(visit_count));
}

static void BM_Mixed_VersionTag(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    const auto& data = DataProvider::instance();
    const uint32_t num_queries = data.num_queries();
    const uint32_t visit_count = data.visit_count();
    version_tag_pool_t pool(N);
    pool.warmup();

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, num_queries),
            [&](const tbb::blocked_range<uint32_t>& r) {
                auto& visited = pool.acquire();
                for (uint32_t q = r.begin(); q != r.end(); ++q) {
                    visited.clear();
                    const auto& indices = data.query_indices(q);
                    uint32_t new_visits = 0;
                    for (uint32_t i = 0; i < visit_count; ++i) {
                        if (!visited.test(indices[i])) {
                            visited.set(indices[i]);
                            ++new_visits;
                        }
                    }
                    benchmark::DoNotOptimize(new_visits);
                }
            }
        );
    }

    state.SetItemsProcessed(
        state.iterations() * static_cast<int64_t>(num_queries) * static_cast<int64_t>(visit_count));
}

// ============================================================================
// 5. Single-Thread Latency — measures the raw per-operation cost of a single
//    set / test / clear call without any TBB overhead.
//    Each iteration performs one operation; Google Benchmark reports the
//    average wall-clock time directly (typically in nanoseconds).
// ============================================================================

static constexpr size_t LATENCY_PREGEN = 1 << 20;  // 1M pre-generated indices

// Set benchmarks: compare ThreadLocalBitmap
static void BM_Latency_Set_Bitmap(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    ThreadLocalBitmap table(N);

    std::mt19937 rng(g_config.seed);
    std::uniform_int_distribution<vec_num_t> dist(0, N - 1);
    std::vector<vec_num_t> indices(LATENCY_PREGEN);
    for (auto& idx : indices) idx = dist(rng);

    size_t cursor = 0;
    for (auto _ : state) {
        table.set(indices[cursor]);
        benchmark::DoNotOptimize(indices[cursor]);
        cursor = (cursor + 1) & (LATENCY_PREGEN - 1);
    }
}

static void BM_Latency_Set_VersionTag(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    VersionTagTable table(N);

    std::mt19937 rng(g_config.seed);
    std::uniform_int_distribution<vec_num_t> dist(0, N - 1);
    std::vector<vec_num_t> indices(LATENCY_PREGEN);
    for (auto& idx : indices) idx = dist(rng);

    size_t cursor = 0;
    for (auto _ : state) {
        table.set(indices[cursor]);
        benchmark::DoNotOptimize(indices[cursor]);
        cursor = (cursor + 1) & (LATENCY_PREGEN - 1);
    }
}

// Test benchmarks: compare ThreadLocalBitmap
static void BM_Latency_Test_Bitmap(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    ThreadLocalBitmap table(N);

    // Pre-set half the table so we get a mix of hits and misses
    std::mt19937 setup_rng(g_config.seed);
    std::uniform_int_distribution<vec_num_t> setup_dist(0, N - 1);
    for (vec_num_t i = 0; i < N / 2; ++i) {
        table.set(setup_dist(setup_rng));
    }

    std::mt19937 rng(g_config.seed + 1);
    std::uniform_int_distribution<vec_num_t> dist(0, N - 1);
    std::vector<vec_num_t> indices(LATENCY_PREGEN);
    for (auto& idx : indices) idx = dist(rng);

    size_t cursor = 0;
    for (auto _ : state) {
        bool result = table.test(indices[cursor]);
        benchmark::DoNotOptimize(result);
        cursor = (cursor + 1) & (LATENCY_PREGEN - 1);
    }
}

static void BM_Latency_Test_VersionTag(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    VersionTagTable table(N);

    std::mt19937 setup_rng(g_config.seed);
    std::uniform_int_distribution<vec_num_t> setup_dist(0, N - 1);
    for (vec_num_t i = 0; i < N / 2; ++i) {
        table.set(setup_dist(setup_rng));
    }

    std::mt19937 rng(g_config.seed + 1);
    std::uniform_int_distribution<vec_num_t> dist(0, N - 1);
    std::vector<vec_num_t> indices(LATENCY_PREGEN);
    for (auto& idx : indices) idx = dist(rng);

    size_t cursor = 0;
    for (auto _ : state) {
        bool result = table.test(indices[cursor]);
        benchmark::DoNotOptimize(result);
        cursor = (cursor + 1) & (LATENCY_PREGEN - 1);
    }
}

// Clear benchmarks: compare ThreadLocalBitmap and VersionTag
// For ThreadLocalBitmap, pre-set some bits to make clear meaningful
static void BM_Latency_Clear_Bitmap(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    ThreadLocalBitmap table(N);

    // Pre-set 256 bits (typical beam search visit count)
    std::mt19937 setup_rng(g_config.seed);
    std::uniform_int_distribution<vec_num_t> setup_dist(0, N - 1);
    for (vec_num_t i = 0; i < 256; ++i) {
        table.set(setup_dist(setup_rng));
    }

    for (auto _ : state) {
        table.clear();
        benchmark::ClobberMemory();
        // Re-set bits for next iteration
        state.PauseTiming();
        for (vec_num_t i = 0; i < 256; ++i) {
            table.set(setup_dist(setup_rng));
        }
        state.ResumeTiming();
    }
}

static void BM_Latency_Clear_VersionTag(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    VersionTagTable table(N);

    for (auto _ : state) {
        table.clear();
        benchmark::ClobberMemory();
    }
}

// ============================================================================
// 6. Memory Footprint — reports per-table and total pool memory usage.
//    Bitmap: ceil(N/64) * 8 bytes.  VersionTag: N * 2 bytes.
//    Reported via state.counters so it appears in benchmark output columns.
// ============================================================================

static void BM_Memory_Bitmap(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    const int num_threads = tbb_max_num_threads();

    // Per-table: ceil(N / 64) words * 8 bytes/word
    const size_t words_per_table = (N + 63) / 64;
    const size_t bytes_per_table = words_per_table * sizeof(uint64_t);
    const size_t total_bytes = bytes_per_table * static_cast<size_t>(num_threads);

    bitmap_pool_t pool(N);
    pool.warmup();

    for (auto _ : state) {
        // Minimal work so the benchmark row exists for the counters
        tbb::parallel_for(
            tbb::blocked_range<int>(0, num_threads, 1),
            [&](const tbb::blocked_range<int>&) {
                auto& visited = pool.acquire();
                visited.clear();
                benchmark::ClobberMemory();
            }
        );
    }

    state.counters["bytes/table"] = benchmark::Counter(
        static_cast<double>(bytes_per_table), benchmark::Counter::kDefaults);
    state.counters["total_bytes"] = benchmark::Counter(
        static_cast<double>(total_bytes), benchmark::Counter::kDefaults);
    state.counters["total_MB"] = benchmark::Counter(
        static_cast<double>(total_bytes) / (1024.0 * 1024.0), benchmark::Counter::kDefaults);
    state.counters["threads"] = benchmark::Counter(
        static_cast<double>(num_threads), benchmark::Counter::kDefaults);
}

static void BM_Memory_VersionTag(benchmark::State& state) {
    const vec_num_t N = static_cast<vec_num_t>(state.range(0));
    const int num_threads = tbb_max_num_threads();

    // Per-table: N * sizeof(uint16_t)
    const size_t bytes_per_table = static_cast<size_t>(N) * sizeof(uint16_t);
    const size_t total_bytes = bytes_per_table * static_cast<size_t>(num_threads);

    version_tag_pool_t pool(N);
    pool.warmup();

    for (auto _ : state) {
        tbb::parallel_for(
            tbb::blocked_range<int>(0, num_threads, 1),
            [&](const tbb::blocked_range<int>&) {
                auto& visited = pool.acquire();
                visited.clear();
                benchmark::ClobberMemory();
            }
        );
    }

    state.counters["bytes/table"] = benchmark::Counter(
        static_cast<double>(bytes_per_table), benchmark::Counter::kDefaults);
    state.counters["total_bytes"] = benchmark::Counter(
        static_cast<double>(total_bytes), benchmark::Counter::kDefaults);
    state.counters["total_MB"] = benchmark::Counter(
        static_cast<double>(total_bytes) / (1024.0 * 1024.0), benchmark::Counter::kDefaults);
    state.counters["threads"] = benchmark::Counter(
        static_cast<double>(num_threads), benchmark::Counter::kDefaults);
}

// ============================================================================
// Register benchmarks
// ============================================================================

static void RegisterAllBenchmarks() {
    auto apply = [](::benchmark::Benchmark* b) {
        for (auto sz : g_config.table_sizes) {
            b->Arg(sz);
        }
        b->Unit(benchmark::kMicrosecond);
    };

    // Clear
    apply(benchmark::RegisterBenchmark("BM_Clear_Bitmap", BM_Clear_Bitmap));
    apply(benchmark::RegisterBenchmark("BM_Clear_VersionTag", BM_Clear_VersionTag));

    // Set
    apply(benchmark::RegisterBenchmark("BM_Set_Bitmap", BM_Set_Bitmap));
    apply(benchmark::RegisterBenchmark("BM_Set_VersionTag", BM_Set_VersionTag));

    // Test
    apply(benchmark::RegisterBenchmark("BM_Test_Bitmap", BM_Test_Bitmap));
    apply(benchmark::RegisterBenchmark("BM_Test_VersionTag", BM_Test_VersionTag));

    // Mixed (test-then-set, simulating beam search)
    apply(benchmark::RegisterBenchmark("BM_Mixed_Bitmap", BM_Mixed_Bitmap));
    apply(benchmark::RegisterBenchmark("BM_Mixed_VersionTag", BM_Mixed_VersionTag));

    // Single-thread latency (one op per iteration, reported in ns by default)
    auto apply_latency = [](::benchmark::Benchmark* b) {
        for (auto sz : g_config.table_sizes) {
            b->Arg(sz);
        }
        b->Unit(benchmark::kNanosecond);
    };

    // Set latency: compare ThreadLocalBitmap and VersionTag
    apply_latency(benchmark::RegisterBenchmark("BM_Latency_Set_Bitmap", BM_Latency_Set_Bitmap));
    apply_latency(benchmark::RegisterBenchmark("BM_Latency_Set_VersionTag", BM_Latency_Set_VersionTag));

    // Test latency: compare ThreadLocalBitmap and VersionTag
    apply_latency(benchmark::RegisterBenchmark("BM_Latency_Test_Bitmap", BM_Latency_Test_Bitmap));
    apply_latency(benchmark::RegisterBenchmark("BM_Latency_Test_VersionTag", BM_Latency_Test_VersionTag));

    // Clear latency: compare ThreadLocalBitmap and VersionTag
    apply_latency(benchmark::RegisterBenchmark("BM_Latency_Clear_Bitmap", BM_Latency_Clear_Bitmap));
    apply_latency(benchmark::RegisterBenchmark("BM_Latency_Clear_VersionTag", BM_Latency_Clear_VersionTag));

    // Memory footprint
    apply(benchmark::RegisterBenchmark("BM_Memory_Bitmap", BM_Memory_Bitmap));
    apply(benchmark::RegisterBenchmark("BM_Memory_VersionTag", BM_Memory_VersionTag));
}

// ============================================================================
// Main
// ============================================================================

static std::vector<int64_t> parse_size_list(const std::string& s) {
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

    argparse::ArgumentParser program("bench_visited_table");
    program.add_description("Micro-benchmark for ThreadLocalBitmap vs VersionTagTable via VisitedTablePool");

    program.add_argument("-s", "--seed")
        .default_value(uint32_t{42})
        .scan<'u', uint32_t>()
        .help("Random seed for data generation");

    program.add_argument("-n", "--num_queries")
        .default_value(uint32_t{1024})
        .scan<'u', uint32_t>()
        .help("Number of queries per benchmark iteration");

    program.add_argument("-v", "--visit_count")
        .default_value(uint32_t{256})
        .scan<'u', uint32_t>()
        .help("Number of vertices visited per query");

    program.add_argument("-t", "--table_sizes")
        .default_value(std::string("1000000"))
        .help("Comma-separated list of table sizes (num vertices) to benchmark");

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
    g_config.num_queries = program.get<uint32_t>("--num_queries");
    g_config.visit_count = program.get<uint32_t>("--visit_count");
    g_config.table_sizes = parse_size_list(program.get<std::string>("--table_sizes"));

    int64_t max_size = *std::max_element(g_config.table_sizes.begin(), g_config.table_sizes.end());

    ARTEA_INFO("==========================================================");
    ARTEA_INFO("      VisitedTable Micro-Benchmark");
    ARTEA_INFO(fmt::format("      Seed: {}", g_config.seed));
    ARTEA_INFO(fmt::format("      Queries: {}, Visits/query: {}",
                           g_config.num_queries, g_config.visit_count));
    ARTEA_INFO(fmt::format("      Table sizes: {}",
                           program.get<std::string>("--table_sizes")));
    ARTEA_INFO(fmt::format("      TBB threads: {}", tbb_max_num_threads()));
    ARTEA_INFO("==========================================================");

    DataProvider::instance().init(
        g_config.seed, g_config.num_queries, g_config.visit_count,
        static_cast<vec_num_t>(max_size));

    RegisterAllBenchmarks();

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    return 0;
}
