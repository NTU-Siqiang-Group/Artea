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
 * @FilePath: /Artea/unit_tests/test_tbb_concurrent_vector.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Benchmark comparing tbb::concurrent_vector parallel push,
 *               per-thread std::vector parallel push, and single-thread
 *               std::vector push throughput.
 */

#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <mutex>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

static constexpr size_t kTotalElements = 50'000'000;
static constexpr size_t kNumThreads = 64;
static constexpr size_t kElementsPerThread = kTotalElements / kNumThreads;
static constexpr int kWarmupRuns = 2;
static constexpr int kBenchRuns = 5;

struct BenchResult {
    double avg_ms;
    double throughput_mops; // million ops per second
};

template <typename Fn>
BenchResult run_bench(Fn&& fn) {
    // warmup
    for (int i = 0; i < kWarmupRuns; ++i) {
        fn();
    }
    // timed runs
    std::vector<double> times;
    times.reserve(kBenchRuns);
    for (int i = 0; i < kBenchRuns; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }
    double avg_ms = std::accumulate(times.begin(), times.end(), 0.0) / kBenchRuns;
    double throughput = (kTotalElements / avg_ms) / 1e3; // million ops/s
    return {avg_ms, throughput};
}

// 1) tbb::concurrent_vector parallel push_back
TEST(TbbConcurrentVectorBench, ConcurrentVectorParallelPush) {
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, kNumThreads);

    auto result = run_bench([] {
        tbb::concurrent_vector<uint64_t> cv;
        cv.reserve(kTotalElements);
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, kTotalElements),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    cv.push_back(i);
                }
            });
    });

    fmt::print("=== tbb::concurrent_vector parallel push ===\n");
    fmt::print("  Threads:    {}\n", kNumThreads);
    fmt::print("  Elements:   {}\n", kTotalElements);
    fmt::print("  Avg time:   {:.2f} ms\n", result.avg_ms);
    fmt::print("  Throughput: {:.2f} M ops/s\n\n", result.throughput_mops);
}

// 2) Per-thread std::vector parallel push_back
TEST(TbbConcurrentVectorBench, PerThreadVectorParallelPush) {
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, kNumThreads);

    auto result = run_bench([] {
        std::vector<std::vector<uint64_t>> per_thread(kNumThreads);
        for (auto& v : per_thread) {
            v.reserve(kElementsPerThread);
        }
        tbb::parallel_for(
            size_t(0), kNumThreads,
            [&](size_t tid) {
                auto& v = per_thread[tid];
                size_t start = tid * kElementsPerThread;
                size_t end = start + kElementsPerThread;
                for (size_t i = start; i < end; ++i) {
                    v.push_back(i);
                }
            });
    });

    fmt::print("=== Per-thread std::vector parallel push ===\n");
    fmt::print("  Threads:    {}\n", kNumThreads);
    fmt::print("  Elements:   {}\n", kTotalElements);
    fmt::print("  Avg time:   {:.2f} ms\n", result.avg_ms);
    fmt::print("  Throughput: {:.2f} M ops/s\n\n", result.throughput_mops);
}

// 3) Mutex-protected std::vector parallel push_back
TEST(TbbConcurrentVectorBench, MutexVectorParallelPush) {
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, kNumThreads);

    auto result = run_bench([] {
        std::vector<uint64_t> v;
        std::mutex mtx;
        v.reserve(kTotalElements);
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, kTotalElements),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    std::lock_guard<std::mutex> lock(mtx);
                    v.push_back(i);
                }
            });
    });

    fmt::print("=== Mutex std::vector parallel push ===\n");
    fmt::print("  Threads:    {}\n", kNumThreads);
    fmt::print("  Elements:   {}\n", kTotalElements);
    fmt::print("  Avg time:   {:.2f} ms\n", result.avg_ms);
    fmt::print("  Throughput: {:.2f} M ops/s\n\n", result.throughput_mops);
}

// 4) Single-thread std::vector push_back
TEST(TbbConcurrentVectorBench, SingleThreadVectorPush) {
    auto result = run_bench([] {
        std::vector<uint64_t> v;
        v.reserve(kTotalElements);
        for (size_t i = 0; i < kTotalElements; ++i) {
            v.push_back(i);
        }
    });

    fmt::print("=== Single-thread std::vector push ===\n");
    fmt::print("  Threads:    1\n");
    fmt::print("  Elements:   {}\n", kTotalElements);
    fmt::print("  Avg time:   {:.2f} ms\n", result.avg_ms);
    fmt::print("  Throughput: {:.2f} M ops/s\n\n", result.throughput_mops);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
