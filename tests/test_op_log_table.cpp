/*
 * @FilePath: /Artea/tests/test_op_log_table.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Performance (Write/Read) and Correctness test for OpLogTable.
 * @Note: Updated for new GraphOperationLog with op_type and reordered fields.
 */

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cassert>
#include <thread>
#include <mutex>
#include <functional>
#include <random>

// Argument Parser
#include <argparse/argparse.hpp>

// TBB Headers
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/spin_mutex.h>

// Artea Headers
#include <artea/common/logger.hpp>
#include <artea/cpu/containers/concurrent_buffer.hpp>
#include <artea/cpu/containers/locked_buffer.hpp>
#include <artea/cpu/containers/tbb_buffer.hpp>
#include <artea/cpu/propagation/op_log_table.hpp>
#include <artea/cpu/propagation/graph_op_log.hpp>
#include <artea/cpu/utils/clear_cache.hpp>
#include <artea/definitions.hpp>

using namespace artea::cpu;

// --- Configuration & Types ---
using vertex_id_t = uint32_t;
using distance_t = float;
using log_t = GraphOperationLog<vertex_id_t, distance_t>;

constexpr size_t BUF_SIZE = 64;

// --- Helper: Workload Data Structure ---
struct TestOp {
    vertex_id_t target_vid;
    artea::cpu::graph_op_t op_type; // [New]
    artea::cpu::op_direction_t op_direction;
    vertex_id_t old_neighbor_id;
    vertex_id_t new_neighbor_id;
    distance_t distance;
};

// --- Helper: Comparator for Verification ---
struct OpComparator {
    bool operator()(const log_t& a, const log_t& b) const {
        // Sort by fields in order of importance or memory layout
        if (a.op_type != b.op_type) return a.op_type < b.op_type;
        if (a.op_direction != b.op_direction) return a.op_direction < b.op_direction;
        if (a.old_neighbor_id != b.old_neighbor_id) return a.old_neighbor_id < b.old_neighbor_id;
        if (a.new_neighbor_id != b.new_neighbor_id) return a.new_neighbor_id < b.new_neighbor_id;
        return a.new_edge_dist < b.new_edge_dist;
    }
};

bool ops_equal(const log_t& a, const log_t& b) {
    return a.op_type == b.op_type &&
           a.op_direction == b.op_direction &&
           a.old_neighbor_id == b.old_neighbor_id &&
           a.new_neighbor_id == b.new_neighbor_id &&
           std::abs(a.new_edge_dist - b.new_edge_dist) < 1e-5;
}

// -----------------------------------------------------------------------------
// Baseline: Serial Ground Truth Generation
// -----------------------------------------------------------------------------
auto run_serial_baseline(size_t num_vertices, const std::vector<TestOp>& workload)
    -> std::vector<std::vector<log_t>>
{
    artea::logger.info("Generating [Serial Baseline] Ground Truth...");
    std::vector<std::vector<log_t>> table(num_vertices);
    for (const auto& op : workload) {
        // [Modified] Constructor: op_type, direction, old, new, dist
        table[op.target_vid].emplace_back(
            op.op_type,
            op.op_direction,
            op.old_neighbor_id,
            op.new_neighbor_id,
            op.distance
        );
    }
    // Sort for later comparison
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_vertices),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t v = r.begin(); v != r.end(); ++v) {
                std::sort(table[v].begin(), table[v].end(), OpComparator());
            }
        }
    );
    return table;
}

// -----------------------------------------------------------------------------
// PART 1: Verification Pass (Correctness Check via get_log_container)
// -----------------------------------------------------------------------------
template <typename TableType>
void run_verification(
    const std::string& name,
    size_t num_vertices,
    size_t total_ops,
    const std::vector<TestOp>& workload,
    const std::vector<std::vector<log_t>>& ground_truth
) {
    artea::logger.info(fmt::format("Verifying [{}] correctness...", name));

    TableType table(num_vertices);

    // 1. Concurrent Insertion
    tbb::parallel_for(tbb::blocked_range<size_t>(0, total_ops),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                // [Modified] Passing op_type and new order to append_log
                // Assumed Signature: append_log(executor, op_type, direction, old, new, dist)
                table.append_log(
                    workload[i].target_vid,
                    workload[i].op_type,
                    workload[i].op_direction,
                    workload[i].old_neighbor_id,
                    workload[i].new_neighbor_id,
                    workload[i].distance
                );
            }
        }
    );

    // 2. Verification (Read via get_log_container)
    std::atomic<bool> passed{true};

    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_vertices),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t v = r.begin(); v != r.end(); ++v) {
                if (!passed) return;

                const auto& container = table.get_log_container(v);

                const auto& gt_vec = ground_truth[v];

                if (container.size() != gt_vec.size()) {
                    artea::logger.error(fmt::format("Size mismatch at v{}: Expected {}, Got {}",
                        v, gt_vec.size(), container.size()));
                    passed = false;
                    return;
                }

                // Copy to sort (don't modify the buffer in place)
                std::vector<log_t> sorted_res(container.begin(), container.end());
                std::sort(sorted_res.begin(), sorted_res.end(), OpComparator());

                for (size_t k = 0; k < gt_vec.size(); ++k) {
                    if (!ops_equal(gt_vec[k], sorted_res[k])) {
                        passed = false;
                        return;
                    }
                }
            }
        }
    );

    if (passed) {
        artea::logger.success(fmt::format("  [PASSED] {} Content Correctness.", name));
    } else {
        artea::logger.error(fmt::format("  [FAILED] {} Content Verification Failed!", name));
        exit(1);
    }
}

// -----------------------------------------------------------------------------
// PART 2: Performance Benchmark (Write & Read)
// -----------------------------------------------------------------------------
template <typename TableType>
void run_performance_benchmark(
    const std::string& name,
    size_t num_vertices,
    size_t total_ops,
    const std::vector<TestOp>& workload
) {
    clear_cpu_cache();
    TableType table(num_vertices);

    // --- Benchmark 1: Parallel Write (Append) ---
    auto start_add = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(tbb::blocked_range<size_t>(0, total_ops),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const auto& op = workload[i];
                // [Modified] Passing op_type and new order
                table.append_log(
                    op.target_vid,
                    op.op_type,
                    op.op_direction,
                    op.old_neighbor_id,
                    op.new_neighbor_id,
                    op.distance
                );
            }
        }
    );
    auto end_add = std::chrono::high_resolution_clock::now();

    // --- Benchmark 2: Sequential Read (Scan) ---
    clear_cpu_cache();
    auto start_seq_read = std::chrono::high_resolution_clock::now();

    std::atomic<size_t> total_elements_read{0};

    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_vertices),
        [&](const tbb::blocked_range<size_t>& r) {
            size_t local_count = 0;
            volatile distance_t dummy_sum = 0;
            for (size_t v = r.begin(); v != r.end(); ++v) {
                const auto& container = table.get_log_container(v);
                for (const auto& log : container) {
                    dummy_sum = dummy_sum + log.new_edge_dist;
                    local_count++;
                }
            }
            total_elements_read.fetch_add(local_count, std::memory_order_relaxed);
        }
    );
    auto end_seq_read = std::chrono::high_resolution_clock::now();

    // --- Benchmark 3: Random Read (Random Access) ---
    size_t total_items = total_elements_read.load();
    size_t random_access_count = std::min(total_items, size_t(10000000)); // Limit to 10M reads

    struct ReadReq { vertex_id_t v; size_t idx; };
    std::vector<ReadReq> read_workload;
    read_workload.reserve(random_access_count);

    // Generate indices
    {
        std::mt19937 rng(42);
        std::vector<vertex_id_t> non_empty_vertices;
        for(size_t v=0; v<num_vertices; ++v) {
            if(table.get_log_container(v).size() > 0) non_empty_vertices.push_back(v);
        }

        if (!non_empty_vertices.empty()) {
            std::uniform_int_distribution<size_t> v_dist(0, non_empty_vertices.size() - 1);
            for(size_t i=0; i<random_access_count; ++i) {
                vertex_id_t v = non_empty_vertices[v_dist(rng)];
                size_t size = table.get_log_container(v).size();
                std::uniform_int_distribution<size_t> idx_dist(0, size - 1);
                read_workload.push_back({v, idx_dist(rng)});
            }
        }
    }

    clear_cpu_cache();
    auto start_rnd_read = std::chrono::high_resolution_clock::now();

    tbb::parallel_for(tbb::blocked_range<size_t>(0, read_workload.size()),
        [&](const tbb::blocked_range<size_t>& r) {
            volatile distance_t dummy_val = 0;
            for (size_t i = r.begin(); i != r.end(); ++i) {
                const auto& req = read_workload[i];
                const auto& val = table.get_log_container(req.v)[req.idx];
                dummy_val = val.new_edge_dist;
            }
        }
    );
    auto end_rnd_read = std::chrono::high_resolution_clock::now();

    // --- Report ---
    auto calc_throughput = [](size_t count, std::chrono::duration<double, std::milli> ms) {
        return count / 1e6 / (ms.count() / 1000.0);
    };

    auto calc_latency = [](size_t count, std::chrono::duration<double, std::milli> ms) {
        return (ms.count() * 1000.0) / count; // microseconds
    };

    std::chrono::duration<double, std::milli> add_ms = end_add - start_add;
    std::chrono::duration<double, std::milli> seq_read_ms = end_seq_read - start_seq_read;
    std::chrono::duration<double, std::milli> rnd_read_ms = end_rnd_read - start_rnd_read;

    artea::logger.info(fmt::format("[{}] Performance:", name));

    // Write
    artea::logger.info(fmt::format("  -> Write (Append):     {:.2f} Mops/s ({:.2f} ms)",
        calc_throughput(total_ops, add_ms), add_ms.count()));

    // Seq Read
    artea::logger.info(fmt::format("  -> Read (Sequential):  {:.2f} Mops/s ({:.2f} ms)",
        calc_throughput(total_elements_read.load(), seq_read_ms), seq_read_ms.count()));

    // Random Read
    if (read_workload.empty()) {
        artea::logger.warn("  -> Read (Random):      Skipped (Empty table)");
    } else {
        artea::logger.info(fmt::format("  -> Read (Random):      {:.2f} Mops/s ({:.4f} us/op)",
            calc_throughput(read_workload.size(), rnd_read_ms),
            calc_latency(read_workload.size(), rnd_read_ms)));
    }
}

// -----------------------------------------------------------------------------
// Main Runner
// -----------------------------------------------------------------------------
template <typename TableType>
void run_full_suite(
    const std::string& name,
    size_t num_vertices,
    size_t total_ops,
    const std::vector<TestOp>& workload,
    const std::vector<std::vector<log_t>>& ground_truth
) {
    run_verification<TableType>(name, num_vertices, total_ops, workload, ground_truth);
    run_performance_benchmark<TableType>(name, num_vertices, total_ops, workload);
    std::cout << "------------------------------------------------------------" << std::endl;
}

int main(int argc, char* argv[]) {
    // 1. Setup Argparse
    argparse::ArgumentParser program("benchmark_op_log");
    program.add_argument("--vertices").scan<'u', size_t>().default_value(size_t(1000));
    program.add_argument("--ops").scan<'u', size_t>().default_value(size_t(5000000));
    program.add_argument("--threads").scan<'i', int>().default_value(int(std::thread::hardware_concurrency()));

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    size_t num_vertices = program.get<size_t>("--vertices");
    size_t total_ops = program.get<size_t>("--ops");
    int num_threads = program.get<int>("--threads");

    // 2. Control TBB Concurrency
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, num_threads);

    artea::logger.info(fmt::format("Config: {} vertices, {} ops, {} threads", num_vertices, total_ops, num_threads));

    // 3. Generate Workload
    std::vector<TestOp> workload(total_ops);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, total_ops),
        [&](const tbb::blocked_range<size_t>& r) {
            for (size_t i = r.begin(); i != r.end(); ++i) {
                workload[i].target_vid = (i * 7 + 13) % num_vertices;
                workload[i].new_neighbor_id = static_cast<vertex_id_t>(i);
                workload[i].old_neighbor_id = static_cast<vertex_id_t>(i + 1);
                workload[i].distance = static_cast<float>(i) * 0.001f;
                workload[i].op_direction = static_cast<artea::cpu::op_direction_t>(i % 2);
                // [Modified] Generate op_type (alternate APPEND/REPLACE)
                workload[i].op_type = (i % 3 == 0) ? graph_op_t::REPLACE : graph_op_t::APPEND;
            }
        }
    );

    // 4. Generate Ground Truth
    auto ground_truth = run_serial_baseline(num_vertices, workload);

    std::cout << "------------------------------------------------------------" << std::endl;

    // 1. Spin Mutex
    using SpinMutexBufType = LockedBuffer<log_t, BUF_SIZE, tbb::spin_mutex>;
    using SpinMutexTable = OpLogTable<vertex_id_t, distance_t, SpinMutexBufType>;
    run_full_suite<SpinMutexTable>("SpinMutex Table", num_vertices, total_ops, workload, ground_truth);

    // 2. Std Mutex
    using MutexBufType = LockedBuffer<log_t, BUF_SIZE, std::mutex>;
    using MutexTable = OpLogTable<vertex_id_t, distance_t, MutexBufType>;
    run_full_suite<MutexTable>("StdMutex Table", num_vertices, total_ops, workload, ground_truth);

    // 3. TBB Concurrent Vector
    using TbbBufType = TbbBuffer<log_t, BUF_SIZE>;
    using TbbTable = OpLogTable<vertex_id_t, distance_t, TbbBufType>;
    run_full_suite<TbbTable>("TBB Buffer Table", num_vertices, total_ops, workload, ground_truth);

    return 0;
}