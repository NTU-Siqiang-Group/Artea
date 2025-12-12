/*
 * @FilePath: /Artea/benchmarks/benchmark_vector_numa.cpp
 * @Description: Benchmark for NUMA access latency.
 *               FIXED:
 *               1. Manually parses /sys/devices/system/node to bypass broken TBB detection.
 *               2. Uses sched_setaffinity to force pinning when TBB fails to see nodes.
 *               3. Implements PARALLEL cache flushing to ensure cold reads on all cores.
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <atomic>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <thread>
#include <cstring>
#include <filesystem>

// TBB Includes
#include <tbb/tbb.h>
#include <tbb/task_arena.h>

// Linux Native
#include <sched.h>
#include <unistd.h>

// Argparse Include
#include <argparse/argparse.hpp>

// Artea Includes
#include <artea/definitions.hpp>
#include <artea/common/logger.hpp>
#include <artea/cpu/containers/vector_dataset.hpp>
#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/utils/random_seq.hpp>

using namespace artea;
using namespace artea::cpu;

// --- System Topology Parser ---
// TBB failed us, so we parse Linux sysfs manually to find the truth.
struct NumaNodeInfo {
    int node_id;
    std::vector<int> cpu_list;
};

// Parses /sys/devices/system/node/nodeX/cpulist (e.g., "0-15,32-47")
std::vector<int> parse_cpu_list(const std::string& range_str) {
    std::vector<int> cpus;
    std::stringstream ss(range_str);
    std::string segment;
    while (std::getline(ss, segment, ',')) {
        size_t dash = segment.find('-');
        if (dash != std::string::npos) {
            int start = std::stoi(segment.substr(0, dash));
            int end = std::stoi(segment.substr(dash + 1));
            for (int i = start; i <= end; ++i) cpus.push_back(i);
        } else {
            cpus.push_back(std::stoi(segment));
        }
    }
    return cpus;
}

std::vector<NumaNodeInfo> get_manual_topology() {
    std::vector<NumaNodeInfo> topology;
    std::string base_path = "/sys/devices/system/node";

    // Check node0, node1, node2... until not found
    for (int i = 0; ; ++i) {
        std::string node_dir = base_path + "/node" + std::to_string(i);
        if (!std::filesystem::exists(node_dir)) break;

        std::ifstream f(node_dir + "/cpulist");
        if (f.is_open()) {
            std::string line;
            std::getline(f, line);
            // Remove newline
            if (!line.empty() && line.back() == '\n') line.pop_back();
            topology.push_back({i, parse_cpu_list(line)});
        }
    }
    return topology;
}

// Force pin current thread to a specific core
void pin_thread_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        // Just warn, don't crash
        // perror("sched_setaffinity");
    }
}

// ---------------------------------------------------------

using VecElemT = float;
using VecNumT = uint32_t;

struct alignas(64) ThreadLocalBlock {
    VecElemT* data_ptr = nullptr;
    size_t num_vecs = 0;
    size_t vec_dim = 0;
    int numa_node_idx = -1;
    int global_thread_id = -1;
};

std::vector<ThreadLocalBlock> global_memory_table;
std::vector<NumaNodeInfo> sys_topology;

// --- Parallel Cache Flushing ---
// Must run on ALL threads to effectively evict private L1/L2 caches
void parallel_flush_cache(int total_threads) {
    // 1GB Dummy Buffer
    constexpr size_t FLUSH_SIZE_BYTES = 1024UL * 1024UL * 1024UL;
    constexpr size_t ELEM_COUNT = FLUSH_SIZE_BYTES / sizeof(float);
    static MmapAllocator<float> alloc;
    static float* dummy_buffer = alloc.allocate(ELEM_COUNT);

    // Initialize once
    static bool inited = false;
    if (!inited) {
        std::fill(dummy_buffer, dummy_buffer + ELEM_COUNT, 1.0f);
        inited = true;
    }

    size_t chunk_size = ELEM_COUNT / total_threads;

    tbb::parallel_for(0, total_threads, [&](int t) {
        // Simple linear scan write to force eviction
        size_t start = t * chunk_size;
        size_t end = (t == total_threads - 1) ? ELEM_COUNT : start + chunk_size;
        float val = 1.0f + t;

        // Stride 16 (64 bytes) to hit every cache line
        for (size_t i = start; i < end; i += 16) {
            dummy_buffer[i] = val;
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
    });
}

// Helper to execute code on specific topology
// Replaces TBB constraints with manual pinning
template <typename Func>
void execute_on_topology_manual(int total_threads, Func kernel_func) {
    size_t num_nodes = sys_topology.size();
    int threads_per_node = total_threads / num_nodes;

    // Use a flat parallel_for, but inside we determine where we belong
    tbb::parallel_for(0, total_threads, [&](int global_tid) {
        // 1. Calculate which node and local index this thread belongs to
        int node_idx = global_tid / threads_per_node;
        int local_idx = global_tid % threads_per_node;

        if (node_idx >= num_nodes) return; // Should not happen

        // 2. Identify the target physical CPU core
        // We cycle through the available CPUs on that node
        const auto& cpus = sys_topology[node_idx].cpu_list;
        int target_cpu = cpus[local_idx % cpus.size()];

        // 3. HARD PINNING
        pin_thread_to_cpu(target_cpu);

        // 4. Run kernel
        kernel_func(global_tid, node_idx);
    });
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("benchmark_vector_numa");
    program.add_argument("-c", "--config").default_value(std::string("./datasets.json")).help("Path to config");
    program.add_argument("-d", "--dataset").required().help("Dataset name");
    program.add_argument("-t", "--threads").default_value(0).scan<'i', int>().help("Threads (0=auto)");
    program.add_argument("-l", "--limit").default_value(size_t(0)).scan<'u', size_t>().help("Limit vecs");

    try { program.parse_args(argc, argv); }
    catch (const std::exception& err) { std::cerr << err.what() << std::endl; return 1; }

    std::string config_path = program.get<std::string>("--config");
    std::string dataset_name = program.get<std::string>("--dataset");
    int req_num_threads = program.get<int>("--threads");
    size_t limit_vecs = program.get<size_t>("--limit");

    // --- 1. Detect Topology Manually ---
    sys_topology = get_manual_topology();
    size_t num_nodes = sys_topology.size();

    if (num_nodes == 0) {
        artea::logger.error("Failed to detect NUMA nodes from /sys. Is /sys mounted?");
        return 1;
    }

    if (req_num_threads <= 0) {
        // Sum up all CPUs
        req_num_threads = 0;
        for (const auto& node : sys_topology) req_num_threads += node.cpu_list.size();
        artea::logger.info(fmt::format("Auto-detected max concurrency: {} threads", req_num_threads));
    }

    int threads_per_node = req_num_threads / num_nodes;
    int actual_total_threads = threads_per_node * num_nodes;

    artea::logger.info(fmt::format("Manually detected {} NUMA nodes via /sys.", num_nodes));
    for (const auto& node : sys_topology) {
        artea::logger.info(fmt::format("  Node {}: {} CPUs", node.node_id, node.cpu_list.size()));
    }
    artea::logger.info(fmt::format("Using {} threads per node (Total {}).", threads_per_node, actual_total_threads));

    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, actual_total_threads);
    global_memory_table.resize(actual_total_threads);

    // --- 2. Load Dataset ---
    artea::logger.info("Step 1: Loading Dataset...");
    VectorDataset<VecNumT, VecElemT> dataset(config_path, dataset_name);
    auto& base_vecs = dataset.get_base_vecs();

    size_t total_available_vecs = base_vecs.get_num_vecs();
    size_t vec_dim = base_vecs.get_vec_dim();
    size_t dataset_vecs = (limit_vecs > 0 && limit_vecs < total_available_vecs) ? limit_vecs : total_available_vecs;
    size_t vecs_per_thread = dataset_vecs / actual_total_threads;

    artea::logger.info(fmt::format("Total Vecs: {}, Dim: {}, Vecs/Thread: {}", dataset_vecs, vec_dim, vecs_per_thread));

    // --- 3. Initialize Memory ---
    artea::logger.info("Step 2: Initializing NUMA layout (Manual Pinning & First-touch)...");
    std::vector<std::vector<uint32_t>> thread_random_indices(actual_total_threads);
    const size_t OPS_PER_THREAD = 1000000;

    execute_on_topology_manual(actual_total_threads, [&](int global_tid, int node_idx) {
        MmapAllocator<VecElemT> allocator;
        size_t total_elements = vecs_per_thread * vec_dim;
        VecElemT* local_ptr = allocator.allocate(total_elements);

        const VecElemT* src_ptr = base_vecs.get(global_tid * vecs_per_thread);
        std::copy(src_ptr, src_ptr + total_elements, local_ptr);

        global_memory_table[global_tid].data_ptr = local_ptr;
        global_memory_table[global_tid].num_vecs = vecs_per_thread;
        global_memory_table[global_tid].vec_dim = vec_dim;
        global_memory_table[global_tid].numa_node_idx = node_idx;
        global_memory_table[global_tid].global_thread_id = global_tid;

        RandomSeq<uint32_t> rng(vecs_per_thread);
        thread_random_indices[global_tid].resize(OPS_PER_THREAD);
        rng.generate(thread_random_indices[global_tid], OPS_PER_THREAD);
    });

    artea::logger.success("Memory initialized.");

    // --- Benchmark Runner ---
    auto run_benchmark = [&](const std::string& name, bool is_write, auto get_target_block_func) {
        artea::logger.info("Flushing caches (Parallel)...");
        parallel_flush_cache(actual_total_threads);

        std::atomic<size_t> total_dummy(0);
        auto start = std::chrono::high_resolution_clock::now();

        execute_on_topology_manual(actual_total_threads, [&](int global_tid, int node_idx) {
            size_t local_dummy = 0;
            const auto& my_block = global_memory_table[global_tid];
            ThreadLocalBlock* target_block = get_target_block_func(global_tid, my_block);

            if (target_block) {
                VecElemT* base_addr = target_block->data_ptr;
                size_t dim = target_block->vec_dim;
                const auto& indices = thread_random_indices[global_tid];
                VecElemT write_val = static_cast<VecElemT>(global_tid);

                if (is_write) {
                    for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                        uint32_t vec_idx = indices[i];
                        VecElemT* vec_ptr = base_addr + (size_t)vec_idx * dim;
                        for (size_t d = 0; d < dim; ++d) vec_ptr[d] = write_val;
                    }
                } else {
                    for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                        uint32_t vec_idx = indices[i];
                        VecElemT* vec_ptr = base_addr + (size_t)vec_idx * dim;
                        VecElemT sum = 0;
                        for (size_t d = 0; d < dim; ++d) sum += vec_ptr[d];
                        local_dummy += static_cast<size_t>(sum);
                    }
                }
            }
            total_dummy += local_dummy;
        });

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        double total_ops = (double)actual_total_threads * OPS_PER_THREAD;
        double throughput = total_ops / diff.count() / 1e6;

        std::string check_msg = is_write ? "" : fmt::format("(Check: {})", total_dummy.load());
        artea::logger.info(fmt::format("[{}] Time: {:.4f} s, Throughput: {:.2f} M Vec/s {}",
            name, diff.count(), throughput, check_msg));
    };

    // --- Define Target Getters ---
    auto get_local = [&](int tid, const ThreadLocalBlock& mb) { return &global_memory_table[tid]; };

    auto get_intra = [&](int tid, const ThreadLocalBlock& mb) -> ThreadLocalBlock* {
        int node_start_tid = mb.numa_node_idx * threads_per_node;
        int local_idx = tid - node_start_tid;
        int target_local = (local_idx + 1) % threads_per_node;
        int target_tid = node_start_tid + target_local;

        return &global_memory_table[target_tid];
    };

    // Inter-NUMA
    // Node 0, Thread i -> Node 1, Thread i
    auto get_inter = [&](int tid, const ThreadLocalBlock& mb) -> ThreadLocalBlock* {
        int target_tid = (tid + threads_per_node) % actual_total_threads;
        if (global_memory_table[target_tid].numa_node_idx == mb.numa_node_idx) {
            return nullptr;
        }

        return &global_memory_table[target_tid];
    };

    artea::logger.info("--- Starting READ Benchmarks ---");
    run_benchmark("Read-Local", false, get_local);
    if (threads_per_node > 1) run_benchmark("Read-IntraNUMA", false, get_intra);
    if (num_nodes > 1) run_benchmark("Read-InterNUMA", false, get_inter);

    artea::logger.info("--- Starting WRITE Benchmarks ---");
    run_benchmark("Write-Local", true, get_local);
    if (threads_per_node > 1) run_benchmark("Write-IntraNUMA", true, get_intra);
    if (num_nodes > 1) run_benchmark("Write-InterNUMA", true, get_inter);

    // --- Cleanup ---
    MmapAllocator<VecElemT> deallocator;
    for(auto& block : global_memory_table) if(block.data_ptr) deallocator.deallocate(block.data_ptr, block.num_vecs * block.vec_dim);

    return 0;
}