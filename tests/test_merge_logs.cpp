// --- START OF FILE test_merge_logs.cpp ---

#define private public
#define protected public
#include <artea/cpu/propagation/propagate_engine.hpp>
#include <artea/cpu/propagation/nbr_log_table.hpp>
#undef private
#undef protected

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <unordered_map>
#include <set>
#include <map>
#include <unordered_set>

#include <argparse/argparse.hpp>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/containers/locked_buffer.hpp>
#include <artea/cpu/index/index_graph.hpp>
#include <artea/cpu/index/neighbor.hpp>
#include <artea/cpu/utils/clear_cache.hpp>

using namespace artea::cpu;

using vertex_id_t = uint32_t;
using distance_t = float;

using LogBufferType = LockedBuffer<Neighbor<vertex_id_t, distance_t>, 64, tbb::spin_mutex>;
using EngineType = PropagateEngine<vertex_id_t, distance_t, LogBufferType>;
using GraphType = IndexGraph<vertex_id_t, distance_t, graph_direction_t::HIBRID>;
using NbrType = Neighbor<vertex_id_t, distance_t>;

struct TestConfig {
    size_t num_vertices;
    size_t init_nbrs_per_vertex;
    size_t total_ops;
    double remove_ratio;
    int num_threads;
};

enum class OpType { APPEND, REMOVE };

struct EdgeOp {
    OpType type;
    vertex_id_t src;
    vertex_id_t dst;
    distance_t distance;
};

// --- Helper: Initialize Hybrid Graph ---
void fill_initial_graph(GraphType& graph, size_t nbrs_count) {
    artea::logger.info("Initializing Graph...");

    size_t num_v = graph.get_num_vertices();
    std::vector<std::vector<NbrType>> in_buffers(num_v);
    std::vector<std::mutex> in_mutexes(num_v);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_v), [&](const tbb::blocked_range<size_t>& r) {
        std::mt19937 rng(42 + r.begin());
        std::uniform_real_distribution<float> dist_dist(0.1f, 100.0f);

        for (size_t u = r.begin(); u != r.end(); ++u) {
            auto& out_nbrs = graph.fetch_nbrs<op_direction_t::OUT>(u);
            std::set<vertex_id_t> used_ids;
            used_ids.insert(u);

            while (out_nbrs.size() < nbrs_count) {
                vertex_id_t v = rng() % num_v;
                if (used_ids.find(v) == used_ids.end()) {
                    distance_t dist = dist_dist(rng);
                    out_nbrs.emplace_back(v, dist, false);
                    {
                        std::lock_guard<std::mutex> lock(in_mutexes[v]);
                        in_buffers[v].emplace_back(u, dist, false);
                    }
                    used_ids.insert(v);
                }
            }
            std::sort(out_nbrs.begin(), out_nbrs.end(), NeighborComparator<vertex_id_t, distance_t>);
        }
    });

    tbb::parallel_for(size_t(0), static_cast<size_t>(num_v), [&](size_t v) {
        auto& in_nbrs = graph.fetch_nbrs<op_direction_t::IN>(v);
        in_nbrs = std::move(in_buffers[v]);
        std::sort(in_nbrs.begin(), in_nbrs.end(), NeighborComparator<vertex_id_t, distance_t>);
    });

    artea::logger.success("Graph Initialization Complete.");
}

/*--- Workload Generation ---*/

// --- Helper: Deterministic Distance Generator ---
// Ensures that if multiple threads generate an APPEND for (u, v),
// they calculate the exact same distance without communication.
inline distance_t hash_distance(vertex_id_t u, vertex_id_t v) {
    // Simple hash to map (u, v) to [0.1, 100.0]
    uint64_t h = (uint64_t)u * 0x9e3779b9 + (uint64_t)v;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;

    // Map uint64 to float range [0.1, 100.0]
    return 0.1f + (static_cast<float>(h % 99900) / 1000.0f);
}

// --- Generate Workload (Parallel Optimized) ---
std::vector<EdgeOp> generate_workload(const GraphType& graph, const TestConfig& config) {
    artea::logger.info("Generating workload (Parallel)...");
    auto start_time = std::chrono::high_resolution_clock::now();

    const size_t num_v = graph.get_num_vertices();

    // --- Step 1: Flatten existing edges for fast random sampling (Parallel) ---
    // We need a list of all existing edges to perform REMOVE operations validly.

    // 1.1 Prefix sum to calculate offsets
    std::vector<size_t> offsets(num_v + 1);
    offsets[0] = 0;
    // Serial scan for offsets is fast enough, or parallel scan if needed.
    // Given fetching sizes is fast, we'll just loop.
    for (size_t i = 0; i < num_v; ++i) {
        offsets[i+1] = offsets[i] + graph.fetch_nbrs<op_direction_t::OUT>(i).size();
    }
    size_t total_initial_edges = offsets[num_v];

    struct SimpleEdge { vertex_id_t u; vertex_id_t v; distance_t dist; };
    std::vector<SimpleEdge> initial_edges(total_initial_edges);

    // 1.2 Fill edge list in parallel
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_v), [&](const tbb::blocked_range<size_t>& r) {
        for (size_t u = r.begin(); u != r.end(); ++u) {
            const auto& nbrs = graph.fetch_nbrs<op_direction_t::OUT>(u);
            size_t idx = offsets[u];
            for (const auto& nbr : nbrs) {
                initial_edges[idx++] = {static_cast<vertex_id_t>(u), nbr.get_id(), nbr.get_distance()};
            }
        }
    });

    artea::logger.info(fmt::format("  -> Indexed {} initial edges.", total_initial_edges));

    // --- Step 2: Generate Ops in Parallel ---

    // Use Thread Local Storage to avoid lock contention
    using LocalOps = std::vector<EdgeOp>;
    tbb::enumerable_thread_specific<LocalOps> tls_ops;

    tbb::parallel_for(tbb::blocked_range<size_t>(0, config.total_ops),
        [&](const tbb::blocked_range<size_t>& r) {
            LocalOps& local = tls_ops.local();
            // Pre-reserve approximate size to reduce allocation
            if (local.empty()) local.reserve((config.total_ops / config.num_threads) * 1.2);

            // Seed RNG uniquely per thread/chunk
            std::mt19937 rng(12345 + r.begin());
            std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
            std::uniform_int_distribution<vertex_id_t> vid_dist(0, num_v - 1);

            // Distribution for picking an edge to remove
            std::uniform_int_distribution<size_t> edge_idx_dist(0, total_initial_edges - 1);

            for (size_t i = r.begin(); i != r.end(); ++i) {
                bool do_remove = (total_initial_edges > 0) && (prob_dist(rng) < config.remove_ratio);

                if (do_remove) {
                    // Pick a random existing edge
                    // Note: This only picks from INITIAL edges.
                    // This is sufficient to create conflict and test correctness.
                    const auto& edge = initial_edges[edge_idx_dist(rng)];

                    EdgeOp op;
                    op.type = OpType::REMOVE;
                    op.src = edge.u;
                    op.dst = edge.v;
                    op.distance = edge.dist; // Exact distance from graph
                    local.push_back(op);
                } else {
                    // Pick random u, v
                    vertex_id_t u = vid_dist(rng);
                    vertex_id_t v = vid_dist(rng);
                    if (u == v) { i--; continue; } // Retry

                    EdgeOp op;
                    op.type = OpType::APPEND;
                    op.src = u;
                    op.dst = v;

                    // Consistency Check:
                    // 1. Check if it exists in the initial graph (Read-only check)
                    const auto& out_nbrs = graph.fetch_nbrs<op_direction_t::OUT>(u);
                    bool found = false;
                    // Linear scan is very fast for small N (N < 64)
                    for (const auto& nbr : out_nbrs) {
                        if (nbr.get_id() == v) {
                            op.distance = nbr.get_distance();
                            found = true;
                            break;
                        }
                    }

                    // 2. If not in graph, use Deterministic Hash
                    // This ensures if Thread A and Thread B both add (u,v), they get same dist.
                    if (!found) {
                        op.distance = hash_distance(u, v);
                    }

                    local.push_back(op);
                }
            }
        }
    );

    // --- Step 3: Merge Results ---
    std::vector<EdgeOp> final_ops;
    final_ops.reserve(config.total_ops);

    for (auto& local : tls_ops) {
        final_ops.insert(final_ops.end(), local.begin(), local.end());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    artea::logger.success(fmt::format("Workload Generated: {} ops in {:.3f} ms", final_ops.size(), elapsed.count()));
    return final_ops;
}

void populate_logs(EngineType& engine, const std::vector<EdgeOp>& ops) {
    auto& log_table = engine._log_table;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, ops.size()),
        [&](const tbb::blocked_range<size_t>& r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
            const auto& op = ops[i];
            if (op.type == OpType::APPEND) {
                log_table.add_append_log(op.src, op.dst, op.distance, op_direction_t::OUT);
                log_table.add_append_log(op.dst, op.src, op.distance, op_direction_t::IN);
            } else {
                log_table.add_remove_log(op.src, op.dst, op.distance, op_direction_t::OUT);
                log_table.add_remove_log(op.dst, op.src, op.distance, op_direction_t::IN);
            }
        }
    });
}

// --- Serial Baseline (Batch Semantics: Remove Wins) ---
void run_serial_baseline(GraphType& graph, const std::vector<EdgeOp>& ops) {
    artea::logger.info("Running Serial Baseline (Batch Semantics: Remove Wins)...");
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Group ops by vertex for OUT and IN directions
    struct VertexOps {
        std::unordered_set<vertex_id_t> removes; // Set of IDs to remove
        std::unordered_map<vertex_id_t, distance_t> appends; // Map ID -> Dist to append
    };

    std::vector<VertexOps> out_ops(graph.get_num_vertices());
    std::vector<VertexOps> in_ops(graph.get_num_vertices());

    for (const auto& op : ops) {
        if (op.type == OpType::REMOVE) {
            out_ops[op.src].removes.insert(op.dst);
            in_ops[op.dst].removes.insert(op.src);
        } else {
            out_ops[op.src].appends[op.dst] = op.distance;
            in_ops[op.dst].appends[op.src] = op.distance;
        }
    }

    // 2. Apply logic per vertex
    auto apply_batch = [](std::vector<NbrType>& nbrs, const VertexOps& v_ops) {
        std::vector<NbrType> next_nbrs;
        next_nbrs.reserve(nbrs.size());

        // A. Keep existing neighbors UNLESS they are in the remove set
        for (const auto& nbr : nbrs) {
            if (v_ops.removes.find(nbr.get_id()) == v_ops.removes.end()) {
                next_nbrs.push_back(nbr);
            }
        }

        // B. Add appended neighbors UNLESS they are in the remove set or already exist
        // (Note: Already exist check is handled by checking next_nbrs contents,
        //  but for speed we can use a set or just linear scan since N is small)
        for (const auto& kv : v_ops.appends) {
            vertex_id_t id = kv.first;
            distance_t dist = kv.second;

            // Strict Rule: If it's in the remove set, it stays removed (Remove Wins)
            if (v_ops.removes.find(id) != v_ops.removes.end()) {
                continue;
            }

            // Check if already in next_nbrs (Deduplication)
            bool exists = false;
            for (const auto& n : next_nbrs) {
                if (n.get_id() == id) { exists = true; break; }
            }
            if (!exists) {
                next_nbrs.emplace_back(id, dist, true);
            }
        }

        // C. Sort
        std::sort(next_nbrs.begin(), next_nbrs.end(), NeighborComparator<vertex_id_t, distance_t>);
        nbrs = std::move(next_nbrs);
    };

    // Parallel application for baseline (safe since we work on disjoint vertices)
    tbb::parallel_for(size_t(0), static_cast<size_t>(graph.get_num_vertices()), [&](size_t v) {
        apply_batch(graph.fetch_nbrs<op_direction_t::OUT>(v), out_ops[v]);
        apply_batch(graph.fetch_nbrs<op_direction_t::IN>(v), in_ops[v]);
    });

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    artea::logger.info(fmt::format("[Serial Baseline] Time: {:.3f} ms", elapsed.count()));
}

template <bool Selective>
void run_engine_benchmark(const std::string& name, GraphType& graph, const std::vector<EdgeOp>& ops) {
    EngineType engine(graph);
    populate_logs(engine, ops);
    clear_cpu_cache();

    artea::logger.info(fmt::format("Running Engine [{}]...", name));
    auto start = std::chrono::high_resolution_clock::now();

    if constexpr (!Selective) {
        tbb::parallel_for(size_t(0), static_cast<size_t>(graph.get_num_vertices()), [&](size_t v) {
            engine.get_log_table().apply_logs(v, graph, op_direction_t::OUT);
            engine.get_log_table().apply_logs(v, graph, op_direction_t::IN);
        });
    } else {
        engine.merge_logs<Selective>();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    artea::logger.info(fmt::format("[{}] Time: {:.3f} ms", name, elapsed.count()));
}

bool verify_graphs(const GraphType& truth, const GraphType& test) {
    std::atomic<bool> success{true};
    tbb::parallel_for(size_t(0), static_cast<size_t>(truth.get_num_vertices()), [&](size_t v) {
        if (!success) return;

        auto check_dir = [&](const std::vector<NbrType>& gt, const std::vector<NbrType>& t, const char* label) {
            if (gt.size() != t.size()) {
                artea::logger.error(fmt::format("Mismatch {} size at v{}: GT={} vs Test={}", label, v, gt.size(), t.size()));
                success = false; return;
            }
            for (size_t i = 0; i < gt.size(); ++i) {
                if (gt[i].get_id() != t[i].get_id() || std::abs(gt[i].get_distance() - t[i].get_distance()) > 1e-6) {
                    artea::logger.error(fmt::format("Mismatch {} content at v{}[{}]: GT{{{},{}}} vs Test{{{},{}}}",
                        label, v, i, gt[i].get_id(), gt[i].get_distance(), t[i].get_id(), t[i].get_distance()));
                    success = false; return;
                }
            }
        };

        check_dir(truth.fetch_nbrs<op_direction_t::OUT>(v), test.fetch_nbrs<op_direction_t::OUT>(v), "OUT");
        if (!success) return;
        check_dir(truth.fetch_nbrs<op_direction_t::IN>(v), test.fetch_nbrs<op_direction_t::IN>(v), "IN");
    });
    return success;
}

void deep_copy_graph(const GraphType& src, GraphType& dst) {
    tbb::parallel_for(size_t(0), static_cast<size_t>(src.get_num_vertices()), [&](size_t v) {
        dst.fetch_nbrs<op_direction_t::OUT>(v) = src.fetch_nbrs<op_direction_t::OUT>(v);
        dst.fetch_nbrs<op_direction_t::IN>(v) = src.fetch_nbrs<op_direction_t::IN>(v);
    });
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("test_merge_logs");
    program.add_argument("-v", "--vertices").scan<'u', size_t>().default_value(size_t(50000));
    program.add_argument("-n", "--neighbors").scan<'u', size_t>().default_value(size_t(20));
    program.add_argument("-o", "--ops").scan<'u', size_t>().default_value(size_t(100000));
    program.add_argument("-r", "--remove").scan<'g', double>().default_value(0.3);
    program.add_argument("-t", "--threads").scan<'i', int>().default_value(int(std::thread::hardware_concurrency()));

    try { program.parse_args(argc, argv); }
    catch (const std::runtime_error& err) { std::cerr << err.what() << std::endl; return 1; }

    TestConfig config;
    config.num_vertices = program.get<size_t>("--vertices");
    config.init_nbrs_per_vertex = program.get<size_t>("--neighbors");
    config.total_ops = program.get<size_t>("--ops");
    config.remove_ratio = program.get<double>("--remove");
    config.num_threads = program.get<int>("--threads");

    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, config.num_threads);

    artea::logger.info(fmt::format("Vertices: {}, Neighbors: {}, Total Ops: {}",
        config.num_vertices, config.init_nbrs_per_vertex, config.total_ops));

    GraphType master_graph(config.num_vertices, config.init_nbrs_per_vertex * 2);
    fill_initial_graph(master_graph, config.init_nbrs_per_vertex);

    auto workload = generate_workload(master_graph, config);

    GraphType gt_graph(config.num_vertices, config.init_nbrs_per_vertex * 2);
    deep_copy_graph(master_graph, gt_graph);
    run_serial_baseline(gt_graph, workload);

    std::cout << "------------------------------------------------------------" << std::endl;

    {
        GraphType test_graph(config.num_vertices, config.init_nbrs_per_vertex * 2);
        deep_copy_graph(master_graph, test_graph);
        run_engine_benchmark<false>("Parallel (No Selective)", test_graph, workload);
        if (verify_graphs(gt_graph, test_graph)) artea::logger.success("-> Verification PASSED");
        else { artea::logger.error("-> Verification FAILED"); return 1; }
    }

    {
        GraphType test_graph(config.num_vertices, config.init_nbrs_per_vertex * 2);
        deep_copy_graph(master_graph, test_graph);
        run_engine_benchmark<true>("Parallel (With Selective)", test_graph, workload);
        if (verify_graphs(gt_graph, test_graph)) artea::logger.success("-> Verification PASSED");
        else { artea::logger.error("-> Verification FAILED"); return 1; }
    }

    std::cout << "------------------------------------------------------------" << std::endl;
    return 0;
}