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
 * @FilePath: /Artea/unit_tests/test_artea_graph.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: End-to-end test for artea_graph. Builds the index via the
 *               3-step pipeline (stacked r-net insertion → per-layer
 *               refinement → hierarchical writeback), then measures
 *               search recall and throughput via the compact
 *               hierarchical graph router. The L0-only single-layer
 *               baseline comparison has been moved to the dedicated
 *               vldb27-exp/compare_hier_vs_L0 binary; this test only
 *               exercises the hierarchical path.
 *
 *               All configuration comes from a bench-artea-style workload
 *               JSONC (-w/--workload): dataset-config / dataset / metric /
 *               warmup_runs / test_runs at the top level, build params from
 *               the indexes-config.artea base entry, and the search grid
 *               from its throughput_search sweeps. There are no per-param
 *               CLI flags.
 */

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <argparse/argparse.hpp>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>

using namespace artea;
using namespace artea::cpu;

using artea_index_t = typename index_traits_t::artea_graph::index_t;

// ============================================================
//  Global configuration (populated from the workload JSON in main())
// ============================================================

// One (topk, queue-size grid) entry from the workload's throughput_search.
struct SearchSweep {
    uint32_t topk;
    uint32_t queue_start;
    uint32_t queue_end;
    uint32_t queue_step;
};

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    std::string metric;

    // Stacked r-net backbone
    float    rnet_beta;
    bool     l0_radius_provided;
    float    l0_rnet_radius;
    uint32_t ul_max_nbr_size;
    uint32_t bl_max_nbr_size;
    uint32_t search_nn_qs;
    uint32_t ul_select_nbrs_qs;
    uint32_t bl_select_nbrs_qs;
    float    scale_coeffs;
    float    shifted_coeffs;
    float    l0_min_distance;

    // L0-radius auto-probe
    uint32_t probe_num_samples;
    float    probe_quantile;

    // Per-layer conv_graph-style refinement
    // (refining_max_nbr_size / reserved derived as rgraph max_nbr_size × 1.5)
    uint32_t num_build_loops;
    uint32_t num_triu_iters;
    float    prefill_ratio;
    uint32_t num_routing_loops;
    uint32_t routing_topk;
    uint32_t routing_queue_size;

    // Insertion-order randomization (see stacked_rgraph::IndexFactory).
    bool     shuffle_insertion_order;

    // When true, the inherited stacked_rgraph factory builds L0 edges
    // at insertion time; when false (default), L0 is left empty and
    // artea_graph::refine_layer seeds L0 via random prefill.
    bool     insert_on_L0;

    // Search-time params (the workload's throughput_search sweeps)
    std::vector<SearchSweep> sweeps;
    uint32_t warmup_runs;
    uint32_t test_runs;
} g_config;

namespace {

auto dump_config(const char* banner) -> void {
    std::ostringstream os;
    os << "\n=== " << banner << " ===\n"
       << "Dataset:                    " << g_config.dataset_name << "\n"
       << "Config path:                " << g_config.config_path << "\n"
       << "--- Stacked r-net backbone ---\n"
       << "rnet_beta:                  " << g_config.rnet_beta << "\n";
    if (g_config.l0_radius_provided) {
        os << "L0 radius:                  " << g_config.l0_rnet_radius
           << " (user-provided)\n";
    } else {
        os << "L0 radius:                  auto-probe ("
           << static_cast<int>(g_config.probe_quantile * 100.0f)
           << "th pct, " << g_config.probe_num_samples << " samples)\n";
    }
    os << "ul_max_nbr_size:            " << g_config.ul_max_nbr_size << "\n"
       << "bl_max_nbr_size:            " << g_config.bl_max_nbr_size << "\n"
       << "search_nn_qs:               " << g_config.search_nn_qs << "\n"
       << "ul_select_nbrs_qs:          " << g_config.ul_select_nbrs_qs << "\n"
       << "bl_select_nbrs_qs:          " << g_config.bl_select_nbrs_qs << "\n"
       << "scale_coeffs:               " << g_config.scale_coeffs
       << " (applied to both ul insertion and refinement pruning)\n"
       << "shifted_coeffs:             " << g_config.shifted_coeffs
       << " (consumed by refinement pruning only; ul insertion reads "
       << "scale_coeffs only and ignores shift)\n"
       << "l0_min_distance:            " << g_config.l0_min_distance
       << " (per-vertex L0 minimum-distance gate; consumed by refinement pruning)\n"
       << "--- Per-layer refinement ---\n"
       << "ul_refining_max_nbr_size:   "
       << static_cast<uint32_t>(g_config.ul_max_nbr_size * 1.5f)
       << " (= ul_max_nbr_size × 1.5; reserved matches)\n"
       << "bl_refining_max_nbr_size:   "
       << static_cast<uint32_t>(g_config.bl_max_nbr_size * 1.5f)
       << " (= bl_max_nbr_size × 1.5; reserved matches)\n"
       << "num_build_loops:            " << g_config.num_build_loops << "\n"
       << "num_triu_iters:             " << g_config.num_triu_iters << "\n"
       << "prefill_ratio:              " << g_config.prefill_ratio
       << " (L0 random prefill target: "
       << static_cast<uint32_t>(g_config.prefill_ratio
              * g_config.bl_max_nbr_size * 1.5f)
       << " edges/vertex)\n"
       << "num_routing_loops:          " << g_config.num_routing_loops << "\n"
       << "routing_topk:               " << g_config.routing_topk << "\n"
       << "routing_queue_size:         " << g_config.routing_queue_size << "\n"
       << "shuffle_insertion_order:    "
       << (g_config.shuffle_insertion_order ? "true" : "false") << "\n"
       << "insert_on_L0:               "
       << (g_config.insert_on_L0 ? "true" : "false")
       << " (false = L0 built via random prefill in refine_layer)\n"
       << "--- Search ---\n";
    for (const auto& sw : g_config.sweeps) {
        os << "throughput_search:          topk=" << sw.topk
           << ", queue=" << sw.queue_start << "," << sw.queue_end
           << "," << sw.queue_step << "\n";
    }
    os << "warmup_runs:                " << g_config.warmup_runs << "\n"
       << "test_runs:                  " << g_config.test_runs << "\n"
       << "========================================\n";
    std::cout << os.str();
}

}  // anonymous namespace

// ============================================================
//  DataProvider singleton
// ============================================================

class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error(
                "Config file not found: " + g_config.config_path);
        }
        ARTEA_INFO(fmt::format("Loading dataset: {} from {}",
            g_config.dataset_name, g_config.config_path));
        _dataset = std::make_unique<vector_dataset_t>(
            g_config.config_path, g_config.dataset_name);

        const auto& base_vecs = _dataset->get_base_vecs();
        ARTEA_INFO(fmt::format("Dataset loaded: {} vectors, {} dims",
            base_vecs.get_num_vecs(), base_vecs.get_vec_dim()));

        // Resolve BOTH compile-time axes: metric from the workload's
        // `metric` field, padded dim from the loaded dataset. The dataset is
        // metric/dim-independent and stays out here; the stateless dist_func
        // / metric-dependent prober run inside dispatch.
        _dataset_info = DatasetInfra{parse_metric(g_config.metric), base_vecs.get_vec_dim()};

        if (g_config.l0_radius_provided) {
            _l0_radius = g_config.l0_rnet_radius;
            ARTEA_INFO(fmt::format(
                "Using user-provided L0 rnet_radius = {:.6f}", _l0_radius));
        } else {
            infra_dispatch(_dataset_info, ARTEA_METRIC_LAMBDA(void) {
                // Stateless functor: the dimension is a compile-time trait now.
                dist_func_t<Metric, Dim> dist_func;
                dataset_prober_t<Metric, Dim> prober(base_vecs, dist_func);
                const std::vector<float> quantiles = { g_config.probe_quantile };
                ARTEA_INFO(fmt::format(
                    "Probing L0 rnet_radius ({}th pct, {} samples)...",
                    static_cast<int>(g_config.probe_quantile * 100.0f),
                    g_config.probe_num_samples));
                auto result = prober.probe(quantiles, g_config.probe_num_samples);
                _l0_radius = static_cast<float>(result.table[0][0]);
                ARTEA_INFO(fmt::format(
                    "Auto-probed L0 rnet_radius = {:.6f}", _l0_radius));
            });
        }
    }

    auto get_dataset()      -> vector_dataset_t& { return *_dataset; }
    auto get_dataset_info() const -> DatasetInfra { return _dataset_info; }
    auto get_l0_radius() const -> float          { return _l0_radius; }

private:
    DataProvider() = default;

    std::unique_ptr<vector_dataset_t> _dataset;
    DatasetInfra                      _dataset_info{};
    float                             _l0_radius = 0.0f;
};

// ============================================================
//  Fixture: build the artea_graph index once for the suite.
// ============================================================

class ArteaGraphTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto& provider = DataProvider::instance();
        const auto& base_vecs = provider.get_dataset().get_base_vecs();

        const vertex_num_t total_vertices =
            static_cast<vertex_num_t>(base_vecs.get_num_vecs());

        infra_dispatch(provider.get_dataset_info(), ARTEA_METRIC_LAMBDA(void) {
            // Stateless functor: the dimension is a compile-time trait now.
            dist_func_t<Metric, Dim> dist_func;

            artea_graph::rgraph_config_t<Metric, Dim> rgraph_config(
                g_config.rnet_beta,
                provider.get_l0_radius(),
                static_cast<vertex_num_t>(g_config.search_nn_qs),
                static_cast<vertex_num_t>(g_config.ul_select_nbrs_qs),
                static_cast<vertex_num_t>(g_config.bl_select_nbrs_qs),
                g_config.ul_max_nbr_size,
                g_config.bl_max_nbr_size);

            // Refining layer_configs (ul + bl) are derived inside
            // artea_graph::IndexStructure as rgraph.{ul,bl}_max_nbr_size × 1.5.

            // prefill_ratio drives L0 random top-up in refine_layer.
            artea_graph::propagate_config_t<Metric, Dim> propagate_config(
                static_cast<iter_t>(g_config.num_build_loops),
                static_cast<iter_t>(g_config.num_triu_iters),
                static_cast<ratio_t>(g_config.prefill_ratio),
                static_cast<iter_t>(g_config.num_routing_loops),
                static_cast<vertex_num_t>(g_config.routing_topk),
                static_cast<vertex_num_t>(g_config.routing_queue_size));

            // Single PruningConfig instance drives both the stacked_rgraph
            // upper-layer insertion (reads scale_coeffs only, shift ignored)
            // and the per-layer refinement PruningUpdater (reads both).
            artea_graph::pruning_config_t<Metric, Dim> pruning_config(
                static_cast<ratio_t>(g_config.scale_coeffs),
                static_cast<ratio_t>(g_config.shifted_coeffs),
                static_cast<ratio_t>(g_config.l0_min_distance));

            auto graph = std::make_unique<artea_index_t>(
                total_vertices, rgraph_config,
                propagate_config, pruning_config);

            ARTEA_INFO("Building artea_graph (3 steps: r-net insert → "
                       "per-layer refine → writeback)...");
            auto t0 = std::chrono::high_resolution_clock::now();

            vector_array_t owned_batch =
                base_vecs.extract_subset(0, total_vertices);
            artea_graph::factory_t<Metric, Dim>::add_vertices(
                *graph, std::move(owned_batch), dist_func,
                g_config.insert_on_L0,
                g_config.shuffle_insertion_order);

            auto t1 = std::chrono::high_resolution_clock::now();
            _build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                t1 - t0).count();

            const layer_id_t top_level_id = graph->top_occupied_level_id();
            ARTEA_INFO(fmt::format(
                "artea_graph built in {} ms: top_occupied_level={}, "
                "max_restrict_level={}",
                _build_ms,
                (top_level_id == dynamic::hierarchical_graph_t
                    ::unassigned_highest_level_id)
                    ? -1 : static_cast<int>(top_level_id),
                graph->max_restrict_level()));

            // Preview what HierarchicalGraphCompactor would trim, using the
            // same rule it applies in compact_graph(): walk from the top
            // down and take the first bucket with >= min_layer_cap vids as
            // the new top. Anything above that gets demoted; the entry
            // point is the bucket-centroid argmin on (new_top bucket +
            // demoted vids).
            constexpr vertex_num_t min_cap =
                hierarchical_graph_compactor_t::min_layer_cap;
            const bool has_vertices = (top_level_id !=
                dynamic::hierarchical_graph_t::unassigned_highest_level_id);
            layer_id_t compactor_new_top = 0;
            if (has_vertices) {
                for (layer_id_t h = top_level_id; ; --h) {
                    if (graph->get_vids_with_highest_level(h).size() >= min_cap) {
                        compactor_new_top = h;
                        break;
                    }
                    if (h == 0) { compactor_new_top = 0; break; }
                }
                ARTEA_INFO(fmt::format(
                    "Compactor trim preview: min_layer_cap={}, new_top=L{} "
                    "(entry-point = argmin dist-to-centroid over this bucket + demoted vids)",
                    min_cap, compactor_new_top));
            }

            // Per-layer vertex counts are emitted by
            // artea_graph::IndexFactory::add_vertices itself, so no
            // duplicate breakdown here.

            _graph = std::move(graph);
        });
    }

    static void TearDownTestSuite() { _graph.reset(); }

    static std::unique_ptr<artea_index_t> _graph;
    static int64_t                       _build_ms;
};

std::unique_ptr<artea_index_t> ArteaGraphTest::_graph = nullptr;
int64_t                       ArteaGraphTest::_build_ms = 0;

// ============================================================
//  Build sanity + search recall/throughput.
// ============================================================

TEST_F(ArteaGraphTest, BuildSanity) {
    ASSERT_NE(_graph, nullptr);
    const auto& graph = _graph->get_hierarchical_graph();
    const auto& base_vecs =
        DataProvider::instance().get_dataset().get_base_vecs();
    EXPECT_EQ(graph.get_num_vertices(),
              static_cast<vertex_num_t>(base_vecs.get_num_vecs()));
    using HG = dynamic::hierarchical_graph_t;
    EXPECT_NE(graph.top_occupied_level_id(),
              HG::unassigned_highest_level_id);
}

TEST_F(ArteaGraphTest, SearchRecallAndThroughput) {
    ASSERT_NE(_graph, nullptr);
    auto& provider = DataProvider::instance();
    const auto& dataset    = provider.get_dataset();
    const auto& base_vecs  = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt         = dataset.get_gt_vecs();

    // ---- Grid search over the workload's throughput_search sweeps ----
    const uint32_t num_queries = static_cast<uint32_t>(query_vecs.get_num_vecs());

    struct Row {
        uint32_t topk;
        uint32_t queue_size;
        // Static artea: compact hierarchical graph, greedy-upper + beam-L0.
        double   s_batch_ms;
        double   s_qps;
        float    s_recall;
    };
    std::vector<Row> rows;
    int64_t compact_ms = 0;

    infra_dispatch(provider.get_dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        // Stateless functor: the dimension is a compile-time trait now.
        dist_func_t<Metric, Dim> dist_func;

        // ---- Compact the dynamic hierarchical graph (once, shared by
        //      every sweep) ----
        const auto& dyn_hg = _graph->get_hierarchical_graph();
        auto tc0 = std::chrono::high_resolution_clock::now();
        auto compact_hg = hierarchical_graph_compactor_t::compact_graph(
            dyn_hg, base_vecs, dist_func);
        auto tc1 = std::chrono::high_resolution_clock::now();
        compact_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(tc1 - tc0).count();
        ARTEA_INFO(fmt::format(
            "Hierarchical graph compacted in {} ms", compact_ms));

        recall_estimator_t<Metric, Dim> re;

        // Runs @p batch_call (a batch query closure) over warmup + test_runs
        // iterations, averages latency/recall, and returns them.
        auto time_batch = [&](auto&& batch_call, uint32_t topk)
            -> std::tuple<double, float, knn_results_t<Metric, Dim>>
        {
            for (uint32_t w = 0; w < g_config.warmup_runs; ++w) {
                [[maybe_unused]] auto _ = batch_call();
            }
            double total_us = 0.0;
            float  total_recall = 0.0f;
            knn_results_t<Metric, Dim> last_results;
            for (uint32_t r = 0; r < g_config.test_runs; ++r) {
                auto t0 = std::chrono::high_resolution_clock::now();
                last_results = batch_call();
                auto t1 = std::chrono::high_resolution_clock::now();
                total_us += std::chrono::duration_cast<std::chrono::microseconds>(
                    t1 - t0).count();
                total_recall += re.calculate_recall_at_k(
                    last_results, gt, topk, num_queries);
            }
            return {
                total_us    / g_config.test_runs,
                total_recall / g_config.test_runs,
                std::move(last_results)
            };
        };

        for (const auto& sweep : g_config.sweeps) {
            const uint32_t topk = std::min<uint32_t>(
                sweep.topk, gt.get_vec_dim());

            ARTEA_INFO(fmt::format(
                "Grid search: candidate queue size {} to {}, step {} "
                "(topk={}, warmup={}, test_runs={})",
                sweep.queue_start, sweep.queue_end, sweep.queue_step,
                topk, g_config.warmup_runs, g_config.test_runs));

            for (uint32_t queue_size = sweep.queue_start;
                 queue_size <= sweep.queue_end;
                 queue_size += sweep.queue_step)
            {
                const uint32_t effective_queue_size =
                    std::max<uint32_t>(queue_size, topk);

                hierarchical_graph_router_t<Metric, Dim> s_router(
                    base_vecs, dist_func,
                    /*topk=*/topk,
                    /*candidate_queue_size=*/effective_queue_size);
                s_router.initialize();

                // --- Hierarchical router: compact hier_graph, greedy-upper + beam-L0
                //     (RandomSeeding=false uses compact_hg.entry_point_vid()) ---
                auto [s_avg_us, s_recall, s_last] = time_batch([&]() {
                    return s_router.template batch_query</*RandomSeeding=*/false, /*UpperLevelBeamSearch=*/false>(
                        query_vecs, compact_hg);
                }, topk);
                ASSERT_EQ(s_last.size(), static_cast<std::size_t>(num_queries) * topk);

                Row row;
                row.topk        = topk;
                row.queue_size  = effective_queue_size;
                row.s_batch_ms  = s_avg_us  / 1000.0;
                row.s_qps       = num_queries * 1e6 / s_avg_us;
                row.s_recall    = s_recall;
                rows.push_back(row);

                ARTEA_INFO(fmt::format(
                    "CandidateQueue={:4}: "
                    "hierarchical [R@{}={:.4f}, QPS={:8.1f}, batch={:.2f} ms]",
                    effective_queue_size,
                    topk, row.s_recall,  row.s_qps,  row.s_batch_ms));
            }
        }
    });

    ARTEA_INFO("=== hierarchical router summary ===");
    ARTEA_INFO(fmt::format("  build_time    : {} ms", _build_ms));
    ARTEA_INFO(fmt::format("  compact_time  : {} ms", compact_ms));
    ARTEA_INFO(fmt::format("  num_queries   : {}", num_queries));
    ARTEA_INFO(fmt::format(
        "{:<6} {:<8} | {:<10} {:<10} {:<10}",
        "Topk", "Queue", "H.Recall@k", "H.QPS", "H.batch(ms)"));
    ARTEA_INFO(std::string(6 + 1 + 8 + 3 + 10 + 10 + 10, '-'));
    for (const auto& row : rows) {
        ARTEA_INFO(fmt::format(
            "{:<6} {:<8} | {:<10.4f} {:<10.1f} {:<10.2f}",
            row.topk, row.queue_size, row.s_recall, row.s_qps, row.s_batch_ms));
    }

    bool has_any_hier = false;
    for (const auto& row : rows) {
        if (row.s_recall > 0.0f) has_any_hier = true;
    }
    EXPECT_TRUE(has_any_hier)
        << "At least one queue-size should return hierarchical router results";
}

// ============================================================
//  main: workload JSON + test suite runner
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_artea_graph");
    program.add_description(
        "artea_graph end-to-end test, driven by a bench-artea-style workload");
    program.add_argument("-w", "--workload")
        .required()
        .help("Path to workload JSONC file (same schema as bench-artea: "
              "dataset-config / dataset / metric / warmup_runs / test_runs "
              "at the top level, build params from the indexes-config.artea "
              "base entry, search grid from its throughput_search sweeps)");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program;
        return 1;
    }

    const std::string workload_path = program.get<std::string>("--workload");
    std::ifstream wf(workload_path);
    if (!wf.is_open()) {
        std::cerr << "Failed to open workload: " << workload_path << "\n";
        return 1;
    }
    nlohmann::json wl = nlohmann::json::parse(wf);
    wf.close();
    std::cout << "Loaded workload: " << workload_path << "\n";

    g_config.config_path  = wl["dataset-config"];
    g_config.dataset_name = wl["dataset"];
    g_config.metric       = wl.value("metric", "euclidean");
    g_config.warmup_runs  = wl.value("warmup_runs", 1u);
    g_config.test_runs    = wl.value("test_runs", 3u);

    // Resolve the artea base config: object form is the config itself;
    // array form (tuning workloads) contributes its unique untagged base
    // entry — same resolution as bench-artea's parse_index_base_config.
    const auto& artea_entry = wl["indexes-config"]["artea"];
    nlohmann::json params;
    if (artea_entry.is_object()) {
        params = artea_entry;
    } else if (artea_entry.is_array()) {
        for (const auto& e : artea_entry) {
            if (e.is_object() && !e.contains("tag")) { params = e; break; }
        }
    }
    if (!params.is_object() || params.empty()) {
        std::cerr << "workload indexes-config.artea has no base config\n";
        return 1;
    }

    // Build params: same keys and defaults as bench-artea's run_all().
    g_config.rnet_beta          = params.value("rnet_beta", 2.0f);
    g_config.l0_radius_provided = params.contains("l0_rnet_radius");
    g_config.l0_rnet_radius     = params.value("l0_rnet_radius", -1.0f);
    g_config.ul_max_nbr_size    = params.value("ul_max_nbr_size", 32u);
    g_config.bl_max_nbr_size    = params.value("bl_max_nbr_size", 64u);
    g_config.search_nn_qs       = params.value("search_nn_qs", 30u);
    g_config.ul_select_nbrs_qs  = params.value("ul_select_nbrs_qs", 100u);
    g_config.bl_select_nbrs_qs  = params.value("bl_select_nbrs_qs", 100u);
    g_config.scale_coeffs       = params.value("scale_coeffs", 1.1f);
    g_config.shifted_coeffs     = params.value("shifted_coeffs", 0.0f);
    g_config.l0_min_distance    = params.value("l0_min_distance", 1.0f);
    g_config.num_build_loops    = params.value("num_build_loops", 5u);
    g_config.num_triu_iters     = params.value("num_triu_iters", 12u);
    g_config.prefill_ratio      = params.value("prefill_ratio", 0.34f);
    g_config.num_routing_loops  = params.value("num_routing_loops", 0u);
    g_config.routing_topk       = params.value("routing_topk", 64u);
    g_config.routing_queue_size = params.value("routing_queue_size", 96u);
    g_config.shuffle_insertion_order =
        params.value("shuffle_insertion_order", false);
    g_config.insert_on_L0       = params.value("insert_on_L0", false);

    // L0-radius auto-probe kicks in when the workload omits l0_rnet_radius.
    g_config.probe_num_samples  = params.value("probe_num_samples", 500u);
    g_config.probe_quantile     = params.value("probe_quantile", 0.9f);

    // Search sweeps: the artea entry's own throughput_search, falling back
    // to the workload-level array (same resolution as bench-artea).
    const nlohmann::json* sweeps_json = nullptr;
    if (params.contains("throughput_search")) {
        sweeps_json = &params.at("throughput_search");
    } else if (wl.contains("throughput_search")) {
        sweeps_json = &wl.at("throughput_search");
    }
    if (sweeps_json == nullptr || !sweeps_json->is_array()
        || sweeps_json->empty()) {
        std::cerr << "workload has no throughput_search sweep for artea\n";
        return 1;
    }
    for (const auto& entry : *sweeps_json) {
        const std::string qc = entry.at("candidate_queue_config");
        std::istringstream ss(qc);
        std::string token;
        std::vector<uint32_t> values;
        while (std::getline(ss, token, ',')) {
            values.push_back(static_cast<uint32_t>(std::stoul(token)));
        }
        if (values.size() != 3) {
            std::cerr << "candidate_queue_config must be start,end,step "
                         "(got '" << qc << "')\n";
            return 1;
        }
        g_config.sweeps.push_back(SearchSweep{
            entry.at("topk").get<uint32_t>(),
            values[0], values[1], values[2]});
    }

    dump_config("artea_graph test — configuration (start)");

    DataProvider::instance().init();
    const int rc = RUN_ALL_TESTS();

    dump_config("artea_graph test — configuration (end)");
    return rc;
}
