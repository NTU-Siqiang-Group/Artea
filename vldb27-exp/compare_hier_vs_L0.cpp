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
 * @FilePath: /Artea/vldb27-exp/compare_hier_vs_L0.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Head-to-head benchmark comparing the hierarchical-router
 *               path (greedy-upper + beam-L0 over the compact hierarchy
 *               from @c entry_point_vid) against the L0-only single-layer
 *               baseline (flat beam on compact L0 from a random entry
 *               vertex). Mirrors the build pipeline of test_artea_graph,
 *               but focuses on the hier-vs-L0 comparison and lives under
 *               vldb27-exp so it can be driven by its own experiment
 *               scripts independently of the unit-test target.
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

using namespace artea;
using namespace artea::cpu;

// ============================================================
//  Global configuration (populated by argparse in main())
// ============================================================

struct TestConfig {
    std::string config_path;
    std::string dataset_name;

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
    bool     perform_arc;
    float    aspect_ratio_constraint;

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

    // Search-time params
    uint32_t query_topk;
    uint32_t queue_size_start;
    uint32_t queue_size_end;
    uint32_t queue_size_step;
    uint32_t warmup_runs;
    uint32_t test_runs;

    // Result output
    std::string output_json;
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
       << "perform_arc:                "
       << (g_config.perform_arc ? "true" : "false") << "\n"
       << "aspect_ratio_constraint:    " << g_config.aspect_ratio_constraint
       << " (final refine_layer sweep drops edges > arc * radius_at(h); "
       << "skipped when perform_arc=false)\n"
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
       << "--- Search ---\n"
       << "query_topk:                 " << g_config.query_topk << "\n"
       << "candidate-queue-config:     "
       << g_config.queue_size_start << ","
       << g_config.queue_size_end   << ","
       << g_config.queue_size_step  << "\n"
       << "warmup_runs:                " << g_config.warmup_runs << "\n"
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

        _dist_func = std::make_unique<dist_func_t>(base_vecs.get_vec_dim());

        if (g_config.l0_radius_provided) {
            _l0_radius = g_config.l0_rnet_radius;
            ARTEA_INFO(fmt::format(
                "Using user-provided L0 rnet_radius = {:.6f}", _l0_radius));
        } else {
            dataset_prober_t prober(base_vecs, *_dist_func);
            const std::vector<float> quantiles = { g_config.probe_quantile };
            ARTEA_INFO(fmt::format(
                "Probing L0 rnet_radius ({}th pct, {} samples)...",
                static_cast<int>(g_config.probe_quantile * 100.0f),
                g_config.probe_num_samples));
            auto result = prober.probe(quantiles, g_config.probe_num_samples);
            _l0_radius = static_cast<float>(result.table[0][0]);
            ARTEA_INFO(fmt::format(
                "Auto-probed L0 rnet_radius = {:.6f}", _l0_radius));
        }
    }

    auto get_dataset()   -> vector_dataset_t& { return *_dataset; }
    auto get_dist_func() -> dist_func_t&      { return *_dist_func; }
    auto get_l0_radius() const -> float       { return _l0_radius; }

private:
    DataProvider() = default;

    std::unique_ptr<vector_dataset_t> _dataset;
    std::unique_ptr<dist_func_t>      _dist_func;
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
        auto& dist_func = provider.get_dist_func();

        const vertex_num_t total_vertices =
            static_cast<vertex_num_t>(base_vecs.get_num_vecs());

        artea_graph::rgraph_config_t rgraph_config(
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
        artea_graph::propagate_config_t propagate_config(
            static_cast<iter_t>(g_config.num_build_loops),
            static_cast<iter_t>(g_config.num_triu_iters),
            static_cast<ratio_t>(g_config.prefill_ratio),
            static_cast<iter_t>(g_config.num_routing_loops),
            static_cast<vertex_num_t>(g_config.routing_topk),
            static_cast<vertex_num_t>(g_config.routing_queue_size));

        // Single PruningConfig instance drives both the stacked_rgraph
        // upper-layer insertion (reads scale_coeffs only, shift ignored)
        // and the per-layer refinement PruningUpdater (reads both).
        artea_graph::pruning_config_t pruning_config(
            static_cast<ratio_t>(g_config.scale_coeffs),
            static_cast<ratio_t>(g_config.shifted_coeffs),
            g_config.perform_arc,
            static_cast<ratio_t>(g_config.aspect_ratio_constraint));

        _graph = std::make_unique<artea_graph::index_t>(
            total_vertices, rgraph_config,
            propagate_config, pruning_config);

        ARTEA_INFO("Building artea_graph (3 steps: r-net insert → "
                   "per-layer refine → writeback)...");
        auto t0 = std::chrono::high_resolution_clock::now();

        vector_array_t owned_batch =
            base_vecs.extract_subset(0, total_vertices);
        artea_graph::factory_t::add_vertices(
            *_graph, std::move(owned_batch), dist_func,
            g_config.insert_on_L0,
            g_config.shuffle_insertion_order);

        auto t1 = std::chrono::high_resolution_clock::now();
        _build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t1 - t0).count();

        const layer_id_t top_level_id = _graph->top_occupied_level_id();
        ARTEA_INFO(fmt::format(
            "artea_graph built in {} ms: top_occupied_level={}, "
            "max_restrict_level={}",
            _build_ms,
            (top_level_id == dynamic::hierarchical_graph_t
                ::unassigned_highest_level_id)
                ? -1 : static_cast<int>(top_level_id),
            _graph->max_restrict_level()));

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
                if (_graph->get_vids_with_highest_level(h).size() >= min_cap) {
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

        // A vertex with highest_level_id=h' participates in every level
        // 0..h', so the count *assigned to* level h is the cumulative
        // sum of bucket sizes from h to the top. Compute top-down.
        const layer_id_t max_level = _graph->max_restrict_level();
        std::vector<vertex_num_t> num_at_level(max_level + 1, 0);
        {
            vertex_num_t running = 0;
            for (layer_id_t h = max_level; ; --h) {
                running += _graph->get_vids_with_highest_level(h).size();
                num_at_level[h] = running;
                if (h == 0) break;
            }
        }

        for (layer_id_t h = 0; h <= max_level; ++h) {
            const vertex_num_t count = num_at_level[h];
            const float ratio = 100.0f * count / base_vecs.get_num_vecs();
            const char* tag;
            if (!has_vertices)               tag = "-";
            else if (h < compactor_new_top)  tag = "kept";
            else if (h == compactor_new_top) tag = "kept, compactor new-top";
            else                             tag = "trimmed -> demoted";
            ARTEA_INFO(fmt::format(
                "  level_id={}: {} vertices ({:.2f}% of base) [{}]",
                h, count, ratio, tag));
        }
    }

    static void TearDownTestSuite() { _graph.reset(); }

    static std::unique_ptr<artea_graph::index_t> _graph;
    static int64_t                               _build_ms;
};

std::unique_ptr<artea_graph::index_t> ArteaGraphTest::_graph = nullptr;
int64_t                               ArteaGraphTest::_build_ms = 0;

// ============================================================
//  Build sanity + search recall/throughput.
// ============================================================

TEST_F(ArteaGraphTest, BuildSanity) {
    ASSERT_NE(_graph, nullptr);
    const auto& base_vecs =
        DataProvider::instance().get_dataset().get_base_vecs();
    EXPECT_EQ(_graph->get_num_vertices(),
              static_cast<vertex_num_t>(base_vecs.get_num_vecs()));
    using HG = dynamic::hierarchical_graph_t;
    EXPECT_NE(_graph->top_occupied_level_id(),
              HG::unassigned_highest_level_id);
}

TEST_F(ArteaGraphTest, SearchRecallAndThroughput) {
    auto& provider = DataProvider::instance();
    const auto& dataset    = provider.get_dataset();
    const auto& base_vecs  = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt         = dataset.get_gt_vecs();
    auto&       dist_func  = provider.get_dist_func();

    // ---- Compact the dynamic hierarchical graph ----
    const auto& dyn_hg = _graph->get_hierarchical_graph();
    auto tc0 = std::chrono::high_resolution_clock::now();
    auto compact_hg = hierarchical_graph_compactor_t::compact_graph(
        dyn_hg, base_vecs, dist_func);
    auto tc1 = std::chrono::high_resolution_clock::now();
    const int64_t compact_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(tc1 - tc0).count();
    ARTEA_INFO(fmt::format(
        "Hierarchical graph compacted in {} ms", compact_ms));

    // ---- Grid search over candidate queue size ----
    const uint32_t topk = std::min<uint32_t>(
        g_config.query_topk, gt.get_vec_dim());
    const uint32_t num_queries = static_cast<uint32_t>(query_vecs.get_num_vecs());

    ARTEA_INFO(fmt::format(
        "Grid search: candidate queue size {} to {}, step {} "
        "(topk={}, warmup={}, test_runs={})",
        g_config.queue_size_start, g_config.queue_size_end,
        g_config.queue_size_step, topk,
        g_config.warmup_runs, g_config.test_runs));

    recall_estimator_t re;
    adr_estimator_t    ae;

    struct Row {
        uint32_t queue_size;
        // Static artea: compact hierarchical graph, greedy-upper + beam-L0.
        double   s_batch_ms;
        double   s_qps;
        float    s_recall;
        float    s_adr;
        // Static artea L0-only baseline: flat beam on compact L0 from a random entry vertex.
        double   l0_batch_ms;
        double   l0_qps;
        float    l0_recall;
        float    l0_adr;
    };
    std::vector<Row> rows;

    // Runs @p batch_call (a batch query closure) over warmup + test_runs
    // iterations, averages latency/recall/ADR, and returns them.
    auto time_batch = [&](auto&& batch_call)
        -> std::tuple<double, float, float, knn_results_t>
    {
        for (uint32_t w = 0; w < g_config.warmup_runs; ++w) {
            [[maybe_unused]] auto _ = batch_call();
        }
        double total_us     = 0.0;
        float  total_recall = 0.0f;
        float  total_adr    = 0.0f;
        knn_results_t last_results;
        for (uint32_t r = 0; r < g_config.test_runs; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            last_results = batch_call();
            auto t1 = std::chrono::high_resolution_clock::now();
            total_us += std::chrono::duration_cast<std::chrono::microseconds>(
                t1 - t0).count();
            total_recall += re.calculate_recall_at_k(
                last_results, gt, topk, num_queries);
            total_adr += static_cast<float>(ae.calculate_adr_at_1(
                last_results, topk, gt, query_vecs, base_vecs, dist_func, num_queries));
        }
        return {
            total_us     / g_config.test_runs,
            total_recall / g_config.test_runs,
            total_adr    / g_config.test_runs,
            std::move(last_results)
        };
    };

    for (uint32_t queue_size = g_config.queue_size_start;
         queue_size <= g_config.queue_size_end;
         queue_size += g_config.queue_size_step)
    {
        const uint32_t effective_queue_size =
            std::max<uint32_t>(queue_size, topk);

        hierarchical_graph_router_t s_router(
            base_vecs, dist_func,
            /*topk=*/topk,
            /*candidate_queue_size=*/effective_queue_size);
        s_router.initialize();

        // --- Hierarchical router: compact hier_graph, greedy-upper + beam-L0
        //     (RandomSeeding=false uses compact_hg.entry_point_vid()) ---
        auto [s_avg_us, s_recall, s_adr, s_last] = time_batch([&]() {
            return s_router.template batch_query</*RandomSeeding=*/false, /*UpperLevelBeamSearch=*/false>(
                query_vecs, compact_hg);
        });
        ASSERT_EQ(s_last.size(), static_cast<std::size_t>(num_queries) * topk);

        // --- L0-only single-layer router (flat beam on compact L0 from a random entry vertex) ---
        auto [l0_avg_us, l0_recall, l0_adr, l0_last] = time_batch(
            [&]() {
                return s_router.template batch_query_l0_only</*RandomSeeding=*/true>(
                    query_vecs, compact_hg);
            });
        ASSERT_EQ(l0_last.size(),
                  static_cast<std::size_t>(num_queries) * topk);

        Row row;
        row.queue_size  = effective_queue_size;
        row.s_batch_ms  = s_avg_us  / 1000.0;
        row.s_qps       = num_queries * 1e6 / s_avg_us;
        row.s_recall    = s_recall;
        row.s_adr       = s_adr;
        row.l0_batch_ms = l0_avg_us / 1000.0;
        row.l0_qps      = num_queries * 1e6 / l0_avg_us;
        row.l0_recall   = l0_recall;
        row.l0_adr      = l0_adr;
        rows.push_back(row);

        ARTEA_INFO(fmt::format(
            "CandidateQueue={:4}: "
            "hierarchical [R@{}={:.4f}, ADR={:.6f}, QPS={:8.1f}, batch={:.2f} ms] | "
            "L0-only      [R@{}={:.4f}, ADR={:.6f}, QPS={:8.1f}, batch={:.2f} ms]",
            effective_queue_size,
            topk, row.s_recall,  row.s_adr,  row.s_qps,  row.s_batch_ms,
            topk, row.l0_recall, row.l0_adr, row.l0_qps, row.l0_batch_ms));
    }

    ARTEA_INFO("=== hierarchical router vs. L0-only single-layer router summary ===");
    ARTEA_INFO(fmt::format("  build_time    : {} ms", _build_ms));
    ARTEA_INFO(fmt::format("  compact_time  : {} ms", compact_ms));
    ARTEA_INFO(fmt::format("  num_queries   : {}", num_queries));
    ARTEA_INFO(fmt::format("  topk          : {}", topk));
    ARTEA_INFO(fmt::format(
        "{:<8} | {:<10} {:<10} {:<10} {:<10} | {:<10} {:<10} {:<10} {:<10}",
        "Queue",
        "H.Recall@k",  "H.ADR",  "H.QPS",  "H.batch(ms)",
        "L0.Recall@k", "L0.ADR", "L0.QPS", "L0.batch(ms)"));
    ARTEA_INFO(std::string(
        8 + 3 + 10 + 10 + 10 + 10 + 3 + 10 + 10 + 10 + 10, '-'));
    for (const auto& row : rows) {
        ARTEA_INFO(fmt::format(
            "{:<8} | {:<10.4f} {:<10.6f} {:<10.1f} {:<10.2f} | "
            "{:<10.4f} {:<10.6f} {:<10.1f} {:<10.2f}",
            row.queue_size,
            row.s_recall,  row.s_adr,  row.s_qps,  row.s_batch_ms,
            row.l0_recall, row.l0_adr, row.l0_qps, row.l0_batch_ms));
    }

    // ---- Dump one row per queue-size so the companion plotter can draw
    //      Recall-vs-QPS for both routers. Writes unconditionally; done
    //      before the EXPECT_TRUE assertions so the JSON survives an
    //      assertion failure.
    if (!g_config.output_json.empty()) {
        nlohmann::json out;
        out["dataset"]     = g_config.dataset_name;
        out["topk"]        = topk;
        out["num_queries"] = num_queries;
        out["build_ms"]    = _build_ms;
        out["compact_ms"]  = compact_ms;
        out["rows"] = nlohmann::json::array();
        for (const auto& row : rows) {
            out["rows"].push_back({
                {"queue_size", row.queue_size},
                {"hier", {
                    {"recall",   row.s_recall},
                    {"adr",      row.s_adr},
                    {"qps",      row.s_qps},
                    {"batch_ms", row.s_batch_ms},
                }},
                {"l0", {
                    {"recall",   row.l0_recall},
                    {"adr",      row.l0_adr},
                    {"qps",      row.l0_qps},
                    {"batch_ms", row.l0_batch_ms},
                }},
            });
        }
        const std::filesystem::path out_path(g_config.output_json);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }
        std::ofstream ofs(out_path);
        if (ofs.is_open()) {
            ofs << out.dump(2);
            ARTEA_INFO(fmt::format("Results written to {}", g_config.output_json));
        } else {
            ARTEA_INFO(fmt::format("WARNING: failed to open {} for write", g_config.output_json));
        }
    }

    bool has_any_hier = false;
    bool has_any_l0   = false;
    for (const auto& row : rows) {
        if (row.s_recall  > 0.0f) has_any_hier = true;
        if (row.l0_recall > 0.0f) has_any_l0   = true;
    }
    EXPECT_TRUE(has_any_hier)
        << "At least one queue-size should return hierarchical router results";
    EXPECT_TRUE(has_any_l0)
        << "At least one queue-size should return L0-only results";
}

// ============================================================
//  main: argparse + test suite runner
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("compare_hier_vs_L0");
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to datasets.json config file");
    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name (as listed in datasets.json)");

    // Stacked r-net backbone
    program.add_argument("--beta")
        .default_value(2.0f).scan<'g', float>();
    program.add_argument("--l0-radius")
        .default_value(-1.0f).scan<'g', float>()
        .help("L0 rnet_radius (covering radius at the bottom layer). "
              "L1 and higher radii are derived as L0 * beta^h. "
              "If negative, auto-probe via DatasetProber.");
    program.add_argument("--ul-max-nbr-size")
        .default_value(32u).scan<'u', uint32_t>()
        .help("Per-vertex neighbor capacity at every upper layer (level_id > 0).");
    program.add_argument("--bl-max-nbr-size")
        .default_value(64u).scan<'u', uint32_t>()
        .help("Per-vertex neighbor capacity at the bottom layer (L0). "
              "Independent of --ul-max-nbr-size.");
    program.add_argument("--search-nn-qs")
        .default_value(40u).scan<'u', uint32_t>();
    program.add_argument("--ul-select-nbrs-qs")
        .default_value(100u).scan<'u', uint32_t>()
        .help("Upper-layer (L1+) beam-search queue size for the select phase.");
    program.add_argument("--bl-select-nbrs-qs")
        .default_value(100u).scan<'u', uint32_t>()
        .help("Bottom-layer (L0) beam-search queue size for the select phase.");
    program.add_argument("--scale-coeffs")
        .default_value(1.1f).scan<'g', float>();
    program.add_argument("--shifted-coeffs")
        .default_value(0.0f).scan<'g', float>();
    program.add_argument("--perform-arc")
        .default_value(false).implicit_value(true)
        .help("If present, refine_layer runs a final aspect-ratio-constrained "
              "pruning sweep that drops edges longer than "
              "aspect_ratio_constraint * radius_at(level_id).");
    program.add_argument("--aspect-ratio-constraint")
        .default_value(1.0f).scan<'g', float>()
        .help("Multiplier on the per-layer r-net radius for the ARC sweep "
              "(only consumed when --perform-arc is set). Default 1.0.");

    program.add_argument("--probe-num-samples")
        .default_value(500u).scan<'u', uint32_t>();
    program.add_argument("--probe-quantile")
        .default_value(0.9f).scan<'g', float>();

    // Per-layer refinement
    program.add_argument("--num-build-loops")
        .default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters")
        .default_value(14u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio")
        .default_value(0.34f).scan<'g', float>()
        .help("L0 random-prefill density: each L0 vertex is topped up to "
              "prefill_ratio * (bl_max_nbr_size * 1.5) random neighbors "
              "before the propagate loop.");
    program.add_argument("--num-routing-loops")
        .default_value(1u).scan<'u', uint32_t>();
    program.add_argument("--routing-topk")
        .default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--routing-queue-size")
        .default_value(96u).scan<'u', uint32_t>();

    program.add_argument("--shuffle")
        .default_value(false).implicit_value(true)
        .help("If present, shuffle the order in which new vids are "
              "inserted into the hierarchy (storage layout unchanged). "
              "Absent by default.");

    program.add_argument("--insert-on-l0")
        .default_value(false).implicit_value(true)
        .help("If present, the parent stacked_rgraph factory builds L0 "
              "edges at insertion time. Absent (default) skips L0 "
              "insertion and lets refine_layer seed L0 via random "
              "prefill over the newly-inserted vid window.");

    // Search
    program.add_argument("--query-topk")
        .default_value(10u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-config")
        .default_value(std::string("40,200,20"))
        .help("Candidate queue size grid search: start,end,step "
              "(default: 40,200,20)");
    program.add_argument("--warmup-runs")
        .default_value(1u).scan<'u', uint32_t>();
    program.add_argument("--test-runs")
        .default_value(3u).scan<'u', uint32_t>();

    program.add_argument("-o", "--output")
        .default_value(std::string(""))
        .help("Path to write the per-queue-size Recall/QPS JSON consumed by "
              "plot_compare_hier_vs_L0.py. Empty = skip JSON dump.");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program;
        return 1;
    }

    g_config.config_path               = program.get<std::string>("--config");
    g_config.dataset_name              = program.get<std::string>("--dataset");
    g_config.rnet_beta                 = program.get<float>("--beta");
    g_config.ul_max_nbr_size           = program.get<uint32_t>("--ul-max-nbr-size");
    g_config.bl_max_nbr_size           = program.get<uint32_t>("--bl-max-nbr-size");
    g_config.search_nn_qs              = program.get<uint32_t>("--search-nn-qs");
    g_config.ul_select_nbrs_qs         = program.get<uint32_t>("--ul-select-nbrs-qs");
    g_config.bl_select_nbrs_qs         = program.get<uint32_t>("--bl-select-nbrs-qs");
    g_config.scale_coeffs              = program.get<float>("--scale-coeffs");
    g_config.shifted_coeffs            = program.get<float>("--shifted-coeffs");
    g_config.perform_arc               = program.get<bool>("--perform-arc");
    g_config.aspect_ratio_constraint   = program.get<float>("--aspect-ratio-constraint");
    g_config.probe_num_samples         = program.get<uint32_t>("--probe-num-samples");
    g_config.probe_quantile            = program.get<float>("--probe-quantile");
    g_config.num_build_loops           = program.get<uint32_t>("--num-build-loops");
    g_config.num_triu_iters            = program.get<uint32_t>("--num-triu-iters");
    g_config.prefill_ratio             = program.get<float>("--prefill-ratio");
    g_config.num_routing_loops         = program.get<uint32_t>("--num-routing-loops");
    g_config.routing_topk              = program.get<uint32_t>("--routing-topk");
    g_config.routing_queue_size        = program.get<uint32_t>("--routing-queue-size");
    g_config.shuffle_insertion_order   = program.get<bool>("--shuffle");
    g_config.insert_on_L0              = program.get<bool>("--insert-on-l0");
    g_config.query_topk                = program.get<uint32_t>("--query-topk");
    g_config.warmup_runs               = program.get<uint32_t>("--warmup-runs");
    g_config.test_runs                 = program.get<uint32_t>("--test-runs");
    g_config.output_json               = program.get<std::string>("--output");

    // Parse candidate-queue-config: "start,end,step"
    {
        const std::string s = program.get<std::string>("--candidate-queue-config");
        std::istringstream ss(s);
        std::string token;
        std::vector<uint32_t> values;
        while (std::getline(ss, token, ',')) {
            values.push_back(static_cast<uint32_t>(std::stoul(token)));
        }
        if (values.size() != 3) {
            std::cerr << "Error: --candidate-queue-config must have 3 "
                         "values (start,end,step)\n";
            return 1;
        }
        g_config.queue_size_start = values[0];
        g_config.queue_size_end   = values[1];
        g_config.queue_size_step  = values[2];
    }

    const float l0_radius_arg = program.get<float>("--l0-radius");
    g_config.l0_radius_provided = (l0_radius_arg >= 0.0f);
    g_config.l0_rnet_radius     = l0_radius_arg;

    dump_config("artea_graph test — configuration (start)");

    DataProvider::instance().init();
    const int rc = RUN_ALL_TESTS();

    dump_config("artea_graph test — configuration (end)");
    return rc;
}
