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
 *               hierarchical graph router. Configurations are dumped at
 *               start and end of the run.
 */

#include <chrono>
#include <cstdint>
#include <filesystem>
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
    bool     l1_radius_provided;
    float    l1_rnet_radius;
    uint32_t max_nbr_size;
    uint32_t search_nn_qs;
    uint32_t select_nbrs_qs;
    float    scale_coeffs;
    float    shifted_coeffs;

    // L1-radius auto-probe
    uint32_t probe_num_samples;
    float    probe_quantile;

    // Per-layer conv_graph-style refinement
    uint32_t refining_max_nbr_size;
    uint32_t num_build_loops;
    uint32_t num_triu_iters;
    float    prefill_ratio;
    uint32_t num_routing_loops;
    uint32_t routing_topk;
    uint32_t routing_queue_size;

    // Search-time params
    uint32_t query_topk;
    uint32_t queue_size_start;
    uint32_t queue_size_end;
    uint32_t queue_size_step;
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
    if (g_config.l1_radius_provided) {
        os << "L1 radius:                  " << g_config.l1_rnet_radius
           << " (user-provided)\n";
    } else {
        os << "L1 radius:                  auto-probe ("
           << static_cast<int>(g_config.probe_quantile * 100.0f)
           << "th pct, " << g_config.probe_num_samples << " samples)\n";
    }
    os << "max_nbr_size:               " << g_config.max_nbr_size << "\n"
       << "search_nn_qs:               " << g_config.search_nn_qs << "\n"
       << "select_nbrs_qs:             " << g_config.select_nbrs_qs << "\n"
       << "scale_coeffs:               " << g_config.scale_coeffs << "\n"
       << "shifted_coeffs:             " << g_config.shifted_coeffs << "\n"
       << "--- Per-layer refinement ---\n"
       << "refining_max_nbr_size:      " << g_config.refining_max_nbr_size
       << " (reserved = 1.5x = "
       << static_cast<uint32_t>(g_config.refining_max_nbr_size * 1.5f) << ")\n"
       << "num_build_loops:            " << g_config.num_build_loops << "\n"
       << "num_triu_iters:             " << g_config.num_triu_iters << "\n"
       << "prefill_ratio:              " << g_config.prefill_ratio
       << " (L0 random prefill: "
       << static_cast<uint32_t>(g_config.prefill_ratio
              * g_config.refining_max_nbr_size)
       << " edges/vertex)\n"
       << "num_routing_loops:          " << g_config.num_routing_loops << "\n"
       << "routing_topk:               " << g_config.routing_topk << "\n"
       << "routing_queue_size:         " << g_config.routing_queue_size << "\n"
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

        if (g_config.l1_radius_provided) {
            _l1_radius = g_config.l1_rnet_radius;
            ARTEA_INFO(fmt::format(
                "Using user-provided L1 rnet_radius = {:.6f}", _l1_radius));
        } else {
            dataset_prober_t prober(base_vecs, *_dist_func);
            const std::vector<float> quantiles = { g_config.probe_quantile };
            ARTEA_INFO(fmt::format(
                "Probing L1 rnet_radius ({}th pct, {} samples)...",
                static_cast<int>(g_config.probe_quantile * 100.0f),
                g_config.probe_num_samples));
            auto result = prober.probe(quantiles, g_config.probe_num_samples);
            _l1_radius = static_cast<float>(result.table[0][0]);
            ARTEA_INFO(fmt::format(
                "Auto-probed L1 rnet_radius = {:.6f}", _l1_radius));
        }
    }

    auto get_dataset()   -> vector_dataset_t& { return *_dataset; }
    auto get_dist_func() -> dist_func_t&      { return *_dist_func; }
    auto get_l1_radius() const -> float       { return _l1_radius; }

private:
    DataProvider() = default;

    std::unique_ptr<vector_dataset_t> _dataset;
    std::unique_ptr<dist_func_t>      _dist_func;
    float                             _l1_radius = 0.0f;
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
            provider.get_l1_radius(),
            static_cast<vertex_num_t>(g_config.search_nn_qs),
            static_cast<vertex_num_t>(g_config.select_nbrs_qs),
            g_config.max_nbr_size);

        // Reserved capacity is fixed at 1.5x the max — sized for the
        // overflow headroom the per-layer refiner needs without
        // exposing it as a separate knob.
        const vertex_num_t refining_reserved_nbr_size =
            static_cast<vertex_num_t>(g_config.refining_max_nbr_size * 1.5f);
        layer_config_t refining_layer_config(
            g_config.refining_max_nbr_size,
            refining_reserved_nbr_size);

        // prefill_ratio drives L0 random prefill in refine_layer:
        // each L0 vertex gets prefill_ratio * refining_max_nbr_size
        // random neighbors sampled from the new-vid window before the
        // triangle/reverse/truncate pipeline starts.
        artea_graph::propagate_config_t propagate_config(
            static_cast<iter_t>(g_config.num_build_loops),
            static_cast<iter_t>(g_config.num_triu_iters),
            static_cast<ratio_t>(g_config.prefill_ratio),
            static_cast<iter_t>(g_config.num_routing_loops),
            static_cast<vertex_num_t>(g_config.routing_topk),
            static_cast<vertex_num_t>(g_config.routing_queue_size));

        artea_graph::pruning_config_t pruning_config(
            static_cast<ratio_t>(g_config.scale_coeffs),
            static_cast<ratio_t>(g_config.shifted_coeffs));

        _graph = std::make_unique<artea_graph::index_t>(
            total_vertices, rgraph_config,
            refining_layer_config, propagate_config, pruning_config);

        ARTEA_INFO("Building artea_graph (3 steps: r-net insert → "
                   "per-layer refine → writeback)...");
        auto t0 = std::chrono::high_resolution_clock::now();

        vector_array_t owned_batch =
            base_vecs.extract_subset(0, total_vertices);
        artea_graph::factory_t::add_vertices(
            *_graph, std::move(owned_batch), dist_func);

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

        for (layer_id_t h = 0; h <= _graph->max_restrict_level(); ++h) {
            const auto& bucket = _graph->get_vids_with_highest_level(h);
            const float ratio = 100.0f * bucket.size() /
                base_vecs.get_num_vecs();
            ARTEA_INFO(fmt::format(
                "  highest_level_id={}: {} vertices ({:.2f}% of base)",
                h, bucket.size(), ratio));
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
    struct Row {
        uint32_t queue_size;
        // Static artea: compact hierarchical graph, greedy-upper + beam-L0.
        double   s_batch_ms;
        double   s_qps;
        float    s_recall;
        // Dynamic artea: dynamic hierarchical graph, same greedy-upper + beam-L0.
        double   d_batch_ms;
        double   d_qps;
        float    d_recall;
        // Static artea L0-only baseline: flat beam on compact L0 from entry_point.
        double   l0_batch_ms;
        double   l0_qps;
        float    l0_recall;
    };
    std::vector<Row> rows;

    // Runs @p batch_call (a batch query closure) over warmup + test_runs
    // iterations, averages latency/recall, and returns them.
    auto time_batch = [&](auto&& batch_call)
        -> std::tuple<double, float, knn_results_t>
    {
        for (uint32_t w = 0; w < g_config.warmup_runs; ++w) {
            [[maybe_unused]] auto _ = batch_call();
        }
        double total_us = 0.0;
        float  total_recall = 0.0f;
        knn_results_t last_results;
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

    for (uint32_t queue_size = g_config.queue_size_start;
         queue_size <= g_config.queue_size_end;
         queue_size += g_config.queue_size_step)
    {
        const uint32_t effective_queue_size =
            std::max<uint32_t>(queue_size, topk);

        compact::hierarchical_graph_router_t s_router(
            base_vecs, dist_func,
            /*topk=*/topk,
            /*candidate_queue_size=*/effective_queue_size);
        s_router.initialize();

        dynamic::hierarchical_graph_router_t d_router(
            base_vecs, dist_func,
            /*topk=*/topk,
            /*candidate_queue_size=*/effective_queue_size);
        d_router.initialize();

        // --- Static artea: compact hg, greedy-upper + beam-L0 ---
        auto [s_avg_us, s_recall, s_last] = time_batch([&]() {
            return s_router.template batch_query</*RandomSeeding=*/false, /*UpperLevelBeamSearch=*/false>(
                query_vecs, compact_hg);
        });
        ASSERT_EQ(s_last.size(), static_cast<std::size_t>(num_queries) * topk);

        // --- Dynamic artea: dynamic hg, greedy-upper + beam-L0 ---
        auto [d_avg_us, d_recall, d_last] = time_batch(
            [&]() {
                return d_router.template batch_query</*UpperLevelBeamSearch=*/false>(
                    query_vecs, dyn_hg);
            });
        ASSERT_EQ(d_last.size(),
                  static_cast<std::size_t>(num_queries) * topk);

        // --- Static artea L0-only (flat beam on compact L0 from entry_point) ---
        auto [l0_avg_us, l0_recall, l0_last] = time_batch(
            [&]() {
                return s_router.template batch_query_l0_only</*RandomSeeding=*/false>(
                    query_vecs, compact_hg);
            });
        ASSERT_EQ(l0_last.size(),
                  static_cast<std::size_t>(num_queries) * topk);

        Row row;
        row.queue_size  = effective_queue_size;
        row.s_batch_ms  = s_avg_us  / 1000.0;
        row.s_qps       = num_queries * 1e6 / s_avg_us;
        row.s_recall    = s_recall;
        row.d_batch_ms  = d_avg_us  / 1000.0;
        row.d_qps       = num_queries * 1e6 / d_avg_us;
        row.d_recall    = d_recall;
        row.l0_batch_ms = l0_avg_us / 1000.0;
        row.l0_qps      = num_queries * 1e6 / l0_avg_us;
        row.l0_recall   = l0_recall;
        rows.push_back(row);

        ARTEA_INFO(fmt::format(
            "CandidateQueue={:4}: "
            "static  [R@{}={:.4f}, QPS={:8.1f}, batch={:.2f} ms] | "
            "dynamic [R@{}={:.4f}, QPS={:8.1f}, batch={:.2f} ms] | "
            "L0-only [R@{}={:.4f}, QPS={:8.1f}, batch={:.2f} ms]",
            effective_queue_size,
            topk, row.s_recall,  row.s_qps,  row.s_batch_ms,
            topk, row.d_recall,  row.d_qps,  row.d_batch_ms,
            topk, row.l0_recall, row.l0_qps, row.l0_batch_ms));
    }

    ARTEA_INFO("=== static artea vs. dynamic artea vs. static artea L0 summary ===");
    ARTEA_INFO(fmt::format("  build_time    : {} ms", _build_ms));
    ARTEA_INFO(fmt::format("  compact_time  : {} ms", compact_ms));
    ARTEA_INFO(fmt::format("  num_queries   : {}", num_queries));
    ARTEA_INFO(fmt::format("  topk          : {}", topk));
    ARTEA_INFO(fmt::format(
        "{:<8} | {:<10} {:<10} {:<10} | {:<10} {:<10} {:<10} | {:<10} {:<10} {:<10}",
        "Queue",
        "S.Recall@k",  "S.QPS",  "S.batch(ms)",
        "D.Recall@k",  "D.QPS",  "D.batch(ms)",
        "L0.Recall@k", "L0.QPS", "L0.batch(ms)"));
    ARTEA_INFO(std::string(
        8 + 3 + 10 + 10 + 10 + 3 + 10 + 10 + 10 + 3 + 10 + 10 + 10, '-'));
    for (const auto& row : rows) {
        ARTEA_INFO(fmt::format(
            "{:<8} | {:<10.4f} {:<10.1f} {:<10.2f} | "
            "{:<10.4f} {:<10.1f} {:<10.2f} | "
            "{:<10.4f} {:<10.1f} {:<10.2f}",
            row.queue_size,
            row.s_recall,  row.s_qps,  row.s_batch_ms,
            row.d_recall,  row.d_qps,  row.d_batch_ms,
            row.l0_recall, row.l0_qps, row.l0_batch_ms));
    }

    bool has_any_static  = false;
    bool has_any_dynamic = false;
    bool has_any_l0      = false;
    for (const auto& row : rows) {
        if (row.s_recall  > 0.0f) has_any_static  = true;
        if (row.d_recall  > 0.0f) has_any_dynamic = true;
        if (row.l0_recall > 0.0f) has_any_l0      = true;
    }
    EXPECT_TRUE(has_any_static)
        << "At least one queue-size should return static artea results";
    EXPECT_TRUE(has_any_dynamic)
        << "At least one queue-size should return dynamic artea results";
    EXPECT_TRUE(has_any_l0)
        << "At least one queue-size should return L0-only results";
}

// ============================================================
//  main: argparse + test suite runner
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_artea_graph");
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to datasets.json config file");
    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name (as listed in datasets.json)");

    // Stacked r-net backbone
    program.add_argument("--beta")
        .default_value(2.0f).scan<'g', float>();
    program.add_argument("--l1-radius")
        .default_value(-1.0f).scan<'g', float>()
        .help("L1 rnet_radius. If negative, auto-probe via DatasetProber.");
    program.add_argument("--max-nbr-size")
        .default_value(32u).scan<'u', uint32_t>();
    program.add_argument("--search-nn-qs")
        .default_value(40u).scan<'u', uint32_t>();
    program.add_argument("--select-nbrs-qs")
        .default_value(100u).scan<'u', uint32_t>();
    program.add_argument("--scale-coeffs")
        .default_value(1.1f).scan<'g', float>();
    program.add_argument("--shifted-coeffs")
        .default_value(0.0f).scan<'g', float>();

    program.add_argument("--probe-num-samples")
        .default_value(500u).scan<'u', uint32_t>();
    program.add_argument("--probe-quantile")
        .default_value(0.9f).scan<'g', float>();

    // Per-layer refinement
    program.add_argument("--refining-max-nbr-size")
        .default_value(32u).scan<'u', uint32_t>();
    program.add_argument("--num-build-loops")
        .default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters")
        .default_value(14u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio")
        .default_value(0.34f).scan<'g', float>()
        .help("L0 random-prefill density: each L0 vertex receives "
              "prefill_ratio * refining_max_nbr_size random neighbors "
              "before the propagate loop.");
    program.add_argument("--num-routing-loops")
        .default_value(1u).scan<'u', uint32_t>();
    program.add_argument("--routing-topk")
        .default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--routing-queue-size")
        .default_value(96u).scan<'u', uint32_t>();

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

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program;
        return 1;
    }

    g_config.config_path               = program.get<std::string>("--config");
    g_config.dataset_name              = program.get<std::string>("--dataset");
    g_config.rnet_beta                 = program.get<float>("--beta");
    g_config.max_nbr_size              = program.get<uint32_t>("--max-nbr-size");
    g_config.search_nn_qs              = program.get<uint32_t>("--search-nn-qs");
    g_config.select_nbrs_qs            = program.get<uint32_t>("--select-nbrs-qs");
    g_config.scale_coeffs              = program.get<float>("--scale-coeffs");
    g_config.shifted_coeffs            = program.get<float>("--shifted-coeffs");
    g_config.probe_num_samples         = program.get<uint32_t>("--probe-num-samples");
    g_config.probe_quantile            = program.get<float>("--probe-quantile");
    g_config.refining_max_nbr_size     = program.get<uint32_t>("--refining-max-nbr-size");
    g_config.num_build_loops           = program.get<uint32_t>("--num-build-loops");
    g_config.num_triu_iters            = program.get<uint32_t>("--num-triu-iters");
    g_config.prefill_ratio             = program.get<float>("--prefill-ratio");
    g_config.num_routing_loops         = program.get<uint32_t>("--num-routing-loops");
    g_config.routing_topk              = program.get<uint32_t>("--routing-topk");
    g_config.routing_queue_size        = program.get<uint32_t>("--routing-queue-size");
    g_config.query_topk                = program.get<uint32_t>("--query-topk");
    g_config.warmup_runs               = program.get<uint32_t>("--warmup-runs");
    g_config.test_runs                 = program.get<uint32_t>("--test-runs");

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

    const float l1 = program.get<float>("--l1-radius");
    g_config.l1_radius_provided = (l1 >= 0.0f);
    g_config.l1_rnet_radius     = l1;

    dump_config("artea_graph test — configuration (start)");

    DataProvider::instance().init();
    const int rc = RUN_ALL_TESTS();

    dump_config("artea_graph test — configuration (end)");
    return rc;
}
