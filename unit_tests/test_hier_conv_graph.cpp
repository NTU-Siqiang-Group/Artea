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
 * @FilePath: /Artea/unit_tests/test_hier_conv_graph.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: End-to-end test for hier_conv_graph. Builds the index
 *               via the 2-step pipeline (random pull-out level
 *               assignment → per-layer conv_graph refinement), then
 *               measures search recall and throughput on the compact
 *               hierarchical graph router. Mirrors test_artea_graph in
 *               argparse / fixture layout but drops the r-net,
 *               ARC-pruning and stacked_rgraph-specific knobs.
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

    // Hierarchy shape
    uint32_t ul_max_nbr_size;
    uint32_t bl_max_nbr_size;
    float    sample_ratio;

    // conv_graph pruning + propagate
    float    scale_coeffs;
    float    shifted_coeffs;
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
       << "--- Hierarchy shape ---\n"
       << "ul_max_nbr_size:            " << g_config.ul_max_nbr_size << "\n"
       << "bl_max_nbr_size:            " << g_config.bl_max_nbr_size << "\n"
       << "sample_ratio:               " << g_config.sample_ratio
       << " (fraction of L_h vids promoted to L_{h+1})\n"
       << "--- Per-layer conv_graph refinement ---\n"
       << "ul_refining_max_nbr_size:   "
       << static_cast<uint32_t>(g_config.ul_max_nbr_size * 1.5f)
       << " (= ul_max_nbr_size × 1.5)\n"
       << "bl_refining_max_nbr_size:   "
       << static_cast<uint32_t>(g_config.bl_max_nbr_size * 1.5f)
       << " (= bl_max_nbr_size × 1.5)\n"
       << "scale_coeffs:               " << g_config.scale_coeffs << "\n"
       << "shifted_coeffs:             " << g_config.shifted_coeffs << "\n"
       << "num_build_loops:            " << g_config.num_build_loops << "\n"
       << "num_triu_iters:             " << g_config.num_triu_iters << "\n"
       << "prefill_ratio:              " << g_config.prefill_ratio
       << " (init_nbr_size = max_nbr_size * prefill_ratio random neighbors per row)\n"
       << "num_routing_loops:          " << g_config.num_routing_loops << "\n"
       << "routing_topk:               " << g_config.routing_topk << "\n"
       << "routing_queue_size:         " << g_config.routing_queue_size << "\n"
       << "--- Search ---\n"
       << "query_topk:                 " << g_config.query_topk << "\n"
       << "candidate-queue-config:     "
       << g_config.queue_size_start << "," << g_config.queue_size_end << ","
       << g_config.queue_size_step << "\n"
       << "warmup_runs:                " << g_config.warmup_runs << "\n"
       << "test_runs:                  " << g_config.test_runs << "\n"
       << "========================================\n";
    std::cout << os.str();
}

}  // anonymous namespace

// ============================================================
//  DataProvider singleton (same shape as test_artea_graph, minus the
//  L0-radius probe — hier_conv_graph has no r-net radius)
// ============================================================

class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }
        ARTEA_INFO(fmt::format("Loading dataset: {} from {}",
            g_config.dataset_name, g_config.config_path));
        _dataset = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);

        const auto& base_vecs = _dataset->get_base_vecs();
        ARTEA_INFO(fmt::format("Dataset loaded: {} vectors, {} dims",
            base_vecs.get_num_vecs(), base_vecs.get_vec_dim()));

        _dist_func = std::make_unique<dist_func_t>(base_vecs.get_vec_dim());
    }

    auto get_dataset()   -> vector_dataset_t& { return *_dataset; }
    auto get_dist_func() -> dist_func_t&      { return *_dist_func; }

private:
    DataProvider() = default;

    std::unique_ptr<vector_dataset_t> _dataset;
    std::unique_ptr<dist_func_t>      _dist_func;
};

// ============================================================
//  Fixture: build the hier_conv_graph index once for the suite.
// ============================================================

class HierConvGraphTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto& provider  = DataProvider::instance();
        const auto& base_vecs = provider.get_dataset().get_base_vecs();
        auto& dist_func = provider.get_dist_func();

        const vertex_num_t total_vertices = static_cast<vertex_num_t>(base_vecs.get_num_vecs());

        hier_conv_graph::hierarchy_config_t hierarchy_config(
            g_config.ul_max_nbr_size, g_config.bl_max_nbr_size,
            static_cast<ratio_t>(g_config.sample_ratio));

        hier_conv_graph::propagate_config_t propagate_config(
            static_cast<iter_t>(g_config.num_build_loops),
            static_cast<iter_t>(g_config.num_triu_iters),
            static_cast<ratio_t>(g_config.prefill_ratio),
            static_cast<iter_t>(g_config.num_routing_loops),
            static_cast<vertex_num_t>(g_config.routing_topk),
            static_cast<vertex_num_t>(g_config.routing_queue_size));

        hier_conv_graph::pruning_config_t pruning_config(
            static_cast<ratio_t>(g_config.scale_coeffs),
            static_cast<ratio_t>(g_config.shifted_coeffs));

        _graph = std::make_unique<hier_conv_graph::index_t>(
            total_vertices, hierarchy_config, propagate_config, pruning_config);

        ARTEA_INFO("Building hier_conv_graph (2 steps: random level assignment → per-layer refine)...");
        auto wallclock_start = std::chrono::high_resolution_clock::now();

        vector_array_t owned_batch = base_vecs.extract_subset(0, total_vertices);
        hier_conv_graph::factory_t::add_vertices(*_graph, std::move(owned_batch), dist_func);

        auto wallclock_end = std::chrono::high_resolution_clock::now();
        _build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            wallclock_end - wallclock_start).count();

        const layer_id_t top_level_id = _graph->top_occupied_level_id();
        ARTEA_INFO(fmt::format(
            "hier_conv_graph built in {} ms: top_occupied_level={}, max_restrict_level={}",
            _build_ms,
            (top_level_id == dynamic::hierarchical_graph_t::unassigned_highest_level_id)
                ? -1 : static_cast<int>(top_level_id),
            _graph->max_restrict_level()));

        // Compactor trim preview — same rule as test_artea_graph: the
        // first bucket (walking top-down) with >= min_layer_cap vids
        // becomes the new top; anything above is demoted.
        constexpr vertex_num_t min_cap = hierarchical_graph_compactor_t::min_layer_cap;
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
                "Compactor trim preview: min_layer_cap={}, new_top=L{}",
                min_cap, compactor_new_top));
        }
    }

    static void TearDownTestSuite() { _graph.reset(); }

    static std::unique_ptr<hier_conv_graph::index_t> _graph;
    static int64_t                                   _build_ms;
};

std::unique_ptr<hier_conv_graph::index_t> HierConvGraphTest::_graph = nullptr;
int64_t                                   HierConvGraphTest::_build_ms = 0;

// ============================================================
//  Build sanity: vertex count + roughly-geometric layer decay.
// ============================================================

TEST_F(HierConvGraphTest, BuildSanity) {
    ASSERT_NE(_graph, nullptr);
    const auto& base_vecs = DataProvider::instance().get_dataset().get_base_vecs();
    const auto total_vertices = static_cast<vertex_num_t>(base_vecs.get_num_vecs());
    EXPECT_EQ(_graph->get_num_vertices(), total_vertices);

    using HG = dynamic::hierarchical_graph_t;
    const layer_id_t top_level_id = _graph->top_occupied_level_id();
    ASSERT_NE(top_level_id, HG::unassigned_highest_level_id);

    // L_h vid count is everything-with-highest_level_id >= h, i.e. the
    // running total when walking from the apex down. The factory
    // prints the same breakdown, but we recompute it here so we can
    // assert on the shape.
    const layer_id_t max_level = _graph->max_restrict_level();
    std::vector<vertex_num_t> vids_at_or_above(max_level + 1, 0);
    {
        vertex_num_t running = 0;
        for (layer_id_t h = max_level; ; --h) {
            running += static_cast<vertex_num_t>(_graph->get_vids_with_highest_level(h).size());
            vids_at_or_above[h] = running;
            if (h == 0) break;
        }
    }
    EXPECT_EQ(vids_at_or_above[0], total_vertices)
        << "Every vid should participate in L0";

    // Verify the decay ratio between L_{h} and L_{h-1} sits in a sane
    // band around the configured sample_ratio. Allow 2x slack to
    // absorb the rounding (floor) + min_layer_cap cutoff. Only check
    // layers where the parent already has a non-trivial population —
    // tiny pools produce noisy ratios.
    const float configured_ratio    = g_config.sample_ratio;
    const float lower_ratio_bound   = 0.25f * configured_ratio;
    const float upper_ratio_bound   = 4.0f  * configured_ratio;
    for (layer_id_t h = 1; h <= top_level_id; ++h) {
        const auto parent_population = vids_at_or_above[h - 1];
        const auto layer_population  = vids_at_or_above[h];
        if (parent_population < 1000) continue;   // noisy at tiny parents
        const float observed_ratio = static_cast<float>(layer_population) /
                                     static_cast<float>(parent_population);
        ARTEA_INFO(fmt::format(
            "  L{}: {} vids (parent L{}: {}; observed ratio {:.4f} vs configured {:.4f})",
            h, layer_population, h - 1, parent_population,
            observed_ratio, configured_ratio));
        EXPECT_GE(observed_ratio, lower_ratio_bound)
            << "Layer " << h << " decay ratio fell below sample_ratio band";
        EXPECT_LE(observed_ratio, upper_ratio_bound)
            << "Layer " << h << " decay ratio exceeded sample_ratio band";
    }
}

// ============================================================
//  Search recall + throughput via the hierarchical router.
// ============================================================

TEST_F(HierConvGraphTest, SearchRecallAndThroughput) {
    auto& provider = DataProvider::instance();
    const auto& dataset    = provider.get_dataset();
    const auto& base_vecs  = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt         = dataset.get_gt_vecs();
    auto&       dist_func  = provider.get_dist_func();

    // ---- Compact the dynamic hierarchical graph ----
    const auto& dyn_hg = _graph->get_hierarchical_graph();
    auto compact_t0 = std::chrono::high_resolution_clock::now();
    auto compact_hg = hierarchical_graph_compactor_t::compact_graph(dyn_hg, base_vecs, dist_func);
    auto compact_t1 = std::chrono::high_resolution_clock::now();
    const int64_t compact_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(compact_t1 - compact_t0).count();
    ARTEA_INFO(fmt::format("Hierarchical graph compacted in {} ms", compact_ms));

    // ---- Grid search over candidate queue size ----
    const uint32_t topk = std::min<uint32_t>(g_config.query_topk, gt.get_vec_dim());
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
        double   batch_ms;
        double   qps;
        float    recall_at_k;
    };
    std::vector<Row> rows;

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
            total_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            total_recall += re.calculate_recall_at_k(last_results, gt, topk, num_queries);
        }
        return {
            total_us / g_config.test_runs,
            total_recall / g_config.test_runs,
            std::move(last_results)
        };
    };

    for (uint32_t queue_size = g_config.queue_size_start;
         queue_size <= g_config.queue_size_end;
         queue_size += g_config.queue_size_step)
    {
        const uint32_t effective_queue_size = std::max<uint32_t>(queue_size, topk);

        hierarchical_graph_router_t router(
            base_vecs, dist_func,
            /*topk=*/topk,
            /*candidate_queue_size=*/effective_queue_size);
        router.initialize();

        // Hierarchical router: compact hier_graph, greedy-upper + beam-L0
        // (RandomSeeding=false uses compact_hg.entry_point_vid()).
        auto [avg_us, recall, last_results] = time_batch([&]() {
            return router.template batch_query</*RandomSeeding=*/false, /*UpperLevelBeamSearch=*/false>(
                query_vecs, compact_hg);
        });
        ASSERT_EQ(last_results.size(), static_cast<std::size_t>(num_queries) * topk);

        Row row;
        row.queue_size  = effective_queue_size;
        row.batch_ms    = avg_us / 1000.0;
        row.qps         = num_queries * 1e6 / avg_us;
        row.recall_at_k = recall;
        rows.push_back(row);

        ARTEA_INFO(fmt::format(
            "CandidateQueue={:4}: hierarchical [R@{}={:.4f}, QPS={:8.1f}, batch={:.2f} ms]",
            effective_queue_size, topk, row.recall_at_k, row.qps, row.batch_ms));
    }

    ARTEA_INFO("=== hier_conv_graph hierarchical router summary ===");
    ARTEA_INFO(fmt::format("  build_time    : {} ms", _build_ms));
    ARTEA_INFO(fmt::format("  compact_time  : {} ms", compact_ms));
    ARTEA_INFO(fmt::format("  num_queries   : {}", num_queries));
    ARTEA_INFO(fmt::format("  topk          : {}", topk));
    ARTEA_INFO(fmt::format("{:<8} | {:<10} {:<10} {:<10}",
                           "Queue", "H.Recall@k", "H.QPS", "H.batch(ms)"));
    ARTEA_INFO(std::string(8 + 3 + 10 + 10 + 10, '-'));
    for (const auto& row : rows) {
        ARTEA_INFO(fmt::format("{:<8} | {:<10.4f} {:<10.1f} {:<10.2f}",
            row.queue_size, row.recall_at_k, row.qps, row.batch_ms));
    }

    bool has_any_nonzero_recall = false;
    for (const auto& row : rows) {
        if (row.recall_at_k > 0.0f) has_any_nonzero_recall = true;
    }
    EXPECT_TRUE(has_any_nonzero_recall)
        << "At least one queue-size should return hierarchical router results";
}

// ============================================================
//  main: argparse + test suite runner
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_hier_conv_graph");
    program.add_argument("-c", "--config")
        .default_value(artea::default_dataset_config_path())
        .help("Path to datasets.json config file");
    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name (as listed in datasets.json)");

    // Hierarchy shape
    program.add_argument("--ul-max-nbr-size")
        .default_value(32u).scan<'u', uint32_t>()
        .help("Per-vertex neighbor capacity at every upper layer (level_id > 0).");
    program.add_argument("--bl-max-nbr-size")
        .default_value(64u).scan<'u', uint32_t>()
        .help("Per-vertex neighbor capacity at the bottom layer (L0). "
              "Independent of --ul-max-nbr-size.");
    program.add_argument("--sample-ratio")
        .default_value(0.02f).scan<'g', float>()
        .help("Fraction of L_h vids promoted to L_{h+1} via uniform "
              "random pull-out. Default 0.02 (HNSW-like sparse upper layers).");

    // conv_graph pruning + propagate
    program.add_argument("--scale-coeffs")
        .default_value(1.1f).scan<'g', float>();
    program.add_argument("--shifted-coeffs")
        .default_value(0.0f).scan<'g', float>();
    program.add_argument("--num-build-loops")
        .default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters")
        .default_value(14u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio")
        .default_value(0.34f).scan<'g', float>()
        .help("Random-prefill density per refining layer: "
              "init_nbr_size = max_nbr_size * prefill_ratio.");
    program.add_argument("--num-routing-loops")
        .default_value(0u).scan<'u', uint32_t>();
    program.add_argument("--routing-topk")
        .default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--routing-queue-size")
        .default_value(96u).scan<'u', uint32_t>();

    // Search
    program.add_argument("--query-topk")
        .default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-config")
        .default_value(std::string("20,200,20"))
        .help("Candidate queue size grid search: start,end,step (default: 20,200,20)");
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

    g_config.config_path        = program.get<std::string>("--config");
    g_config.dataset_name       = program.get<std::string>("--dataset");
    g_config.ul_max_nbr_size    = program.get<uint32_t>("--ul-max-nbr-size");
    g_config.bl_max_nbr_size    = program.get<uint32_t>("--bl-max-nbr-size");
    g_config.sample_ratio       = program.get<float>("--sample-ratio");
    g_config.scale_coeffs       = program.get<float>("--scale-coeffs");
    g_config.shifted_coeffs     = program.get<float>("--shifted-coeffs");
    g_config.num_build_loops    = program.get<uint32_t>("--num-build-loops");
    g_config.num_triu_iters     = program.get<uint32_t>("--num-triu-iters");
    g_config.prefill_ratio      = program.get<float>("--prefill-ratio");
    g_config.num_routing_loops  = program.get<uint32_t>("--num-routing-loops");
    g_config.routing_topk       = program.get<uint32_t>("--routing-topk");
    g_config.routing_queue_size = program.get<uint32_t>("--routing-queue-size");
    g_config.query_topk         = program.get<uint32_t>("--query-topk");
    g_config.warmup_runs        = program.get<uint32_t>("--warmup-runs");
    g_config.test_runs          = program.get<uint32_t>("--test-runs");

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
            std::cerr << "Error: --candidate-queue-config must have 3 values (start,end,step)\n";
            return 1;
        }
        g_config.queue_size_start = values[0];
        g_config.queue_size_end   = values[1];
        g_config.queue_size_step  = values[2];
    }

    dump_config("hier_conv_graph test — configuration (start)");

    DataProvider::instance().init();
    const int rc = RUN_ALL_TESTS();

    dump_config("hier_conv_graph test — configuration (end)");
    return rc;
}
