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
 * @FilePath: /Artea/unit_tests/test_hierarchical_graph_persistence.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Round-trip test for HierarchicalGraphFileManager.
 *               Builds a stacked_rgraph backbone on a real dataset
 *               subset, compacts it, snapshots the compact graph,
 *               restores it from disk, and asserts the restored graph
 *               is observationally equivalent to the original via the
 *               public read API (scalar metadata, per-vid
 *               highest_level_id, per-(vid, level) valid-neighbor
 *               spans, per-h bucket membership, and entry_point_vid).
 *
 *               Note on bucket ordering: the compactor and the
 *               restorer may produce different vid orderings inside
 *               each bucket (the compactor bulk-copies from a TBB
 *               concurrent_vector; the restorer scans in
 *               vid-ascending order). Bucket comparisons are
 *               therefore set-based, not sequence-based.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include <argparse/argparse.hpp>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

// ============================================================
//  Global configuration
// ============================================================

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    std::string snapshot_path;

    // Graph build params (mirror test_stacked_rgraph.cpp defaults
    // with tighter neighbor caps to keep the round-trip test
    // tractable on a typical workstation).
    float    rnet_beta;
    bool     l0_radius_provided;
    float    l0_rnet_radius;
    uint32_t ul_max_nbr_size;
    uint32_t bl_max_nbr_size;

    uint32_t probe_num_samples;
    float    probe_quantile;

    uint32_t search_nn_qs;
    uint32_t ul_select_nbrs_qs;
    uint32_t bl_select_nbrs_qs;

    float    scale_coeffs;

    bool     verbose;
} g_config;

// ============================================================
//  DataProvider singleton (lazy: dataset + auto-probed L0 radius)
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

    void cleanup() {
        std::error_code ec;
        std::filesystem::remove(g_config.snapshot_path, ec);
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
//  Fixture: build → compact once per test-suite, reuse compact
//  graph across tests.
// ============================================================

class HierarchicalGraphPersistenceTest : public ::testing::Test {
protected:
    static auto build_compact_graph()
        -> std::unique_ptr<compact::hierarchical_graph_t>
    {
        auto& provider = DataProvider::instance();
        const auto& base_vecs = provider.get_dataset().get_base_vecs();
        auto& dist_func = provider.get_dist_func();

        const vertex_num_t total_vertices =
            static_cast<vertex_num_t>(base_vecs.get_num_vecs());

        stacked_rgraph::rgraph_config_t rgraph_config(
            g_config.rnet_beta, provider.get_l0_radius(),
            static_cast<vertex_num_t>(g_config.search_nn_qs),
            static_cast<vertex_num_t>(g_config.ul_select_nbrs_qs),
            static_cast<vertex_num_t>(g_config.bl_select_nbrs_qs),
            g_config.ul_max_nbr_size,
            g_config.bl_max_nbr_size);
        stacked_rgraph::pruning_config_t pruning_config(
            static_cast<ratio_t>(g_config.scale_coeffs),
            static_cast<ratio_t>(0));

        auto graph = std::make_unique<stacked_rgraph::index_t>(
            total_vertices, rgraph_config, pruning_config);

        vector_array_t owned_batch =
            base_vecs.extract_subset(0, total_vertices);

        auto t0 = std::chrono::high_resolution_clock::now();
        stacked_rgraph::factory_t::add_vertices(
            *graph, std::move(owned_batch), dist_func,
            /*insert_on_L0=*/true);
        auto t1 = std::chrono::high_resolution_clock::now();
        ARTEA_INFO(fmt::format("Built stacked_rgraph in {} ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                .count()));

        auto t2 = std::chrono::high_resolution_clock::now();
        auto compact_hg = hierarchical_graph_compactor_t::compact_graph(
            graph->get_hierarchical_graph(), base_vecs, dist_func);
        auto t3 = std::chrono::high_resolution_clock::now();
        ARTEA_INFO(fmt::format("Compacted in {} ms; "
            "max_restrict_level={}, num_vertices={}",
            std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2)
                .count(),
            static_cast<int>(compact_hg.max_restrict_level()),
            compact_hg.get_num_vertices()));

        return std::make_unique<compact::hierarchical_graph_t>(
            std::move(compact_hg));
    }

    static void SetUpTestSuite() {
        _compact_graph = build_compact_graph();
    }

    static void TearDownTestSuite() {
        _compact_graph.reset();
    }

    static std::unique_ptr<compact::hierarchical_graph_t> _compact_graph;
};

std::unique_ptr<compact::hierarchical_graph_t>
    HierarchicalGraphPersistenceTest::_compact_graph = nullptr;

// ============================================================
//  Helper: assert that two compact::HierarchicalGraph instances
//  give identical answers through their public read API.
// ============================================================

static auto expect_compact_graphs_equal(
    const compact::hierarchical_graph_t& original,
    const compact::hierarchical_graph_t& restored
) -> void {
    // Scalar metadata.
    ASSERT_EQ(original.get_num_vertices(),     restored.get_num_vertices());
    ASSERT_EQ(original.max_restrict_level(),   restored.max_restrict_level());
    ASSERT_EQ(original.top_occupied_level_id(),
              restored.top_occupied_level_id());
    ASSERT_EQ(original.ul_max_nbr_size(),      restored.ul_max_nbr_size());
    ASSERT_EQ(original.bl_max_nbr_size(),      restored.bl_max_nbr_size());
    ASSERT_EQ(original.entry_point_vid(),      restored.entry_point_vid());

    const vertex_num_t num_vertices = original.get_num_vertices();
    const layer_id_t   max_level    = original.max_restrict_level();

    // Per-vid highest_level_id.
    for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
        ASSERT_EQ(original.get_highest_level_id(vid),
                  restored.get_highest_level_id(vid))
            << "highest_level_id differs at vid=" << vid;
    }

    // Per-h bucket as a set (order may differ — see file header).
    for (layer_id_t h = 0; h <= max_level; ++h) {
        const auto orig_bucket = original.get_vids_with_highest_level(h);
        const auto rest_bucket = restored.get_vids_with_highest_level(h);
        ASSERT_EQ(orig_bucket.size(), rest_bucket.size())
            << "bucket size differs at h=" << static_cast<int>(h);
        std::unordered_set<vertex_id_t> orig_set(
            orig_bucket.begin(), orig_bucket.end());
        for (const vertex_id_t vid : rest_bucket) {
            EXPECT_TRUE(orig_set.count(vid))
                << "restored bucket[" << static_cast<int>(h)
                << "] contains vid=" << vid
                << " not present in original";
        }
    }

    // Per-(vid, level) valid-neighbor span.
    constexpr vertex_id_t invalid_vid =
        compact::hierarchical_graph_t::invalid_vertex_id;
    constexpr layer_id_t  unassigned_h =
        compact::hierarchical_graph_t::unassigned_highest_level_id;

    std::size_t total_levels_checked = 0;
    std::size_t total_valid_edges    = 0;
    for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
        const layer_id_t H = original.get_highest_level_id(vid);
        if (H == unassigned_h) continue;
        for (layer_id_t l = 0; l <= H; ++l) {
            const auto orig_span = original.fetch_layer_nbrs(vid, l);
            const auto rest_span = restored.fetch_layer_nbrs(vid, l);
            ASSERT_EQ(orig_span.size(), rest_span.size())
                << "span size differs at vid=" << vid
                << " l=" << static_cast<int>(l);

            // Walk the valid prefix (sentinel-terminated).
            std::size_t valid_count = 0;
            for (; valid_count < orig_span.size(); ++valid_count) {
                if (orig_span[valid_count] == invalid_vid) break;
            }
            const std::size_t rest_valid =
                restored.num_valid_nbrs(vid, l);
            ASSERT_EQ(valid_count, rest_valid)
                << "valid count differs at vid=" << vid
                << " l=" << static_cast<int>(l);

            for (std::size_t i = 0; i < valid_count; ++i) {
                ASSERT_EQ(orig_span[i], rest_span[i])
                    << "nbr differs at vid=" << vid
                    << " l=" << static_cast<int>(l)
                    << " idx=" << i;
            }
            total_levels_checked += 1;
            total_valid_edges    += valid_count;
        }
    }
    ARTEA_INFO(fmt::format(
        "Verified {} (vid, level) spans / {} valid edges",
        total_levels_checked, total_valid_edges));
}

// ============================================================
//  The test: snapshot → restore → compare.
// ============================================================

TEST_F(HierarchicalGraphPersistenceTest, SnapshotRestoreRoundTrip) {
    ASSERT_NE(_compact_graph, nullptr);
    const auto& original = *_compact_graph;

    // Snapshot.
    auto t0 = std::chrono::high_resolution_clock::now();
    hierarchical_graph_file_manager_t::snapshot(
        original, g_config.snapshot_path);
    auto t1 = std::chrono::high_resolution_clock::now();

    ASSERT_TRUE(std::filesystem::exists(g_config.snapshot_path));
    const std::uintmax_t file_size =
        std::filesystem::file_size(g_config.snapshot_path);
    ARTEA_INFO(fmt::format(
        "Snapshot wrote {} bytes to {} in {} ms",
        file_size, g_config.snapshot_path,
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
            .count()));

    // Restore.
    auto t2 = std::chrono::high_resolution_clock::now();
    auto restored = hierarchical_graph_file_manager_t::restore(
        g_config.snapshot_path);
    auto t3 = std::chrono::high_resolution_clock::now();
    ARTEA_INFO(fmt::format("Restore took {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2)
            .count()));

    // Compare.
    expect_compact_graphs_equal(original, restored);

    if (g_config.verbose) {
        ARTEA_SUCCESS(
            "HierarchicalGraph snapshot/restore round-trip passed.");
    }
}

// ============================================================
//  main: argparse + test runner
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_hierarchical_graph_persistence");
    program.add_argument("-c", "--config")
        .default_value(artea::default_dataset_config_path())
        .help("Path to datasets.json config file");
    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name (as listed in datasets.json)");
    program.add_argument("--snapshot-path")
        .default_value(std::string(
            "./test_hierarchical_graph_persistence_snapshot.graph"))
        .help("Where to write the round-trip snapshot file. "
              "The file is removed on test teardown.");

    program.add_argument("--beta")
        .default_value(2.0f).scan<'g', float>();
    program.add_argument("--l0-radius")
        .default_value(25000f).scan<'g', float>()
        .help("If negative, auto-probe via DatasetProber.");
    program.add_argument("--ul-max-nbr-size")
        .default_value(32u).scan<'u', uint32_t>();
    program.add_argument("--bl-max-nbr-size")
        .default_value(64u).scan<'u', uint32_t>();

    program.add_argument("--probe-num-samples")
        .default_value(500u).scan<'u', uint32_t>();
    program.add_argument("--probe-quantile")
        .default_value(0.9f).scan<'g', float>();

    program.add_argument("--search-nn-qs")
        .default_value(40u).scan<'u', uint32_t>();
    program.add_argument("--ul-select-nbrs-qs")
        .default_value(50u).scan<'u', uint32_t>();
    program.add_argument("--bl-select-nbrs-qs")
        .default_value(50u).scan<'u', uint32_t>();
    program.add_argument("--scale-coeffs")
        .default_value(1.1f).scan<'g', float>();

    program.add_argument("-v", "--verbose")
        .default_value(false).implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program;
        return 1;
    }

    g_config.config_path        = program.get<std::string>("--config");
    g_config.dataset_name       = program.get<std::string>("--dataset");
    g_config.snapshot_path      = program.get<std::string>("--snapshot-path");
    g_config.rnet_beta          = program.get<float>("--beta");
    g_config.ul_max_nbr_size    = program.get<uint32_t>("--ul-max-nbr-size");
    g_config.bl_max_nbr_size    = program.get<uint32_t>("--bl-max-nbr-size");
    g_config.probe_num_samples  = program.get<uint32_t>("--probe-num-samples");
    g_config.probe_quantile     = program.get<float>("--probe-quantile");
    g_config.search_nn_qs       = program.get<uint32_t>("--search-nn-qs");
    g_config.ul_select_nbrs_qs  = program.get<uint32_t>("--ul-select-nbrs-qs");
    g_config.bl_select_nbrs_qs  = program.get<uint32_t>("--bl-select-nbrs-qs");
    g_config.scale_coeffs       = program.get<float>("--scale-coeffs");
    g_config.verbose            = program.get<bool>("--verbose");

    const float l0_radius_arg = program.get<float>("--l0-radius");
    g_config.l0_radius_provided = (l0_radius_arg >= 0.0f);
    g_config.l0_rnet_radius     = l0_radius_arg;

    std::cout << "\n=== Test Configuration ===\n"
              << "Dataset:           " << g_config.dataset_name      << "\n"
              << "Config path:       " << g_config.config_path       << "\n"
              << "Snapshot path:     " << g_config.snapshot_path     << "\n"
              << "ul_max_nbr_size:   " << g_config.ul_max_nbr_size   << "\n"
              << "bl_max_nbr_size:   " << g_config.bl_max_nbr_size   << "\n"
              << "===========================\n\n";

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    DataProvider::instance().cleanup();
    return result;
}
