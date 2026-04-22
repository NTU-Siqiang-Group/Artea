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
 * @FilePath: /Artea/unit_tests/test_stacked_rgraph.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Build-time benchmark for the stacked r-net backbone.
 *               Tests both insert_on_L0=false (upper-layer edges only,
 *               as used by artea_graph refinement) and insert_on_L0=true
 *               (all layers including L0). Reports wall-clock times for
 *               both modes.
 */

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

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

    // StackedRGraph parameters
    float    rnet_beta;
    bool     l1_radius_provided;
    float    l1_rnet_radius;
    uint32_t max_nbr_size;

    // L1-radius auto-probe
    uint32_t probe_num_samples;
    float    probe_quantile;

    // Beam-search queue sizes
    uint32_t search_nn_qs;
    uint32_t ul_select_nbrs_qs;
    uint32_t bl_select_nbrs_qs;

    // Insert-time upper-layer RNG scale (shifted is forced to 0 at the
    // stacked_rgraph config level).
    float    scale_coeffs;
} g_config;

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
//  Fixture: build both insert_on_L0=false and =true graphs.
// ============================================================

class StackedRGraphTest : public ::testing::Test {
protected:
    static auto build_graph(bool insert_on_L0)
        -> std::pair<std::unique_ptr<stacked_rgraph::index_t>, int64_t>
    {
        auto& provider = DataProvider::instance();
        const auto& base_vecs = provider.get_dataset().get_base_vecs();
        auto& dist_func = provider.get_dist_func();

        const vertex_num_t total_vertices =
            static_cast<vertex_num_t>(base_vecs.get_num_vecs());
        stacked_rgraph::rgraph_config_t rgraph_config(
            g_config.rnet_beta, provider.get_l1_radius(),
            static_cast<vertex_num_t>(g_config.search_nn_qs),
            static_cast<vertex_num_t>(g_config.ul_select_nbrs_qs),
            static_cast<vertex_num_t>(g_config.bl_select_nbrs_qs),
            g_config.max_nbr_size);
        // stacked_rgraph::pruning_config_t is an alias of conv_graph's
        // PruningConfig; the stacked_rgraph backbone only reads
        // scale_coeffs from it, so shifted_coeffs is pinned to 0 here.
        stacked_rgraph::pruning_config_t pruning_config(
            static_cast<ratio_t>(g_config.scale_coeffs),
            static_cast<ratio_t>(0));
        auto graph = std::make_unique<stacked_rgraph::index_t>(
            total_vertices, rgraph_config, pruning_config);

        vector_array_t owned_batch =
            base_vecs.extract_subset(0, total_vertices);

        auto t0 = std::chrono::high_resolution_clock::now();
        stacked_rgraph::factory_t::add_vertices(
            *graph, std::move(owned_batch), dist_func, insert_on_L0);
        auto t1 = std::chrono::high_resolution_clock::now();
        const int64_t ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        return {std::move(graph), ms};
    }

    static void dump_graph_info(
        const stacked_rgraph::index_t& graph,
        const char* label, int64_t build_ms)
    {
        const auto& base_vecs =
            DataProvider::instance().get_dataset().get_base_vecs();
        const layer_id_t top_level_id = graph.top_occupied_level_id();
        ARTEA_INFO(fmt::format(
            "[{}] built in {} ms: top_occupied_level={}, max_restrict_level={}",
            label, build_ms,
            (top_level_id == dynamic::hierarchical_graph_t
                ::unassigned_highest_level_id)
                ? -1 : static_cast<int>(top_level_id),
            graph.max_restrict_level()));
        for (layer_id_t h = 0; h <= graph.max_restrict_level(); ++h) {
            const auto& bucket = graph.get_vids_with_highest_level(h);
            const float ratio = 100.0f * bucket.size() /
                base_vecs.get_num_vecs();
            ARTEA_INFO(fmt::format(
                "  highest_level_id={}: {} vertices ({:.2f}% of base)",
                h, bucket.size(), ratio));
        }
    }

    static void SetUpTestSuite() {
        ARTEA_INFO(fmt::format(
            "Building StackedRGraph: beta={:.3f}, L1_radius={:.6f}, "
            "max_nbr={}, search_nn_qs={}, "
            "ul_select_nbrs_qs={}, bl_select_nbrs_qs={}, "
            "scale_coeffs={:.3f}",
            g_config.rnet_beta,
            DataProvider::instance().get_l1_radius(),
            g_config.max_nbr_size,
            g_config.search_nn_qs,
            g_config.ul_select_nbrs_qs, g_config.bl_select_nbrs_qs,
            g_config.scale_coeffs));

        ARTEA_INFO("--- Building with insert_on_L0=false ---");
        std::tie(_graph_no_l0, _build_ms_no_l0) = build_graph(/*insert_on_L0=*/false);
        dump_graph_info(*_graph_no_l0, "insert_on_L0=false", _build_ms_no_l0);

        ARTEA_INFO("--- Building with insert_on_L0=true ---");
        std::tie(_graph_with_l0, _build_ms_with_l0) = build_graph(/*insert_on_L0=*/true);
        dump_graph_info(*_graph_with_l0, "insert_on_L0=true", _build_ms_with_l0);
    }

    static void TearDownTestSuite() {
        _graph_no_l0.reset();
        _graph_with_l0.reset();
    }

    static std::unique_ptr<stacked_rgraph::index_t> _graph_no_l0;
    static std::unique_ptr<stacked_rgraph::index_t> _graph_with_l0;
    static int64_t _build_ms_no_l0;
    static int64_t _build_ms_with_l0;
};

std::unique_ptr<stacked_rgraph::index_t> StackedRGraphTest::_graph_no_l0   = nullptr;
std::unique_ptr<stacked_rgraph::index_t> StackedRGraphTest::_graph_with_l0 = nullptr;
int64_t StackedRGraphTest::_build_ms_no_l0  = 0;
int64_t StackedRGraphTest::_build_ms_with_l0 = 0;

// ============================================================
//  Build time comparison.
// ============================================================

TEST_F(StackedRGraphTest, BuildTime) {
    ASSERT_NE(_graph_no_l0, nullptr);
    ASSERT_NE(_graph_with_l0, nullptr);
    EXPECT_GT(_build_ms_no_l0, 0);
    EXPECT_GT(_build_ms_with_l0, 0);

    ARTEA_INFO("=== Build summary ===");
    ARTEA_INFO(fmt::format("  dataset           : {}", g_config.dataset_name));
    ARTEA_INFO(fmt::format("  num_vertices      : {}", _graph_no_l0->get_num_vertices()));
    ARTEA_INFO(fmt::format("  max_restrict_level: {}",
        static_cast<int>(_graph_no_l0->max_restrict_level())));
    ARTEA_INFO(fmt::format("  insert_on_L0=false: {} ms", _build_ms_no_l0));
    ARTEA_INFO(fmt::format("  insert_on_L0=true : {} ms", _build_ms_with_l0));
    ARTEA_INFO(fmt::format("  L0 overhead       : {} ms ({:.1f}%)",
        _build_ms_with_l0 - _build_ms_no_l0,
        100.0 * (_build_ms_with_l0 - _build_ms_no_l0) /
            std::max<int64_t>(_build_ms_no_l0, 1)));
}

// ============================================================
//  main: argparse + test suite runner
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_stacked_rgraph");
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to datasets.json config file");
    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name (as listed in datasets.json)");

    program.add_argument("--beta")
        .default_value(2.0f).scan<'g', float>()
        .help("R-net radius growth factor between layers");
    program.add_argument("--l1-radius")
        .default_value(-1.0f).scan<'g', float>()
        .help("L1 rnet_radius. If negative, auto-probe via DatasetProber.");
    program.add_argument("--max-nbr-size")
        .default_value(32u).scan<'u', uint32_t>()
        .help("Per-vertex neighbor capacity at upper levels (level 0 = 2x).");

    program.add_argument("--probe-num-samples")
        .default_value(500u).scan<'u', uint32_t>();
    program.add_argument("--probe-quantile")
        .default_value(0.9f).scan<'g', float>();

    program.add_argument("--search-nn-qs")
        .default_value(40u).scan<'u', uint32_t>();
    program.add_argument("--ul-select-nbrs-qs")
        .default_value(100u).scan<'u', uint32_t>()
        .help("Upper-layer (L1+) beam-search queue size for the select phase.");
    program.add_argument("--bl-select-nbrs-qs")
        .default_value(100u).scan<'u', uint32_t>()
        .help("Bottom-layer (L0) beam-search queue size for the select phase.");
    program.add_argument("--scale-coeffs")
        .default_value(1.1f).scan<'g', float>()
        .help("RNG scale coefficient applied at upper layers (default 1.1). "
              "shifted_coeffs is forced to 0 at the stacked_rgraph config.");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program;
        return 1;
    }

    g_config.config_path        = program.get<std::string>("--config");
    g_config.dataset_name       = program.get<std::string>("--dataset");
    g_config.rnet_beta          = program.get<float>("--beta");
    g_config.max_nbr_size       = program.get<uint32_t>("--max-nbr-size");
    g_config.probe_num_samples  = program.get<uint32_t>("--probe-num-samples");
    g_config.probe_quantile     = program.get<float>("--probe-quantile");
    g_config.search_nn_qs       = program.get<uint32_t>("--search-nn-qs");
    g_config.ul_select_nbrs_qs  = program.get<uint32_t>("--ul-select-nbrs-qs");
    g_config.bl_select_nbrs_qs  = program.get<uint32_t>("--bl-select-nbrs-qs");
    g_config.scale_coeffs       = program.get<float>("--scale-coeffs");

    const float l1 = program.get<float>("--l1-radius");
    g_config.l1_radius_provided = (l1 >= 0.0f);
    g_config.l1_rnet_radius     = l1;

    std::cout << "\n=== Test Configuration ===\n";
    std::cout << "Dataset:        " << g_config.dataset_name   << "\n";
    std::cout << "rnet_beta:      " << g_config.rnet_beta      << "\n";
    if (g_config.l1_radius_provided) {
        std::cout << "L1 radius:      " << g_config.l1_rnet_radius
                  << " (user-provided)\n";
    } else {
        std::cout << "L1 radius:      auto-probe ("
                  << static_cast<int>(g_config.probe_quantile * 100.0f)
                  << "th pct, " << g_config.probe_num_samples
                  << " samples)\n";
    }
    std::cout << "max_nbr_size:      " << g_config.max_nbr_size      << "\n";
    std::cout << "search_nn_qs:      " << g_config.search_nn_qs      << "\n";
    std::cout << "ul_select_nbrs_qs: " << g_config.ul_select_nbrs_qs << "\n";
    std::cout << "bl_select_nbrs_qs: " << g_config.bl_select_nbrs_qs << "\n";
    std::cout << "scale_coeffs:      " << g_config.scale_coeffs      << "\n";
    std::cout << "modes:          insert_on_L0={false, true}\n";
    std::cout << "============================\n\n";

    DataProvider::instance().init();
    return RUN_ALL_TESTS();
}
