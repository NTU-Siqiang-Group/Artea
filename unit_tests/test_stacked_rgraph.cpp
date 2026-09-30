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
 * @Description: Build the stacked r-net backbone from a workload JSON/JSONC file.
 *               Reports build time and vertex counts at each level, using the
 *               workload's insert_on_L0 and shuffle_insertion_order settings.
 */

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
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

using stacked_index_t = typename index_traits_t::stacked_rgraph::index_t;

// ============================================================
//  Global configuration (populated from the workload in main())
// ============================================================

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    std::string metric;

    // StackedRGraph parameters
    float    rnet_beta;
    float    shifted_coeffs;
    float    tau_k;
    bool     l0_min_distance_provided;
    float    l0_min_distance;
    uint32_t ul_max_nbr_size;
    uint32_t bl_max_nbr_size;

    // L0 minimum-distance auto-probe
    uint32_t probe_num_samples;
    float    probe_quantile;

    // Beam-search queue sizes
    uint32_t search_nn_qs;
    uint32_t ul_select_nbrs_qs;
    uint32_t bl_select_nbrs_qs;

    // Insert-time upper-layer RNG scale (shifted is forced to 0 at the
    // stacked_rgraph config level).
    float    scale_coeffs;
    bool     insert_on_L0;
    bool     shuffle_insertion_order;
} g_config;

static auto load_workload(const std::string& workload_path) -> TestConfig {
    std::ifstream workload_file(workload_path);
    if (!workload_file.is_open()) {
        throw std::runtime_error("Failed to open workload: " + workload_path);
    }
    const auto workload = nlohmann::json::parse(workload_file, nullptr, true, /*ignore_comments=*/true);
    const auto& artea_entry = workload.at("indexes-config").at("artea");

    // Match benchmark mode: use the object itself or the unique untagged array entry.
    const nlohmann::json* base_config = nullptr;
    if (artea_entry.is_object()) {
        base_config = &artea_entry;
    } else if (artea_entry.is_array()) {
        for (const auto& entry : artea_entry) {
            if (!entry.is_object() || entry.contains("tag")) continue;
            if (base_config != nullptr) {
                throw std::runtime_error("indexes-config.artea must have exactly one untagged base config");
            }
            base_config = &entry;
        }
        if (base_config == nullptr) {
            throw std::runtime_error("indexes-config.artea has no untagged base config");
        }
    } else {
        throw std::runtime_error("indexes-config.artea must be an object or array");
    }
    const auto& params = *base_config;
    if (params.contains("num_skipped_levels")) {
        throw std::runtime_error("num_skipped_levels is no longer supported; use tau_k");
    }

    TestConfig config{};
    config.config_path = workload.at("dataset-config").get<std::string>();
    config.dataset_name = workload.at("dataset").get<std::string>();
    config.metric = workload.value("metric", "euclidean");
    config.rnet_beta = params.value("rnet_beta", 2.0f);
    config.tau_k = params.value("tau_k", 0.0f);
    config.shifted_coeffs = params.value("shifted_coeffs", 0.0f);
    config.l0_min_distance_provided = params.contains("l0_min_distance");
    config.l0_min_distance = params.value("l0_min_distance", -1.0f);
    config.ul_max_nbr_size = params.value("ul_max_nbr_size", 32u);
    config.bl_max_nbr_size = params.value("bl_max_nbr_size", 64u);
    config.search_nn_qs = params.value("search_nn_qs", 30u);
    config.ul_select_nbrs_qs = params.value("ul_select_nbrs_qs", 100u);
    config.bl_select_nbrs_qs = params.value("bl_select_nbrs_qs", 100u);
    config.scale_coeffs = params.value("scale_coeffs", 1.1f);
    config.probe_num_samples = params.value("probe_num_samples", 500u);
    config.probe_quantile = params.value("probe_quantile", 0.9f);
    config.insert_on_L0 = params.value("insert_on_L0", false);
    config.shuffle_insertion_order = params.value("shuffle_insertion_order", false);
    return config;
}

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

        // Resolve BOTH compile-time axes: metric from the workload,
        // padded dim from the loaded dataset. The dataset and the probed L0
        // radius (a plain float) are metric/dim-independent; only the
        // build_dist / prober run behind <Metric, Dim>.
        _dataset_info = DatasetInfra{parse_metric(g_config.metric), base_vecs.get_vec_dim()};

        if (g_config.l0_min_distance_provided) {
            _l0_min_distance = g_config.l0_min_distance;
            ARTEA_INFO(fmt::format(
                "Using user-provided l0_min_distance = {:.6f}", _l0_min_distance));
        } else {
            build_infra_dispatch(_dataset_info, ARTEA_METRIC_LAMBDA(void) {
                // Stateless functor: the dimension is a compile-time trait now.
                dist_func_t<Metric, Dim> build_dist;
                dataset_prober_t<Metric, Dim> prober(base_vecs, build_dist);
                const std::vector<float> quantiles = { g_config.probe_quantile };
                ARTEA_INFO(fmt::format(
                    "Probing l0_min_distance ({}th pct, {} samples)...",
                    static_cast<int>(g_config.probe_quantile * 100.0f),
                    g_config.probe_num_samples));
                auto result = prober.probe(quantiles, g_config.probe_num_samples);
                _l0_min_distance = static_cast<float>(result.table[0][0]);
            });
            ARTEA_INFO(fmt::format(
                "Auto-probed l0_min_distance = {:.6f}", _l0_min_distance));
        }
    }

    auto get_dataset()      -> vector_dataset_t& { return *_dataset; }
    auto get_dataset_info() const -> DatasetInfra { return _dataset_info; }
    auto get_l0_min_distance() const -> float          { return _l0_min_distance; }

private:
    DataProvider() = default;

    std::unique_ptr<vector_dataset_t> _dataset;
    DatasetInfra                      _dataset_info{};
    float                             _l0_min_distance = 0.0f;
};

// ============================================================
//  Fixture: build one graph using the workload's insertion settings.
// ============================================================

class StackedRGraphTest : public ::testing::Test {
protected:
    template <DistanceMetricsT Metric, vec_dim_t Dim>
    static auto build_graph()
        -> std::pair<std::unique_ptr<stacked_index_t>, int64_t>
    {
        auto& provider = DataProvider::instance();
        const auto& base_vecs = provider.get_dataset().get_base_vecs();
        // Stateless functor: the dimension is a compile-time trait now.
        dist_func_t<Metric, Dim> build_dist;

        const vertex_num_t total_vertices =
            static_cast<vertex_num_t>(base_vecs.get_num_vecs());
        stacked_rgraph::rgraph_config_t<Metric, Dim> rgraph_config(
            g_config.rnet_beta, g_config.tau_k, g_config.shifted_coeffs,
            provider.get_l0_min_distance(),
            static_cast<vertex_num_t>(g_config.search_nn_qs),
            static_cast<vertex_num_t>(g_config.ul_select_nbrs_qs),
            static_cast<vertex_num_t>(g_config.bl_select_nbrs_qs),
            g_config.ul_max_nbr_size,
            g_config.bl_max_nbr_size);
        ARTEA_INFO(fmt::format(
            "Building StackedRGraph: beta={:.3f}, tau_k={}, R0 = {:.6f}, R1 = {:.6f}, "
            "ul_max_nbr_size={}, bl_max_nbr_size={}, search_nn_qs={}, "
            "ul_select_nbrs_qs={}, bl_select_nbrs_qs={}, "
            "scale_coeffs={:.3f}",
            rgraph_config.rnet_beta(), rgraph_config.tau_k(),
            rgraph_config.radius_at(0), rgraph_config.radius_at(1),
            g_config.ul_max_nbr_size, g_config.bl_max_nbr_size,
            g_config.search_nn_qs,
            g_config.ul_select_nbrs_qs, g_config.bl_select_nbrs_qs,
            g_config.scale_coeffs));
        // stacked_rgraph::pruning_config_t is an alias of conv_graph's
        // PruningConfig; the stacked_rgraph backbone only reads
        // scale_coeffs from it, so shifted_coeffs is pinned to 0 here.
        stacked_rgraph::pruning_config_t<Metric, Dim> pruning_config(
            static_cast<ratio_t>(g_config.scale_coeffs),
            static_cast<ratio_t>(0));
        auto graph = std::make_unique<stacked_index_t>(
            total_vertices, rgraph_config, pruning_config);

        vector_array_t owned_batch =
            base_vecs.extract_subset(0, total_vertices);

        auto t0 = std::chrono::high_resolution_clock::now();
        stacked_rgraph::factory_t<Metric, Dim>::add_vertices(
            *graph, std::move(owned_batch), build_dist,
            g_config.insert_on_L0, g_config.shuffle_insertion_order);
        auto t1 = std::chrono::high_resolution_clock::now();
        const int64_t ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        return {std::move(graph), ms};
    }

    static void dump_graph_info(
        const dynamic::hierarchical_graph_t& graph,
        const char* label, int64_t build_ms)
    {
        const auto& base_vecs =
            DataProvider::instance().get_dataset().get_base_vecs();
        const layer_id_t top_level_id = graph.top_occupied_level_id();
        ARTEA_INFO(fmt::format(
            "[{}] built in {} ms: top_occupied_level={}, max_allowed_level_id={}",
            label, build_ms,
            (top_level_id == dynamic::hierarchical_graph_t
                ::invalid_level_id)
                ? -1 : static_cast<int>(top_level_id),
            graph.max_allowed_level_id()));
        // Preview what HierarchicalGraphCompactor would trim, using the
        // same rule it applies in compact_graph(): walk from the top
        // down and take the first bucket with >= min_layer_cap vids as
        // the new top. Anything above that gets demoted; the entry
        // point is the bucket-centroid argmin on (new_top bucket +
        // demoted vids).
        constexpr vertex_num_t min_cap =
            hierarchical_graph_compactor_t::min_layer_cap;
        const bool has_vertices = (top_level_id !=
            dynamic::hierarchical_graph_t::invalid_level_id);
        layer_id_t compactor_new_top = 0;
        if (has_vertices) {
            for (layer_id_t h = top_level_id; ; --h) {
                if (graph.get_vids_with_highest_level(h).size() >= min_cap) {
                    compactor_new_top = h;
                    break;
                }
                if (h == 0) { compactor_new_top = 0; break; }
            }
            ARTEA_INFO(fmt::format(
                "[{}] Compactor trim preview: min_layer_cap={}, new_top=L{} "
                "(entry-point = argmin dist-to-centroid over this bucket + demoted vids)",
                label, min_cap, compactor_new_top));
        }

        // A vertex with highest_level_id=h' participates in every level
        // 0..h', so the count *assigned to* level h is the cumulative
        // sum of bucket sizes from h to the top. Compute top-down.
        const layer_id_t max_level = graph.max_allowed_level_id();
        std::vector<vertex_num_t> num_at_level(max_level + 1, 0);
        {
            vertex_num_t running = 0;
            for (layer_id_t h = max_level; ; --h) {
                running += graph.get_vids_with_highest_level(h).size();
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

    static void SetUpTestSuite() {
        build_infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) {
            ARTEA_INFO(fmt::format("--- Building with insert_on_L0={} ---", g_config.insert_on_L0));
            std::tie(_graph, _build_ms) = build_graph<Metric, Dim>();
            const char* label = g_config.insert_on_L0 ? "insert_on_L0=true" : "insert_on_L0=false";
            dump_graph_info(_graph->get_hierarchical_graph(), label, _build_ms);
        });
    }

    static void TearDownTestSuite() {
        _graph.reset();
    }

    static std::unique_ptr<stacked_index_t> _graph;
    static int64_t _build_ms;
};

std::unique_ptr<stacked_index_t> StackedRGraphTest::_graph = nullptr;
int64_t StackedRGraphTest::_build_ms = 0;

// ============================================================
//  Build summary.
// ============================================================

TEST_F(StackedRGraphTest, BuildTime) {
    ASSERT_NE(_graph, nullptr);
    EXPECT_GE(_build_ms, 0);
    EXPECT_EQ(_graph->get_hierarchical_graph().get_num_vertices(),
              DataProvider::instance().get_dataset().get_base_vecs().get_num_vecs());

    ARTEA_INFO("=== Build summary ===");
    ARTEA_INFO(fmt::format("  dataset           : {}", g_config.dataset_name));
    ARTEA_INFO(fmt::format("  num_vertices      : {}", _graph->get_hierarchical_graph().get_num_vertices()));
    ARTEA_INFO(fmt::format("  max_allowed_level_id: {}",
        static_cast<int>(_graph->get_hierarchical_graph().max_allowed_level_id())));
    ARTEA_INFO(fmt::format("  insert_on_L0={}: {} ms", g_config.insert_on_L0, _build_ms));
}

// ============================================================
//  main: workload JSON + test suite runner
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_stacked_rgraph");
    program.add_description("Build a stacked r-net and report its level distribution from an Artea workload");
    program.add_argument("-w", "--workload")
        .required()
        .help("Path to workload JSON/JSONC; reads dataset-config, dataset, metric and indexes-config.artea");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program;
        return 1;
    }

    const auto workload_path = program.get<std::string>("--workload");
    try {
        g_config = load_workload(workload_path);
    } catch (const std::exception& err) {
        std::cerr << "Failed to load workload: " << workload_path << "\n" << err.what() << "\n";
        return 1;
    }

    std::cout << "\n=== Test Configuration ===\n";
    std::cout << "Workload:       " << workload_path << "\n";
    std::cout << "Dataset:        " << g_config.dataset_name   << "\n";
    std::cout << "Metric:         " << g_config.metric << "\n";
    std::cout << "rnet_beta:      " << g_config.rnet_beta      << "\n";
    std::cout << "tau_k:          " << g_config.tau_k << "\n";
    if (g_config.l0_min_distance_provided) {
        std::cout << "L0 minimum distance:      " << g_config.l0_min_distance
                  << " (from workload)\n";
    } else {
        std::cout << "L0 minimum distance:      auto-probe ("
                  << static_cast<int>(g_config.probe_quantile * 100.0f)
                  << "th pct, " << g_config.probe_num_samples
                  << " samples)\n";
    }
    std::cout << "ul_max_nbr_size:   " << g_config.ul_max_nbr_size   << "\n";
    std::cout << "bl_max_nbr_size:   " << g_config.bl_max_nbr_size   << "\n";
    std::cout << "search_nn_qs:      " << g_config.search_nn_qs      << "\n";
    std::cout << "ul_select_nbrs_qs: " << g_config.ul_select_nbrs_qs << "\n";
    std::cout << "bl_select_nbrs_qs: " << g_config.bl_select_nbrs_qs << "\n";
    std::cout << "scale_coeffs:      " << g_config.scale_coeffs      << "\n";
    std::cout << "insert_on_L0:   " << (g_config.insert_on_L0 ? "true" : "false") << "\n";
    std::cout << "shuffle_insertion_order: " << (g_config.shuffle_insertion_order ? "true" : "false") << "\n";
    std::cout << "============================\n\n";

    try {
        DataProvider::instance().init();
    } catch (const std::exception& err) {
        std::cerr << "Failed to initialize dataset: " << err.what() << "\n";
        return 1;
    }
    return RUN_ALL_TESTS();
}
