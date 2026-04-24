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
 * @FilePath: /Artea/vldb27-exp/adr_vs_ndc_profile.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Build an artea_graph index, compact it, then run two
 *               per-query 1-NN profilers on a random subset of queries:
 *                 - HGRouterProfiler on the full compact hierarchy
 *                 - SLRouterProfiler on L0 of the compact hierarchy
 *               Dumps raw per-query (NDC, ADR) trajectories to a JSON
 *               file for the companion Python plotter.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <argparse/argparse.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

// ============================================================
//  Config
// ============================================================

struct ExpConfig {
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
    uint32_t probe_num_samples;
    float    probe_quantile;

    // Refinement (refining max/reserved derived as rgraph max × 1.5)
    uint32_t num_build_loops;
    uint32_t num_triu_iters;
    float    prefill_ratio;
    uint32_t num_routing_loops;
    uint32_t routing_topk;
    uint32_t routing_queue_size;
    bool     shuffle_insertion_order;
    bool     insert_on_L0;

    // Query sampling
    uint32_t num_queries;
    uint32_t sample_seed;

    // Profiler
    uint32_t candidate_queue_size;

    // Output
    std::string output_json;
} g_config;

// ============================================================
//  Dataset loader (mirrors test_artea_graph)
// ============================================================

class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }
        ARTEA_INFO(fmt::format("Loading dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        _dataset = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);

        const auto& base_vecs = _dataset->get_base_vecs();
        ARTEA_INFO(fmt::format("Dataset: {} vectors, {} dims", base_vecs.get_num_vecs(), base_vecs.get_vec_dim()));

        _dist_func = std::make_unique<dist_func_t>(base_vecs.get_vec_dim());

        if (g_config.l0_radius_provided) {
            _l0_radius = g_config.l0_rnet_radius;
            ARTEA_INFO(fmt::format("Using user-provided L0 rnet_radius = {:.6f}", _l0_radius));
        } else {
            dataset_prober_t prober(base_vecs, *_dist_func);
            const std::vector<float> quantiles = { g_config.probe_quantile };
            ARTEA_INFO(fmt::format(
                "Probing L0 rnet_radius ({}th pct, {} samples)...",
                static_cast<int>(g_config.probe_quantile * 100.0f),
                g_config.probe_num_samples));
            auto result = prober.probe(quantiles, g_config.probe_num_samples);
            _l0_radius = static_cast<float>(result.table[0][0]);
            ARTEA_INFO(fmt::format("Auto-probed L0 rnet_radius = {:.6f}", _l0_radius));
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
//  Helpers
// ============================================================

auto sample_query_indices(vertex_num_t total, uint32_t num_to_sample, uint32_t seed) -> std::vector<vertex_num_t> {
    const vertex_num_t clamped = std::min<vertex_num_t>(num_to_sample, total);
    std::vector<vertex_num_t> all(total);
    std::iota(all.begin(), all.end(), vertex_num_t{0});
    std::mt19937 rng(seed);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(clamped);
    std::sort(all.begin(), all.end());   // sorted for friendlier logs / reproducible JSON
    return all;
}

auto trajectories_to_json(
    const Profile1NNResult&          result,
    const std::vector<vertex_num_t>& query_indices
) -> nlohmann::json {
    nlohmann::json trajs = nlohmann::json::array();
    for (std::size_t i = 0; i < result.trajectories.size(); ++i) {
        const auto& traj = result.trajectories[i];
        nlohmann::json points = nlohmann::json::array();
        for (const auto& [ndc, adr] : traj) {
            points.push_back({ {"ndc", ndc}, {"adr", adr} });
        }
        trajs.push_back({
            {"query_vid", query_indices[i]},
            {"points",    points}
        });
    }
    return nlohmann::json{
        {"num_sampled",  result.trajectories.size()},
        {"num_skipped",  result.num_skipped},
        {"trajectories", trajs}
    };
}

// ============================================================
//  Main
// ============================================================

int main(int argc, char** argv) {
    argparse::ArgumentParser program("adr_vs_ndc_profile");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));

    program.add_argument("--beta").default_value(2.0f).scan<'g', float>();
    program.add_argument("--l0-radius").default_value(-1.0f).scan<'g', float>();
    program.add_argument("--ul-max-nbr-size").default_value(32u).scan<'u', uint32_t>()
        .help("Per-vertex neighbor capacity at every upper layer.");
    program.add_argument("--bl-max-nbr-size").default_value(64u).scan<'u', uint32_t>()
        .help("Per-vertex neighbor capacity at the bottom layer (L0).");
    program.add_argument("--search-nn-qs").default_value(40u).scan<'u', uint32_t>();
    program.add_argument("--ul-select-nbrs-qs").default_value(100u).scan<'u', uint32_t>();
    program.add_argument("--bl-select-nbrs-qs").default_value(100u).scan<'u', uint32_t>();
    program.add_argument("--scale-coeffs").default_value(1.1f).scan<'g', float>();
    program.add_argument("--shifted-coeffs").default_value(0.0f).scan<'g', float>();
    program.add_argument("--perform-arc").default_value(false).implicit_value(true)
        .help("Enable ARC pruning sweep at end of refine_layer.");
    program.add_argument("--aspect-ratio-constraint").default_value(1.0f).scan<'g', float>()
        .help("Multiplier on radius_at(h) for the ARC threshold (consumed when --perform-arc).");
    program.add_argument("--probe-num-samples").default_value(500u).scan<'u', uint32_t>();
    program.add_argument("--probe-quantile").default_value(0.9f).scan<'g', float>();

    program.add_argument("--num-build-loops").default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters").default_value(14u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio").default_value(0.34f).scan<'g', float>();
    program.add_argument("--num-routing-loops").default_value(1u).scan<'u', uint32_t>();
    program.add_argument("--routing-topk").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--routing-queue-size").default_value(96u).scan<'u', uint32_t>();
    program.add_argument("--shuffle").default_value(false).implicit_value(true);
    program.add_argument("--insert-on-l0").default_value(false).implicit_value(true);

    program.add_argument("--num-queries").default_value(20u).scan<'u', uint32_t>()
        .help("Number of queries to randomly sample from the query set.");
    program.add_argument("--seed").default_value(42u).scan<'u', uint32_t>()
        .help("RNG seed for query sampling (for reproducibility).");
    program.add_argument("--candidate-queue-size").default_value(1u).scan<'u', uint32_t>()
        .help("Beam-search candidate queue size (SLRouterProfiler; L0 of HGRouterProfiler). "
              "1 = greedy single-cursor walk.");

    program.add_argument("-o", "--output")
        .default_value(std::string("./vldb27-exp/results/adr_vs_ndc_profile_results.json"))
        .help("Output JSON file path.");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program;
        return 1;
    }

    g_config.config_path             = program.get<std::string>("--config");
    g_config.dataset_name            = program.get<std::string>("--dataset");
    g_config.rnet_beta               = program.get<float>("--beta");
    const float l0_arg               = program.get<float>("--l0-radius");
    g_config.l0_radius_provided      = (l0_arg >= 0.0f);
    g_config.l0_rnet_radius          = l0_arg;
    g_config.ul_max_nbr_size         = program.get<uint32_t>("--ul-max-nbr-size");
    g_config.bl_max_nbr_size         = program.get<uint32_t>("--bl-max-nbr-size");
    g_config.search_nn_qs            = program.get<uint32_t>("--search-nn-qs");
    g_config.ul_select_nbrs_qs       = program.get<uint32_t>("--ul-select-nbrs-qs");
    g_config.bl_select_nbrs_qs       = program.get<uint32_t>("--bl-select-nbrs-qs");
    g_config.scale_coeffs            = program.get<float>("--scale-coeffs");
    g_config.shifted_coeffs          = program.get<float>("--shifted-coeffs");
    g_config.perform_arc             = program.get<bool>("--perform-arc");
    g_config.aspect_ratio_constraint = program.get<float>("--aspect-ratio-constraint");
    g_config.probe_num_samples       = program.get<uint32_t>("--probe-num-samples");
    g_config.probe_quantile          = program.get<float>("--probe-quantile");
    g_config.num_build_loops         = program.get<uint32_t>("--num-build-loops");
    g_config.num_triu_iters          = program.get<uint32_t>("--num-triu-iters");
    g_config.prefill_ratio           = program.get<float>("--prefill-ratio");
    g_config.num_routing_loops       = program.get<uint32_t>("--num-routing-loops");
    g_config.routing_topk            = program.get<uint32_t>("--routing-topk");
    g_config.routing_queue_size      = program.get<uint32_t>("--routing-queue-size");
    g_config.shuffle_insertion_order = program.get<bool>("--shuffle");
    g_config.insert_on_L0            = program.get<bool>("--insert-on-l0");
    g_config.num_queries             = program.get<uint32_t>("--num-queries");
    g_config.sample_seed             = program.get<uint32_t>("--seed");
    g_config.candidate_queue_size    = program.get<uint32_t>("--candidate-queue-size");
    g_config.output_json             = program.get<std::string>("--output");

    DataProvider::instance().init();

    auto& provider = DataProvider::instance();
    auto& dataset  = provider.get_dataset();
    const auto& base_vecs  = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt_vecs    = dataset.get_gt_vecs();
    auto& dist_func        = provider.get_dist_func();

    const vertex_num_t total_vertices = static_cast<vertex_num_t>(base_vecs.get_num_vecs());
    const vertex_num_t total_queries  = static_cast<vertex_num_t>(query_vecs.get_num_vecs());

    // ---- Build artea_graph ----
    artea_graph::rgraph_config_t rgraph_config(
        g_config.rnet_beta,
        provider.get_l0_radius(),
        static_cast<vertex_num_t>(g_config.search_nn_qs),
        static_cast<vertex_num_t>(g_config.ul_select_nbrs_qs),
        static_cast<vertex_num_t>(g_config.bl_select_nbrs_qs),
        g_config.ul_max_nbr_size,
        g_config.bl_max_nbr_size);

    artea_graph::propagate_config_t propagate_config(
        static_cast<iter_t>(g_config.num_build_loops),
        static_cast<iter_t>(g_config.num_triu_iters),
        static_cast<ratio_t>(g_config.prefill_ratio),
        static_cast<iter_t>(g_config.num_routing_loops),
        static_cast<vertex_num_t>(g_config.routing_topk),
        static_cast<vertex_num_t>(g_config.routing_queue_size));

    artea_graph::pruning_config_t pruning_config(
        static_cast<ratio_t>(g_config.scale_coeffs),
        static_cast<ratio_t>(g_config.shifted_coeffs),
        g_config.perform_arc,
        static_cast<ratio_t>(g_config.aspect_ratio_constraint));

    auto graph = std::make_unique<artea_graph::index_t>(
        total_vertices, rgraph_config, propagate_config, pruning_config);

    ARTEA_INFO("Building artea_graph...");
    auto build_t0 = std::chrono::high_resolution_clock::now();

    vector_array_t owned_batch = base_vecs.extract_subset(0, total_vertices);
    artea_graph::factory_t::add_vertices(
        *graph, std::move(owned_batch), dist_func,
        g_config.insert_on_L0, g_config.shuffle_insertion_order);

    auto build_t1 = std::chrono::high_resolution_clock::now();
    const int64_t build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_t1 - build_t0).count();
    ARTEA_INFO(fmt::format("artea_graph built in {} ms", build_ms));

    // ---- Compact the hierarchical graph ----
    ARTEA_INFO("Compacting hierarchical graph...");
    auto compact_t0 = std::chrono::high_resolution_clock::now();
    auto compact_hg = hierarchical_graph_compactor_t::compact_graph(
        graph->get_hierarchical_graph(), base_vecs, dist_func);
    auto compact_t1 = std::chrono::high_resolution_clock::now();
    const int64_t compact_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compact_t1 - compact_t0).count();
    ARTEA_INFO(fmt::format("Compaction done in {} ms", compact_ms));

    const vertex_id_t entry_point_vid = compact_hg.entry_point_vid();
    ARTEA_INFO(fmt::format("entry_point_vid = {}", entry_point_vid));

    // ---- Sample queries ----
    const std::vector<vertex_num_t> query_indices =
        sample_query_indices(total_queries, g_config.num_queries, g_config.sample_seed);
    ARTEA_INFO(fmt::format(
        "Sampled {} queries (out of {}) with seed {}",
        query_indices.size(), total_queries, g_config.sample_seed));

    // ---- Curve 1: full hierarchical (HGRouterProfiler) ----
    ARTEA_INFO(fmt::format(
        "Profiling hierarchical artea (HGRouterProfiler, L0 beam_size={})...",
        g_config.candidate_queue_size));
    hg_router_profiler_t hg_profiler(base_vecs, dist_func, g_config.candidate_queue_size);
    hg_profiler.initialize();
    auto hg_t0 = std::chrono::high_resolution_clock::now();
    Profile1NNResult hg_result =
        hg_profiler.profile_adr_vs_ndc(query_vecs, compact_hg, gt_vecs, entry_point_vid, query_indices);
    auto hg_t1 = std::chrono::high_resolution_clock::now();
    const int64_t hg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(hg_t1 - hg_t0).count();
    ARTEA_INFO(fmt::format(
        "Hierarchical profile: {} trajectories ({} skipped), {} ms",
        hg_result.trajectories.size(), hg_result.num_skipped, hg_ms));

    // ---- Curve 2: L0-only (SLRouterProfiler on compact_hg level 0) ----
    ARTEA_INFO(fmt::format(
        "Profiling L0-only (SLRouterProfiler, beam_size={})...",
        g_config.candidate_queue_size));
    sl_router_profiler_t sl_profiler(base_vecs, dist_func, g_config.candidate_queue_size);
    sl_profiler.initialize();
    auto sl_t0 = std::chrono::high_resolution_clock::now();
    Profile1NNResult sl_result = sl_profiler.profile_adr_vs_ndc(
        query_vecs, compact_hg, layer_id_t{0}, gt_vecs, entry_point_vid, query_indices);
    auto sl_t1 = std::chrono::high_resolution_clock::now();
    const int64_t sl_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sl_t1 - sl_t0).count();
    ARTEA_INFO(fmt::format(
        "L0-only profile: {} trajectories ({} skipped), {} ms",
        sl_result.trajectories.size(), sl_result.num_skipped, sl_ms));

    // ---- Write JSON ----
    nlohmann::json out;
    out["dataset"]              = g_config.dataset_name;
    out["num_base_vectors"]     = base_vecs.get_num_vecs();
    out["total_queries"]        = total_queries;
    out["num_sampled"]          = query_indices.size();
    out["sample_seed"]          = g_config.sample_seed;
    out["candidate_queue_size"] = g_config.candidate_queue_size;
    out["entry_point_vid"]      = entry_point_vid;
    out["build_ms"]             = build_ms;
    out["compact_ms"]           = compact_ms;
    out["config"] = {
        {"rnet_beta",             g_config.rnet_beta},
        {"l0_rnet_radius",        provider.get_l0_radius()},
        {"ul_max_nbr_size",       g_config.ul_max_nbr_size},
        {"bl_max_nbr_size",       g_config.bl_max_nbr_size},
        {"ul_select_nbrs_qs",     g_config.ul_select_nbrs_qs},
        {"bl_select_nbrs_qs",     g_config.bl_select_nbrs_qs},
        {"scale_coeffs",          g_config.scale_coeffs},
        {"shifted_coeffs",        g_config.shifted_coeffs},
        {"perform_arc",           g_config.perform_arc},
        {"aspect_ratio_constraint", g_config.aspect_ratio_constraint},
        {"num_build_loops",       g_config.num_build_loops},
        {"num_triu_iters",        g_config.num_triu_iters},
        {"prefill_ratio",         g_config.prefill_ratio},
        {"num_routing_loops",     g_config.num_routing_loops},
        {"routing_topk",          g_config.routing_topk},
        {"routing_queue_size",    g_config.routing_queue_size},
        {"insert_on_L0",          g_config.insert_on_L0}
    };
    out["curves"] = {
        {"hierarchical_artea", trajectories_to_json(hg_result, query_indices)},
        {"l0_only",            trajectories_to_json(sl_result, query_indices)}
    };

    const std::filesystem::path out_path(g_config.output_json);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    std::ofstream ofs(out_path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << g_config.output_json << "\n";
        return 1;
    }
    ofs << out.dump(2);
    ofs.close();

    ARTEA_INFO(fmt::format("Results written to {}", g_config.output_json));
    return 0;
}
