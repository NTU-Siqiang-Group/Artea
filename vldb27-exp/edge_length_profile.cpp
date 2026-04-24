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
 * @FilePath: /Artea/vldb27-exp/edge_length_profile.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Build an artea_graph index, compact it, then collect
 *               the distribution of edge distances adopted by:
 *                 - HGRouterProfiler: all layers of the hierarchy
 *                 - SLRouterProfiler: L0 of the compact hierarchy
 *               Emits per-layer distance samples to JSON for the
 *               companion plotter (optionally reservoir-sampled to
 *               keep output size bounded).
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
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

struct ExpConfig {
    std::string config_path;
    std::string dataset_name;

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

    uint32_t num_build_loops;
    uint32_t num_triu_iters;
    float    prefill_ratio;
    uint32_t num_routing_loops;
    uint32_t routing_topk;
    uint32_t routing_queue_size;
    bool     shuffle_insertion_order;
    bool     insert_on_L0;

    uint32_t candidate_queue_size;
    uint32_t samples_per_layer_cap;
    uint32_t sample_seed;

    std::string output_json;
} g_config;

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

// Randomly subsample @p src down to @p cap entries (in-place permutation
// + truncate). Deterministic for a given @p seed.
auto subsample(std::vector<double>& src, std::size_t cap, std::mt19937& rng) -> void {
    if (src.size() <= cap || cap == 0) return;
    // Partial Fisher–Yates: shuffle enough to fix the first `cap` slots.
    for (std::size_t i = 0; i < cap; ++i) {
        std::uniform_int_distribution<std::size_t> dist(i, src.size() - 1);
        std::swap(src[i], src[dist(rng)]);
    }
    src.resize(cap);
}

auto edges_to_json(
    const EdgeLengthProfileResult& r,
    std::size_t                    cap,
    std::mt19937&                  rng,
    const std::vector<double>&     radii
) -> nlohmann::json {
    nlohmann::json by_level = nlohmann::json::array();
    for (std::size_t l = 0; l < r.edges_by_level.size(); ++l) {
        std::vector<double> bucket = r.edges_by_level[l];
        const std::size_t original_size = bucket.size();
        subsample(bucket, cap, rng);
        nlohmann::json level_json = {
            {"total",   original_size},
            {"sampled", bucket.size()},
            {"lengths", bucket}
        };
        if (l < radii.size()) {
            level_json["radius"] = radii[l];
        }
        by_level.push_back(std::move(level_json));
    }
    return nlohmann::json{
        {"num_queries",   r.num_queries},
        {"edges_by_level", by_level}
    };
}

// Per-level bucket of edge-weight samples pulled from the dynamic
// hierarchical graph before compaction. `total` is the exact count of
// valid neighbor entries observed; `samples` is a reservoir of at most
// @p cap_per_level of those entries.
struct NbrDistLayer {
    std::vector<double> samples;
    std::uint64_t       total  = 0;
    double              radius = 0.0;
};

// Iterate every (v, nbr, d_v_nbr) triple stored in the dynamic
// HierarchicalGraph at every level and reservoir-sample the distances
// into per-level buckets. Must run BEFORE @c compact_graph, since the
// compact graph strips edge distances.
//
// Algorithm R (Vitter) per bucket — O(total) time, O(cap) memory, with
// each observed distance having equal probability of ending up in the
// reservoir.
template <typename HierGraphT>
auto collect_nbr_distances(
    const HierGraphT&          hg,
    const vertex_num_t         total_vertices,
    const layer_id_t           top_level,
    const std::size_t          cap_per_level,
    const std::vector<double>& radii,
    std::mt19937&              rng
) -> std::vector<NbrDistLayer> {
    std::vector<NbrDistLayer> result(static_cast<std::size_t>(top_level) + 1);
    for (std::size_t l = 0; l < result.size(); ++l) {
        result[l].samples.reserve(std::min<std::size_t>(cap_per_level, std::size_t{1'000'000}));
        if (l < radii.size()) result[l].radius = radii[l];
    }

    constexpr auto unassigned = HierGraphT::unassigned_highest_level_id;

    for (vertex_id_t v = 0; v < total_vertices; ++v) {
        const layer_id_t highest = hg.get_highest_level_id(v);
        if (highest == unassigned) continue;
        const layer_id_t h_max = std::min(highest, top_level);
        for (layer_id_t h = 0; h <= h_max; ++h) {
            auto nbrs = hg.fetch_layer_nbrs(v, h);
            auto& layer = result[h];
            for (const auto& nbr : nbrs) {
                if (nbr.is_invalid()) continue;
                const double d = static_cast<double>(nbr.get_distance());
                if (layer.samples.size() < cap_per_level) {
                    layer.samples.push_back(d);
                } else {
                    std::uniform_int_distribution<std::uint64_t> dist(0, layer.total);
                    const std::uint64_t j = dist(rng);
                    if (j < cap_per_level) layer.samples[j] = d;
                }
                ++layer.total;
            }
        }
    }
    return result;
}

auto nbr_dists_to_json(const std::vector<NbrDistLayer>& by_level) -> nlohmann::json {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& layer : by_level) {
        nlohmann::json level_json = {
            {"total",   layer.total},
            {"sampled", layer.samples.size()},
            {"lengths", layer.samples}
        };
        if (layer.radius > 0.0) level_json["radius"] = layer.radius;
        arr.push_back(std::move(level_json));
    }
    return arr;
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("edge_length_profile");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));

    program.add_argument("--beta").default_value(2.0f).scan<'g', float>();
    program.add_argument("--l0-radius").default_value(-1.0f).scan<'g', float>();
    program.add_argument("--ul-max-nbr-size").default_value(32u).scan<'u', uint32_t>();
    program.add_argument("--bl-max-nbr-size").default_value(64u).scan<'u', uint32_t>();
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

    program.add_argument("--candidate-queue-size").default_value(1u).scan<'u', uint32_t>()
        .help("Beam-search candidate queue size for both profilers.");
    program.add_argument("--samples-per-layer-cap").default_value(100000u).scan<'u', uint32_t>()
        .help("Cap on the number of edge distances serialized per layer (random subsample). 0 = no cap.");
    program.add_argument("--seed").default_value(42u).scan<'u', uint32_t>()
        .help("RNG seed for subsampling (for reproducibility).");

    program.add_argument("-o", "--output")
        .default_value(std::string("./vldb27-exp/results/edge_length_profile_results.json"))
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
    g_config.candidate_queue_size    = program.get<uint32_t>("--candidate-queue-size");
    g_config.samples_per_layer_cap   = program.get<uint32_t>("--samples-per-layer-cap");
    g_config.sample_seed             = program.get<uint32_t>("--seed");
    g_config.output_json             = program.get<std::string>("--output");

    DataProvider::instance().init();

    auto& provider = DataProvider::instance();
    auto& dataset  = provider.get_dataset();
    const auto& base_vecs  = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    auto& dist_func        = provider.get_dist_func();

    const vertex_num_t total_vertices = static_cast<vertex_num_t>(base_vecs.get_num_vecs());

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

    // ---- Pre-compaction: harvest every (v, nbr, d) triple. Compaction
    //      drops edge distances, so this must run first. -----
    std::vector<NbrDistLayer> nbr_dist_layers;
    const std::size_t nbr_cap =
        (g_config.samples_per_layer_cap == 0)
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(g_config.samples_per_layer_cap);
    {
        const auto& dyn_hg = graph->get_hierarchical_graph();
        const layer_id_t top_level = graph->top_occupied_level_id();
        constexpr auto unassigned = dynamic::hierarchical_graph_t::unassigned_highest_level_id;
        if (top_level != unassigned) {
            std::vector<double> dyn_radii;
            dyn_radii.reserve(static_cast<std::size_t>(top_level) + 1);
            for (layer_id_t h = 0; h <= top_level; ++h) {
                dyn_radii.push_back(static_cast<double>(rgraph_config.radius_at(h)));
            }

            ARTEA_INFO("Collecting per-level neighbor distances (pre-compaction)...");
            std::mt19937 nbr_rng(g_config.sample_seed + 1u);   // distinct stream from edges_to_json rng
            auto nbr_t0 = std::chrono::high_resolution_clock::now();
            nbr_dist_layers = collect_nbr_distances(
                dyn_hg, total_vertices, top_level,
                nbr_cap, dyn_radii, nbr_rng);
            auto nbr_t1 = std::chrono::high_resolution_clock::now();
            const int64_t nbr_ms = std::chrono::duration_cast<std::chrono::milliseconds>(nbr_t1 - nbr_t0).count();
            for (std::size_t l = 0; l < nbr_dist_layers.size(); ++l) {
                ARTEA_INFO(fmt::format(
                    "  L{} nbrs: total={}, sampled={}",
                    l, nbr_dist_layers[l].total, nbr_dist_layers[l].samples.size()));
            }
            ARTEA_INFO(fmt::format("Neighbor-distance collection done in {} ms", nbr_ms));
        } else {
            ARTEA_INFO("Skipping neighbor-distance collection: no vertices assigned.");
        }
    }

    ARTEA_INFO("Compacting hierarchical graph...");
    auto compact_t0 = std::chrono::high_resolution_clock::now();
    auto compact_hg = hierarchical_graph_compactor_t::compact_graph(
        graph->get_hierarchical_graph(), base_vecs, dist_func);
    auto compact_t1 = std::chrono::high_resolution_clock::now();
    const int64_t compact_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compact_t1 - compact_t0).count();
    ARTEA_INFO(fmt::format("Compaction done in {} ms", compact_ms));

    const vertex_id_t entry_point_vid = compact_hg.entry_point_vid();
    ARTEA_INFO(fmt::format("entry_point_vid = {}", entry_point_vid));

    // ---- Curve 1: Hierarchical edges by layer ----
    ARTEA_INFO(fmt::format(
        "Collecting hierarchical edge lengths (HGRouterProfiler, L0 beam_size={})...",
        g_config.candidate_queue_size));
    hg_router_profiler_t hg_profiler(base_vecs, dist_func, g_config.candidate_queue_size);
    hg_profiler.initialize();
    EdgeLengthProfileResult hg_result =
        hg_profiler.profile_edge_length(query_vecs, compact_hg, entry_point_vid);
    for (std::size_t l = 0; l < hg_result.edges_by_level.size(); ++l) {
        ARTEA_INFO(fmt::format(
            "  HG layer L{}: {} adopted edges", l, hg_result.edges_by_level[l].size()));
    }

    // ---- Curve 2: L0-only edges ----
    ARTEA_INFO(fmt::format(
        "Collecting L0-only edge lengths (SLRouterProfiler, beam_size={})...",
        g_config.candidate_queue_size));
    sl_router_profiler_t sl_profiler(base_vecs, dist_func, g_config.candidate_queue_size);
    sl_profiler.initialize();
    EdgeLengthProfileResult sl_result = sl_profiler.profile_edge_length(
        query_vecs, compact_hg, layer_id_t{0}, entry_point_vid);
    ARTEA_INFO(fmt::format(
        "  L0-only: {} adopted edges", sl_result.edges_by_level[0].size()));

    std::mt19937 rng(g_config.sample_seed);
    const std::size_t cap =
        (g_config.samples_per_layer_cap == 0)
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(g_config.samples_per_layer_cap);

    nlohmann::json out;
    out["dataset"]              = g_config.dataset_name;
    out["num_base_vectors"]     = base_vecs.get_num_vecs();
    out["num_queries"]          = query_vecs.get_num_vecs();
    out["candidate_queue_size"] = g_config.candidate_queue_size;
    out["samples_per_layer_cap"] = g_config.samples_per_layer_cap;
    out["entry_point_vid"]      = entry_point_vid;
    out["build_ms"]             = build_ms;
    out["compact_ms"]           = compact_ms;
    // Per-layer r-net radius: R_h = L0 * beta^h. Attached to each level
    // bucket so the plotter can annotate the covering radius.
    std::vector<double> hg_radii;
    hg_radii.reserve(hg_result.edges_by_level.size());
    for (std::size_t l = 0; l < hg_result.edges_by_level.size(); ++l) {
        hg_radii.push_back(static_cast<double>(
            rgraph_config.radius_at(static_cast<layer_id_t>(l))));
    }
    const std::vector<double> sl_radii = {
        static_cast<double>(rgraph_config.radius_at(layer_id_t{0}))
    };

    auto hier_curve_json = edges_to_json(hg_result, cap, rng, hg_radii);
    // Splice the pre-compaction all-edges distribution under the same
    // hier curve object — the plotter reads both to overlay them.
    hier_curve_json["nbr_dists_by_level"] = nbr_dists_to_json(nbr_dist_layers);
    out["curves"] = {
        {"hierarchical_artea", hier_curve_json},
        {"l0_only",            edges_to_json(sl_result, cap, rng, sl_radii)}
    };

    const std::filesystem::path out_path(g_config.output_json);
    if (out_path.has_parent_path()) std::filesystem::create_directories(out_path.parent_path());
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
