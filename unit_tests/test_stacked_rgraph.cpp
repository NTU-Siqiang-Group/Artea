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
 * @Description: Tests for StackedRGraph (dynamic Stacked R-Nets) driven by a
 *               real dataset (e.g. SIFT). L1 radius can be provided via CLI
 *               or probed via DatasetProber (90% quantile of 1-NN distance).
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include <argparse/argparse.hpp>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

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
    float rnet_beta;                  // layer radius growth factor
    bool  l1_radius_provided;         // if false, auto-probe
    float l1_rnet_radius;             // only used if l1_radius_provided
    uint32_t max_nbr_size;

    // DatasetProber parameters (when L1 radius is auto-probed)
    uint32_t probe_num_samples;
    float    probe_quantile;          // which quantile of 1-NN distance to use

    // Beam-search queue sizes (Phase 1 descent / Phase 2 candidate gathering)
    uint32_t search_nn_qs;
    uint32_t select_nbrs_qs;

    // Coverage/separation sampling parameters
    uint32_t coverage_num_samples;    // # base points sampled per layer
    uint32_t separation_num_samples;  // # layer-h vertices sampled per layer

    bool verbose;
} g_config;

// ============================================================
//  DataProvider singleton: loads dataset and (optionally) probes
//  the L1 radius exactly once for the whole test binary.
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
        _dataset = std::make_unique<vector_dataset_t>(
            g_config.config_path, g_config.dataset_name);

        const auto& base_vecs = _dataset->get_base_vecs();
        ARTEA_INFO(fmt::format("Dataset loaded: {} vectors, {} dims",
            base_vecs.get_num_vecs(), base_vecs.get_vec_dim()));

        _dist_func = std::make_unique<dist_func_t>(base_vecs.get_vec_dim());

        // Resolve the L1 radius: user-provided overrides auto-probe.
        if (g_config.l1_radius_provided) {
            _l1_radius = g_config.l1_rnet_radius;
            ARTEA_INFO(fmt::format(
                "Using user-provided L1 rnet_radius = {:.6f}", _l1_radius));
        } else {
            _l1_radius = _probe_l1_radius(base_vecs, *_dist_func);
            ARTEA_INFO(fmt::format(
                "Auto-probed L1 rnet_radius = {:.6f} "
                "({}th-percentile 1-NN distance, num_samples={})",
                _l1_radius,
                static_cast<int>(g_config.probe_quantile * 100.0f),
                g_config.probe_num_samples));
        }
    }

    auto get_dataset() -> vector_dataset_t& { return *_dataset; }
    auto get_dist_func() -> dist_func_t& { return *_dist_func; }
    auto get_l1_radius() const -> float { return _l1_radius; }

private:
    DataProvider() = default;

    /**
     * @brief Probe the dataset for its @c g_config.probe_quantile -th
     *        percentile of 1-NN distances. Uses @c num_samples samples.
     *        Returns that distance as the L1 rnet_radius.
     */
    auto _probe_l1_radius(const vector_array_t& base_vecs,
                          const dist_func_t& dist_func) -> float {
        dataset_prober_t prober(base_vecs, dist_func);
        const std::vector<float> quantiles = { g_config.probe_quantile };

        ARTEA_INFO(fmt::format(
            "Probing dataset for 1-NN {}th-percentile (num_samples={}) ...",
            static_cast<int>(g_config.probe_quantile * 100.0f),
            g_config.probe_num_samples));

        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = prober.probe(quantiles, g_config.probe_num_samples);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed_s =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        ARTEA_INFO(fmt::format("Dataset probe completed in {:.2f} s", elapsed_s));

        // table[nn_rank_idx][quantile_idx]. nn_rank 1 lives at index 0.
        // quantile 0 (the only one we asked for) is the radius we want.
        const distance_t r = result.table[0][0];
        return static_cast<float>(r);
    }

    std::unique_ptr<vector_dataset_t> _dataset;
    std::unique_ptr<dist_func_t>      _dist_func;
    float                             _l1_radius = 0.0f;
};

// ============================================================
//  Prune / evict functors used by all test cases
// ============================================================

namespace {

/**
 * @brief Simple FIFO pruning functor: shift-left and drop the oldest
 *        neighbor, insert @p new_nbr at the tail. Matches the pattern used
 *        in the existing InternalGraph tests.
 */
struct fifo_pruning_fn_t {
    vertex_num_t max_nbr_size;
    auto operator()(lnbr_t* slots, const lnbr_t new_nbr) const -> uint64_t {
        for (vertex_num_t i = 1; i < max_nbr_size; ++i) slots[i - 1] = slots[i];
        slots[max_nbr_size - 1] = new_nbr;
        return max_nbr_size;
    }
};

/**
 * @brief Compute R_h = L1_radius * beta^(h-1) (h is 1-indexed per paper).
 */
inline auto radius_at_layer(const float l1_radius, const float beta,
                            const layer_num_t h) -> distance_t {
    float r = l1_radius;
    for (layer_num_t i = 1; i < h; ++i) r *= beta;
    return static_cast<distance_t>(r);
}

}  // anonymous namespace

// ============================================================
//  Test fixture: all tests share the single constructed graph
//  built in SetUp, so that we do the (expensive) build only once.
// ============================================================

class StackedRGraphTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        auto& provider = DataProvider::instance();
        const auto& base_vecs = provider.get_dataset().get_base_vecs();
        auto& dist_func = provider.get_dist_func();

        const float l1_radius = provider.get_l1_radius();
        const float beta = g_config.rnet_beta;

        ARTEA_INFO(fmt::format(
            "Building StackedRGraph: beta={:.3f}, L1_radius={:.6f}, max_nbr={}, "
            "search_nn_qs={}, select_nbrs_qs={}",
            beta, l1_radius, g_config.max_nbr_size,
            g_config.search_nn_qs, g_config.select_nbrs_qs));

        auto t0 = std::chrono::high_resolution_clock::now();

        // Build via the incremental interface only:
        //   1. Construct an empty IndexStructure with the chosen knobs.
        //   2. Produce an owned clone of the caller's base_vecs (we hold
        //      it by const& from the DataProvider, so we cannot move it).
        //   3. Hand the clone to add_vertices by move; inside
        //      add_vertices the clone is moved into the index's
        //      (currently empty) owned base storage via append_batch's
        //      zero-copy fast path.
        const vertex_num_t total_vertices =
            static_cast<vertex_num_t>(base_vecs.get_num_vecs());
        _graph = std::make_unique<stacked_rgraph::index_t>(
            total_vertices,
            /*rnet_beta=*/beta,
            /*L1_rnet_radius=*/l1_radius,
            /*search_nn_qs=*/static_cast<vertex_num_t>(g_config.search_nn_qs),
            /*select_nbrs_qs=*/static_cast<vertex_num_t>(g_config.select_nbrs_qs),
            /*max_nbr_size=*/g_config.max_nbr_size);

        vector_array_t owned_batch = base_vecs.extract_subset(0, total_vertices);
        stacked_rgraph::factory_t::add_vertices(
            *_graph,
            std::move(owned_batch),
            dist_func,
            fifo_pruning_fn_t{static_cast<vertex_num_t>(g_config.max_nbr_size)});

        auto t1 = std::chrono::high_resolution_clock::now();
        _build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        ARTEA_INFO(fmt::format(
            "StackedRGraph built in {} ms: {} layers (max_restrict_level={})",
            _build_ms, _graph->get_num_layers(), _graph->max_restrict_level()));

        // Per-layer summary.
        for (layer_id_t l = 0; l < _graph->get_num_layers(); ++l) {
            const auto& layer = _graph->get_layer_graph(l);
            const float ratio = 100.0f * layer.get_num_vertices() / base_vecs.get_num_vecs();
            ARTEA_INFO(fmt::format(
                "  Layer {}: {} vertices ({:.2f}% of base), R_{} = {:.6f}",
                l + 1, layer.get_num_vertices(), ratio, l + 1,
                radius_at_layer(l1_radius, g_config.rnet_beta,
                                static_cast<layer_num_t>(l + 1))));
        }
    }

    static void TearDownTestSuite() {
        _graph.reset();
    }

    static std::unique_ptr<stacked_rgraph::index_t> _graph;
    static int64_t                                  _build_ms;
};

std::unique_ptr<stacked_rgraph::index_t> StackedRGraphTest::_graph = nullptr;
int64_t                                  StackedRGraphTest::_build_ms = 0;

// ============================================================
//  Test cases
// ============================================================

TEST_F(StackedRGraphTest, HierarchyNonEmpty) {
    ASSERT_NE(_graph, nullptr);
    EXPECT_GE(_graph->get_num_layers(), 1u);

    const auto& base_vecs = DataProvider::instance().get_dataset().get_base_vecs();
    const auto& layer0 = _graph->get_layer_graph(0);
    EXPECT_GT(layer0.get_num_vertices(), 0u);
    EXPECT_LE(layer0.get_num_vertices(), base_vecs.get_num_vecs());
}

TEST_F(StackedRGraphTest, LayerSizesShrinkUpward) {
    // Upper layers should be strictly smaller (or at worst equal) to lower
    // ones. Strict inequality is expected once we move past layer 0.
    const auto num_layers = _graph->get_num_layers();
    for (layer_id_t l = 1; l < num_layers; ++l) {
        const auto& lower = _graph->get_layer_graph(l - 1);
        const auto& upper = _graph->get_layer_graph(l);
        EXPECT_LE(upper.get_num_vertices(), lower.get_num_vertices())
            << "Layer " << l << " (" << upper.get_num_vertices()
            << ") should be <= layer " << (l - 1)
            << " (" << lower.get_num_vertices() << ")";
    }
}

TEST_F(StackedRGraphTest, InterLayerLinksAreValid) {
    const auto& base_vecs = DataProvider::instance().get_dataset().get_base_vecs();
    const auto n = base_vecs.get_num_vecs();
    const auto num_layers = _graph->get_num_layers();

    // Layer 0 (L1 in paper terms): inter_layer_link[layer_vid] is a base_vid.
    const auto& layer0 = _graph->get_layer_graph(0);
    for (vertex_num_t v = 0; v < layer0.get_num_vertices(); ++v) {
        const vertex_id_t link = layer0.get_inter_layer_link(v);
        ASSERT_LT(link, n) << "Layer 0 vertex " << v
                           << " has OOB link " << link;
    }

    // Upper layers: inter_layer_link[layer_vid] is a layer_vid in the layer below.
    for (layer_id_t l = 1; l < num_layers; ++l) {
        const auto& upper = _graph->get_layer_graph(l);
        const auto& lower = _graph->get_layer_graph(l - 1);
        for (vertex_num_t v = 0; v < upper.get_num_vertices(); ++v) {
            const vertex_id_t link = upper.get_inter_layer_link(v);
            ASSERT_LT(link, lower.get_num_vertices())
                << "Layer " << l << " vertex " << v
                << " has OOB link " << link;
        }
    }
}

TEST_F(StackedRGraphTest, NeighborListsAreValid) {
    const auto& base_vecs = DataProvider::instance().get_dataset().get_base_vecs();
    const auto n = base_vecs.get_num_vecs();
    const auto num_layers = _graph->get_num_layers();

    // Sample-based verification: for large datasets, verifying every vertex's
    // neighbor list is too slow. Check a uniform sample per layer.
    constexpr vertex_num_t max_per_layer = 5000;

    for (layer_id_t l = 0; l < num_layers; ++l) {
        const auto& layer = _graph->get_layer_graph(l);
        const vertex_num_t n_l = layer.get_num_vertices();
        const vertex_num_t step = std::max<vertex_num_t>(1, n_l / max_per_layer);

        for (vertex_num_t v = 0; v < n_l; v += step) {
            const auto block = layer.fetch_nbrs(v);
            const uint64_t count = layer.num_valid_nbrs(v);
            ASSERT_LE(count, layer.max_nbr_size())
                << "Layer " << l << " vertex " << v
                << " has count " << count
                << " > max_nbr_size " << layer.max_nbr_size();
            for (uint64_t i = 0; i < count; ++i) {
                const lnbr_t& nbr = block[1 + i];
                if (nbr == base_traits_t::invalid_lnbr) continue;
                ASSERT_LT(nbr.base_vid, n)
                    << "Layer " << l << " vertex " << v
                    << " slot " << i << " has OOB base_vid " << nbr.base_vid;
                ASSERT_LT(nbr.layer_vid, n_l)
                    << "Layer " << l << " vertex " << v
                    << " slot " << i << " has OOB layer_vid " << nbr.layer_vid;
            }
        }
    }
}

// ============================================================
//  Coverage rate: for each layer h, what fraction of sampled base
//  points have brute-force nearest-neighbor distance <= R_h?
//
//  An ideal r-net has coverage = 100% at every layer. The dynamic
//  insertion algorithm is approximate, so we expect high coverage at
//  the lower (denser) layers and somewhat lower at the topmost layers
//  (where vertices are few and data is sparse).
// ============================================================

TEST_F(StackedRGraphTest, CoverageRate) {
    const auto& base_vecs = DataProvider::instance().get_dataset().get_base_vecs();
    auto& dist_func = DataProvider::instance().get_dist_func();
    const auto num_layers = _graph->get_num_layers();
    const auto n = base_vecs.get_num_vecs();

    const vertex_num_t num_samples = std::min<vertex_num_t>(
        g_config.coverage_num_samples, n);

    // Deterministic sample: even strides through the base dataset.
    const vertex_num_t step = std::max<vertex_num_t>(1, n / num_samples);

    ARTEA_INFO(fmt::format("--- Coverage rate (num_samples={}) ---", num_samples));

    for (layer_id_t l = 0; l < num_layers; ++l) {
        const auto& layer = _graph->get_layer_graph(l);
        const vertex_num_t n_l = layer.get_num_vertices();
        if (n_l == 0) continue;

        const distance_t R_h = static_cast<distance_t>(
            radius_at_layer(DataProvider::instance().get_l1_radius(),
                            g_config.rnet_beta,
                            static_cast<layer_num_t>(l + 1)));

        // Collect the base_vids present in layer l via the InternalGraph's
        // O(1) per-vertex metadata (base_vid stored alongside layer_vid in
        // _vertex_info).
        std::vector<vertex_id_t> layer_base_vids(n_l);
        for (vertex_num_t v = 0; v < n_l; ++v) {
            layer_base_vids[v] = layer.get_base_vid(v);
        }

        // Parallel coverage test: for each sample, brute-force nearest over
        // layer_base_vids and compare against R_h.
        std::atomic<uint32_t> covered{0};
        std::atomic<uint32_t> tested{0};
        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_samples),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                uint32_t local_covered = 0;
                uint32_t local_tested = 0;
                for (vertex_num_t i = r.begin(); i < r.end(); ++i) {
                    const vertex_id_t q = i * step;
                    if (q >= n) break;
                    const vec_ele_t* q_vec = base_vecs.get(q);

                    distance_t best = std::numeric_limits<distance_t>::max();
                    for (const vertex_id_t bv : layer_base_vids) {
                        const distance_t d = dist_func(q_vec, base_vecs.get(bv));
                        if (d < best) best = d;
                        if (best <= R_h) break;  // early exit: already covered
                    }
                    ++local_tested;
                    if (best <= R_h) ++local_covered;
                }
                covered.fetch_add(local_covered, std::memory_order_relaxed);
                tested.fetch_add(local_tested, std::memory_order_relaxed);
            }
        );

        const uint32_t total_tested = tested.load();
        const uint32_t total_covered = covered.load();
        const float rate = (total_tested > 0)
            ? (100.0f * total_covered / total_tested)
            : 0.0f;

        ARTEA_INFO(fmt::format(
            "  Layer {}: |L_h|={:>8}, R_{}={:>12.4f}, coverage = {}/{} = {:.2f}%",
            l + 1, n_l, l + 1, R_h, total_covered, total_tested, rate));

        // Soft expectation: the bottom layer should cover most of the
        // dataset. Upper layers may have arbitrarily low coverage (their
        // radius is larger but their vertex count is much smaller).
        if (l == 0) {
            EXPECT_GE(rate, 50.0f)
                << "Layer 0 coverage is suspiciously low: " << rate << "%";
        }
    }
}

// ============================================================
//  Separation rate: for each layer h, what fraction of sampled
//  layer-h vertices have their nearest other layer-h vertex at
//  distance >= R_h?
//
//  An ideal r-net has separation = 100% (pairwise distances all >= R).
//  Under optimistic concurrent insertion the paper explicitly allows
//  localized violations; we measure the rate as a sanity check.
// ============================================================

TEST_F(StackedRGraphTest, SeparationRate) {
    const auto& base_vecs = DataProvider::instance().get_dataset().get_base_vecs();
    auto& dist_func = DataProvider::instance().get_dist_func();
    const auto num_layers = _graph->get_num_layers();

    ARTEA_INFO(fmt::format(
        "--- Separation rate (num_samples={}) ---",
        g_config.separation_num_samples));

    for (layer_id_t l = 0; l < num_layers; ++l) {
        const auto& layer = _graph->get_layer_graph(l);
        const vertex_num_t n_l = layer.get_num_vertices();
        if (n_l < 2) {
            ARTEA_INFO(fmt::format(
                "  Layer {}: |L_h|={}, too few vertices for separation test",
                l + 1, n_l));
            continue;
        }

        const distance_t R_h = static_cast<distance_t>(
            radius_at_layer(DataProvider::instance().get_l1_radius(),
                            g_config.rnet_beta,
                            static_cast<layer_num_t>(l + 1)));

        // Resolve every layer-h vertex back to its base_vid via the
        // InternalGraph's O(1) per-vertex metadata.
        std::vector<vertex_id_t> layer_base_vids(n_l);
        for (vertex_num_t v = 0; v < n_l; ++v) {
            layer_base_vids[v] = layer.get_base_vid(v);
        }

        // Sample at most separation_num_samples source vertices from this
        // layer; for each source, brute-force scan all other layer-h
        // vertices for the nearest.
        const vertex_num_t num_samples =
            std::min<vertex_num_t>(g_config.separation_num_samples, n_l);
        const vertex_num_t step = std::max<vertex_num_t>(1, n_l / num_samples);

        std::atomic<uint32_t> separated{0};
        std::atomic<uint32_t> tested{0};
        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_samples),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                uint32_t local_sep = 0;
                uint32_t local_tested = 0;
                for (vertex_num_t i = r.begin(); i < r.end(); ++i) {
                    const vertex_num_t v = i * step;
                    if (v >= n_l) break;
                    const vertex_id_t src_bv = layer_base_vids[v];
                    const vec_ele_t* src_vec = base_vecs.get(src_bv);

                    distance_t nearest = std::numeric_limits<distance_t>::max();
                    for (vertex_num_t u = 0; u < n_l; ++u) {
                        if (u == v) continue;
                        const distance_t d =
                            dist_func(src_vec, base_vecs.get(layer_base_vids[u]));
                        if (d < nearest) nearest = d;
                    }
                    ++local_tested;
                    if (nearest >= R_h) ++local_sep;
                }
                separated.fetch_add(local_sep, std::memory_order_relaxed);
                tested.fetch_add(local_tested, std::memory_order_relaxed);
            }
        );
        const uint32_t total_tested = tested.load();
        const uint32_t total_sep = separated.load();
        const float rate = (total_tested > 0)
            ? (100.0f * total_sep / total_tested)
            : 0.0f;

        ARTEA_INFO(fmt::format(
            "  Layer {}: |L_h|={:>8}, R_{}={:>12.4f}, separation = {}/{} = {:.2f}%",
            l + 1, n_l, l + 1, R_h, total_sep, total_tested, rate));

        // Soft expectation: not a hard assertion, since the paper explicitly
        // allows localized violations under concurrent insertion. We still
        // want at least a meaningful fraction of vertices to satisfy the
        // separation property; under 50% would indicate something broken.
        EXPECT_GE(rate, 50.0f)
            << "Layer " << l << " separation rate is suspiciously low: "
            << rate << "%";
    }
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

    // StackedRGraph parameters.
    program.add_argument("--beta")
        .default_value(2.0f)
        .scan<'g', float>()
        .help("R-net radius growth factor between layers (default: 2.0)");
    program.add_argument("--l1-radius")
        .default_value(-1.0f)
        .scan<'g', float>()
        .help("L1 rnet_radius. If negative or unset, auto-probe via DatasetProber.");
    program.add_argument("--max-nbr-size")
        .default_value(32u)
        .scan<'u', uint32_t>()
        .help("Per-vertex neighbor capacity for every layer (default: 32)");

    // DatasetProber parameters (used only for auto-probing).
    program.add_argument("--probe-num-samples")
        .default_value(500u)
        .scan<'u', uint32_t>()
        .help("Number of samples for DatasetProber (default: 500)");
    program.add_argument("--probe-quantile")
        .default_value(0.9f)
        .scan<'g', float>()
        .help("Quantile of 1-NN distance to use as L1 radius (default: 0.9)");

    // Beam search queue sizes.
    program.add_argument("--search-nn-qs")
        .default_value(40u)
        .scan<'u', uint32_t>()
        .help("Queue size for Phase 1 beam-search descent (default: 40)");
    program.add_argument("--select-nbrs-qs")
        .default_value(500u)
        .scan<'u', uint32_t>()
        .help("Queue size for Phase 2 candidate gathering (default: 500)");

    // Coverage / separation sampling parameters.
    program.add_argument("--coverage-num-samples")
        .default_value(1000u)
        .scan<'u', uint32_t>()
        .help("Base points sampled per layer for coverage rate test (default: 1000)");
    program.add_argument("--separation-num-samples")
        .default_value(500u)
        .scan<'u', uint32_t>()
        .help("Layer-h vertices sampled per layer for separation rate test (default: 500)");

    program.add_argument("-v", "--verbose")
        .default_value(false)
        .implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path  = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.rnet_beta    = program.get<float>("--beta");
    g_config.max_nbr_size = program.get<uint32_t>("--max-nbr-size");
    g_config.probe_num_samples = program.get<uint32_t>("--probe-num-samples");
    g_config.probe_quantile    = program.get<float>("--probe-quantile");
    g_config.search_nn_qs       = program.get<uint32_t>("--search-nn-qs");
    g_config.select_nbrs_qs     = program.get<uint32_t>("--select-nbrs-qs");
    g_config.coverage_num_samples   = program.get<uint32_t>("--coverage-num-samples");
    g_config.separation_num_samples = program.get<uint32_t>("--separation-num-samples");
    g_config.verbose      = program.get<bool>("--verbose");

    const float l1 = program.get<float>("--l1-radius");
    g_config.l1_radius_provided = (l1 >= 0.0f);
    g_config.l1_rnet_radius     = l1;

    // Print configuration.
    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset:      " << g_config.dataset_name << std::endl;
    std::cout << "rnet_beta:    " << g_config.rnet_beta << std::endl;
    if (g_config.l1_radius_provided) {
        std::cout << "L1 radius:    " << g_config.l1_rnet_radius << " (user-provided)" << std::endl;
    } else {
        std::cout << "L1 radius:    auto-probe" << std::endl;
        std::cout << "  probe samples: " << g_config.probe_num_samples << std::endl;
        std::cout << "  probe quantile: " << g_config.probe_quantile << std::endl;
    }
    std::cout << "max_nbr_size: " << g_config.max_nbr_size << std::endl;
    std::cout << "search_nn_qs:   " << g_config.search_nn_qs << std::endl;
    std::cout << "select_nbrs_qs: " << g_config.select_nbrs_qs << std::endl;
    std::cout << "==========================\n" << std::endl;

    DataProvider::instance().init();

    return RUN_ALL_TESTS();
}
