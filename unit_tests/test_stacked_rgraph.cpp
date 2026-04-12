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
 * @Description: Tests for the reworked Stacked R-Nets index. All vertices
 *               live in a unified HierarchicalGraph (no per-layer
 *               InternalGraph), every vertex has one global vid that is
 *               valid at every level it participates in, and edges are
 *               built per-level via HierarchicalPruningUpdater.
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
    float rnet_beta;
    bool  l1_radius_provided;
    float l1_rnet_radius;
    uint32_t max_nbr_size;

    // DatasetProber parameters (when L1 radius is auto-probed)
    uint32_t probe_num_samples;
    float    probe_quantile;

    // Beam-search queue sizes
    uint32_t search_nn_qs;
    uint32_t select_nbrs_qs;

    // RNG pruning scale (passed to HierarchicalPruningUpdater).
    float pruning_scale;

    // Coverage sampling parameters
    uint32_t coverage_num_samples;

    bool verbose;
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
            _l1_radius = _probe_l1_radius(base_vecs, *_dist_func);
            ARTEA_INFO(fmt::format(
                "Auto-probed L1 rnet_radius = {:.6f} "
                "({}th-percentile 1-NN distance, num_samples={})",
                _l1_radius,
                static_cast<int>(g_config.probe_quantile * 100.0f),
                g_config.probe_num_samples));
        }
    }

    auto get_dataset()  -> vector_dataset_t& { return *_dataset; }
    auto get_dist_func()-> dist_func_t&      { return *_dist_func; }
    auto get_l1_radius() const -> float      { return _l1_radius; }

private:
    DataProvider() = default;

    auto _probe_l1_radius(const vector_array_t& base_vecs,
                          const dist_func_t& dist_func) -> float {
        dataset_prober_t prober(base_vecs, dist_func);
        const std::vector<float> quantiles = { g_config.probe_quantile };

        ARTEA_INFO(fmt::format(
            "Probing dataset for 1-NN {}th-percentile (num_samples={}) ...",
            static_cast<int>(g_config.probe_quantile * 100.0f),
            g_config.probe_num_samples));
        auto result = prober.probe(quantiles, g_config.probe_num_samples);
        const distance_t r = result.table[0][0];
        return static_cast<float>(r);
    }

    std::unique_ptr<vector_dataset_t> _dataset;
    std::unique_ptr<dist_func_t>      _dist_func;
    float                             _l1_radius = 0.0f;
};

// ============================================================
//  Helpers
// ============================================================

namespace {

/** @brief R_h = L1_radius * beta^(h-1) (h is 1-indexed per paper). */
inline auto radius_at_paper_layer(
    const float l1_radius, const float beta, const layer_num_t h
) -> distance_t {
    float r = l1_radius;
    for (layer_num_t i = 1; i < h; ++i) r *= beta;
    return static_cast<distance_t>(r);
}

}  // anonymous namespace

// ============================================================
//  Fixture: build the index once for the whole suite.
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
            "Building StackedRGraph: beta={:.3f}, L1_radius={:.6f}, "
            "max_nbr={}, search_nn_qs={}, select_nbrs_qs={}, "
            "pruning_scale={:.3f}",
            beta, l1_radius, g_config.max_nbr_size,
            g_config.search_nn_qs, g_config.select_nbrs_qs,
            g_config.pruning_scale));

        auto t0 = std::chrono::high_resolution_clock::now();

        const vertex_num_t total_vertices =
            static_cast<vertex_num_t>(base_vecs.get_num_vecs());
        stacked_rgraph::rgraph_config_t rgraph_config(
            beta, l1_radius,
            static_cast<vertex_num_t>(g_config.search_nn_qs),
            static_cast<vertex_num_t>(g_config.select_nbrs_qs),
            g_config.max_nbr_size);
        _graph = std::make_unique<stacked_rgraph::index_t>(
            total_vertices, rgraph_config);

        // Build the updater fresh for this suite. It only needs the
        // distance functor, the vector storage (for RNG triangle
        // checks), and a scale coefficient.
        _pruning_updater = std::make_unique<hierarchical_pruning_updater_t>(
            dist_func,
            _graph->get_vecs_storage(),
            static_cast<ratio_t>(g_config.pruning_scale));

        vector_array_t owned_batch =
            base_vecs.extract_subset(0, total_vertices);
        stacked_rgraph::factory_t::add_vertices(
            *_graph,
            std::move(owned_batch),
            dist_func,
            *_pruning_updater);

        auto t1 = std::chrono::high_resolution_clock::now();
        _build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t1 - t0).count();

        const layer_id_t top_level_id =
            _graph->top_occupied_highest_level_id();
        ARTEA_INFO(fmt::format(
            "StackedRGraph built in {} ms: top_occupied_level={}, "
            "max_highest_level_id={}, max_restrict_level={}",
            _build_ms,
            (top_level_id == dynamic::hierarchical_graph_t
                ::unassigned_highest_level_id)
                ? -1 : static_cast<int>(top_level_id),
            _graph->max_highest_level_id(),
            _graph->max_restrict_level()));

        for (layer_id_t h = 0; h <= _graph->max_highest_level_id(); ++h) {
            const auto& bucket = _graph->get_vids_with_highest_level(h);
            const float ratio = 100.0f * bucket.size() /
                base_vecs.get_num_vecs();
            ARTEA_INFO(fmt::format(
                "  highest_level_id={}: {} vertices ({:.2f}% of base), "
                "R_{} = {:.6f}",
                h, bucket.size(), ratio, h + 1,
                radius_at_paper_layer(l1_radius, beta,
                    static_cast<layer_num_t>(h + 1))));
        }
    }

    static void TearDownTestSuite() {
        _graph.reset();
        _pruning_updater.reset();
    }

    static std::unique_ptr<stacked_rgraph::index_t>         _graph;
    static std::unique_ptr<hierarchical_pruning_updater_t>  _pruning_updater;
    static int64_t                                          _build_ms;
};

std::unique_ptr<stacked_rgraph::index_t>
    StackedRGraphTest::_graph = nullptr;
std::unique_ptr<hierarchical_pruning_updater_t>
    StackedRGraphTest::_pruning_updater = nullptr;
int64_t StackedRGraphTest::_build_ms = 0;

// ============================================================
//  Test cases
// ============================================================

TEST_F(StackedRGraphTest, HierarchyNonEmpty) {
    ASSERT_NE(_graph, nullptr);
    using HG = dynamic::hierarchical_graph_t;
    EXPECT_NE(_graph->top_occupied_highest_level_id(),
              HG::unassigned_highest_level_id);

    // Level 0 must hold every vertex — it's the base layer.
    const auto& base_vecs =
        DataProvider::instance().get_dataset().get_base_vecs();
    EXPECT_EQ(_graph->get_num_vertices(),
              static_cast<vertex_num_t>(base_vecs.get_num_vecs()));
}

TEST_F(StackedRGraphTest, GroupSizesShrinkUpward) {
    // Each highest_level_id group should be no larger than the group
    // one below it — vertices "drop out" of higher levels.
    const layer_id_t max_h = _graph->max_highest_level_id();
    for (layer_id_t h = 1; h <= max_h; ++h) {
        const std::size_t lower =
            _graph->get_vids_with_highest_level(h - 1).size();
        const std::size_t upper =
            _graph->get_vids_with_highest_level(h).size();
        EXPECT_LE(upper, lower)
            << "Group highest_level_id=" << static_cast<int>(h)
            << " (" << upper << ") should be <= group "
            << static_cast<int>(h - 1) << " (" << lower << ")";
    }
}

TEST_F(StackedRGraphTest, NeighborListsAreValid) {
    const auto& base_vecs =
        DataProvider::instance().get_dataset().get_base_vecs();
    const vertex_num_t n = static_cast<vertex_num_t>(base_vecs.get_num_vecs());
    const layer_id_t max_h = _graph->max_highest_level_id();

    // Sample-based verification across per-highest-level groups.
    constexpr vertex_num_t max_per_group = 5000;

    for (layer_id_t h = 0; h <= max_h; ++h) {
        const auto& bucket = _graph->get_vids_with_highest_level(h);
        const vertex_num_t n_h = static_cast<vertex_num_t>(bucket.size());
        if (n_h == 0) continue;

        const vertex_num_t step =
            std::max<vertex_num_t>(1, n_h / max_per_group);

        for (vertex_num_t i = 0; i < n_h; i += step) {
            const vertex_id_t vid = bucket[i];

            // Check every level this vertex participates in.
            for (layer_id_t cur_level = 0; cur_level <= h; ++cur_level) {
                const auto slot = _graph->fetch_layer_nbrs(vid, cur_level);
                const vertex_num_t count =
                    _graph->num_valid_nbrs(vid, cur_level);
                const vertex_num_t cap = _graph->max_nbr_size(cur_level);
                ASSERT_LE(count, cap)
                    << "vid=" << vid << " cur_level=" << cur_level
                    << " count=" << count << " > cap=" << cap;
                for (vertex_num_t j = 0; j < count; ++j) {
                    const auto& nbr = slot[j];
                    ASSERT_LT(nbr.get_vid(), n)
                        << "vid=" << vid << " cur_level=" << cur_level
                        << " slot[" << j << "] OOB vid=" << nbr.get_vid();
                }
            }
        }
    }
}

TEST_F(StackedRGraphTest, Level0IsHalfFilled) {
    // IndexFactory writes only the first max_nbr_size positions of each
    // vertex's L0 slot (L0 capacity is 2 * max_nbr_size; the back half
    // stays invalid for a future refiner pass).
    const auto& base_vecs =
        DataProvider::instance().get_dataset().get_base_vecs();
    (void)base_vecs;

    const vertex_num_t cap_l0  = _graph->max_nbr_size(0);
    const vertex_num_t cap_soft = _graph->max_nbr_size();
    EXPECT_EQ(cap_l0, cap_soft * 2);

    const auto& bucket_l0 = _graph->get_vids_with_highest_level(0);
    if (bucket_l0.empty()) return;

    constexpr vertex_num_t max_sample = 2000;
    const vertex_num_t step = std::max<vertex_num_t>(
        1, static_cast<vertex_num_t>(bucket_l0.size()) / max_sample);

    for (vertex_num_t i = 0;
         i < static_cast<vertex_num_t>(bucket_l0.size());
         i += step)
    {
        const vertex_id_t vid = bucket_l0[i];
        const vertex_num_t cnt = _graph->num_valid_nbrs(vid, 0);
        EXPECT_LE(cnt, cap_soft)
            << "vid=" << vid << " L0 count=" << cnt
            << " exceeds soft cap=" << cap_soft
            << " — factory is supposed to leave the rear half as sentinel";
    }
}

// ============================================================
//  Coverage rate: r-net covering property at each level.
//
//  For each cur_level >= 1, every vertex that participates at
//  cur_level - 1 (the level directly below) should have at least one
//  cur_level participant within radius R_{cur_level + 1}
//  (paper 1-indexed). A vertex participates at level L iff its
//  highest_level_id >= L.
// ============================================================

TEST_F(StackedRGraphTest, CoverageRate) {
    const auto& base_vecs =
        DataProvider::instance().get_dataset().get_base_vecs();
    auto& dist_func = DataProvider::instance().get_dist_func();
    const layer_id_t max_h = _graph->max_highest_level_id();

    ARTEA_INFO(fmt::format(
        "--- Coverage rate (num_samples={}) ---",
        g_config.coverage_num_samples));

    for (layer_id_t cur_level = 1; cur_level <= max_h; ++cur_level) {
        // Covering population: every vid whose highest_level_id >= cur_level.
        std::vector<vertex_id_t> cover_vids;
        for (layer_id_t h = cur_level; h <= max_h; ++h) {
            const auto& bucket = _graph->get_vids_with_highest_level(h);
            cover_vids.insert(cover_vids.end(),
                              bucket.begin(), bucket.end());
        }
        const vertex_num_t n_cover =
            static_cast<vertex_num_t>(cover_vids.size());
        if (n_cover == 0) continue;

        // Queried population: vertices at level cur_level - 1 (and below
        // it) that should be covered. Sample from the full base if
        // cur_level == 1.
        const distance_t R_cov = radius_at_paper_layer(
            DataProvider::instance().get_l1_radius(),
            g_config.rnet_beta,
            static_cast<layer_num_t>(cur_level));

        std::vector<vertex_id_t> query_vids;
        if (cur_level == 1) {
            const vertex_num_t n = static_cast<vertex_num_t>(base_vecs.get_num_vecs());
            const vertex_num_t num_samples = std::min<vertex_num_t>(g_config.coverage_num_samples, n);
            const vertex_num_t step =
                std::max<vertex_num_t>(1, n / num_samples);
            query_vids.reserve(num_samples);
            for (vertex_num_t i = 0; i < num_samples; ++i) {
                const vertex_id_t v = i * step;
                if (v >= n) break;
                query_vids.push_back(v);
            }
        } else {
            // Queried population: vids whose highest_level_id >= cur_level-1.
            std::vector<vertex_id_t> below;
            for (layer_id_t h = cur_level - 1; h <= max_h; ++h) {
                const auto& bucket = _graph->get_vids_with_highest_level(h);
                below.insert(below.end(), bucket.begin(), bucket.end());
            }
            const vertex_num_t n_below =
                static_cast<vertex_num_t>(below.size());
            if (n_below == 0) continue;
            const vertex_num_t num_samples = std::min<vertex_num_t>(
                g_config.coverage_num_samples, n_below);
            const vertex_num_t step =
                std::max<vertex_num_t>(1, n_below / num_samples);
            query_vids.reserve(num_samples);
            for (vertex_num_t i = 0; i < num_samples; ++i) {
                const vertex_num_t idx = i * step;
                if (idx >= n_below) break;
                query_vids.push_back(below[idx]);
            }
        }

        const vertex_num_t num_queries =
            static_cast<vertex_num_t>(query_vids.size());

        std::atomic<uint32_t> covered{0};
        std::atomic<uint32_t> tested{0};
        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_queries),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                uint32_t local_covered = 0;
                uint32_t local_tested  = 0;
                for (vertex_num_t i = r.begin(); i < r.end(); ++i) {
                    const vec_ele_t* q_vec =
                        base_vecs.get(query_vids[i]);
                    distance_t best =
                        std::numeric_limits<distance_t>::max();
                    for (const vertex_id_t cv : cover_vids) {
                        const distance_t d =
                            dist_func(q_vec, base_vecs.get(cv));
                        if (d < best) best = d;
                        if (best <= R_cov) break;
                    }
                    ++local_tested;
                    if (best <= R_cov) ++local_covered;
                }
                covered.fetch_add(local_covered,
                                  std::memory_order_relaxed);
                tested.fetch_add(local_tested,
                                 std::memory_order_relaxed);
            }
        );

        const uint32_t total_tested  = tested.load();
        const uint32_t total_covered = covered.load();
        const float    rate = (total_tested > 0)
            ? (100.0f * total_covered / total_tested)
            : 0.0f;

        ARTEA_INFO(fmt::format(
            "  level {}: |cover|={}, R={:.4f}, coverage = {}/{} = {:.2f}%",
            cur_level, n_cover, R_cov, total_covered, total_tested, rate));

        EXPECT_GE(rate, 50.0f)
            << "Level " << static_cast<int>(cur_level)
            << " coverage is suspiciously low: " << rate << "%";
    }
}

// ============================================================
//  Query-path smoke test: HierarchicalGraphRouter returns sane
//  results against the built graph.
// ============================================================

TEST_F(StackedRGraphTest, QueryRouterSmoke) {
    const auto& vecs_storage = _graph->get_vecs_storage();
    auto& dist_func = DataProvider::instance().get_dist_func();

    const vertex_num_t num_queries = std::min<vertex_num_t>(
        100, static_cast<vertex_num_t>(vecs_storage.get_num_vecs()));

    dynamic::hierarchical_graph_router_t router(
        vecs_storage, dist_func,
        /*topk=*/1,
        /*search_nn_qs=*/g_config.search_nn_qs,
        /*candidate_queue_size=*/g_config.search_nn_qs);
    router.initialize();

    const auto& hg = _graph->get_hierarchical_graph();
    for (vertex_num_t i = 0; i < num_queries; ++i) {
        const auto results = router.beam_search(vecs_storage.get(i), hg);
        ASSERT_FALSE(results.empty())
            << "beam_search returned no candidates for query " << i;
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
    program.add_argument("--select-nbrs-qs")
        .default_value(500u).scan<'u', uint32_t>();

    program.add_argument("--pruning-scale")
        .default_value(1.0f).scan<'g', float>()
        .help("RNG triangle-inequality scale for HierarchicalPruningUpdater");

    program.add_argument("--coverage-num-samples")
        .default_value(1000u).scan<'u', uint32_t>();

    program.add_argument("-v", "--verbose")
        .default_value(false).implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program;
        return 1;
    }

    g_config.config_path           = program.get<std::string>("--config");
    g_config.dataset_name          = program.get<std::string>("--dataset");
    g_config.rnet_beta             = program.get<float>("--beta");
    g_config.max_nbr_size          = program.get<uint32_t>("--max-nbr-size");
    g_config.probe_num_samples     = program.get<uint32_t>("--probe-num-samples");
    g_config.probe_quantile        = program.get<float>("--probe-quantile");
    g_config.search_nn_qs          = program.get<uint32_t>("--search-nn-qs");
    g_config.select_nbrs_qs        = program.get<uint32_t>("--select-nbrs-qs");
    g_config.pruning_scale         = program.get<float>("--pruning-scale");
    g_config.coverage_num_samples  = program.get<uint32_t>("--coverage-num-samples");
    g_config.verbose               = program.get<bool>("--verbose");

    const float l1 = program.get<float>("--l1-radius");
    g_config.l1_radius_provided = (l1 >= 0.0f);
    g_config.l1_rnet_radius     = l1;

    std::cout << "\n=== Test Configuration ===\n";
    std::cout << "Dataset:      " << g_config.dataset_name << "\n";
    std::cout << "rnet_beta:    " << g_config.rnet_beta << "\n";
    if (g_config.l1_radius_provided) {
        std::cout << "L1 radius:    " << g_config.l1_rnet_radius
                  << " (user-provided)\n";
    } else {
        std::cout << "L1 radius:    auto-probe\n";
    }
    std::cout << "max_nbr_size: " << g_config.max_nbr_size << "\n";
    std::cout << "search_nn_qs:   " << g_config.search_nn_qs << "\n";
    std::cout << "select_nbrs_qs: " << g_config.select_nbrs_qs << "\n";
    std::cout << "pruning_scale:  " << g_config.pruning_scale << "\n";
    std::cout << "==========================\n\n";

    DataProvider::instance().init();
    return RUN_ALL_TESTS();
}
