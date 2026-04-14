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

    // RNG pruning coefficients (forwarded to stacked_rgraph::pruning_config_t).
    float scale_coeffs;
    float shifted_coeffs;

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
            "scale_coeffs={:.3f}, shifted_coeffs={:.3f}",
            beta, l1_radius, g_config.max_nbr_size,
            g_config.search_nn_qs, g_config.select_nbrs_qs,
            g_config.scale_coeffs, g_config.shifted_coeffs));

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

        stacked_rgraph::pruning_config_t pruning_config(
            static_cast<ratio_t>(g_config.scale_coeffs),
            static_cast<ratio_t>(g_config.shifted_coeffs));

        vector_array_t owned_batch =
            base_vecs.extract_subset(0, total_vertices);
        stacked_rgraph::factory_t::add_vertices(
            *_graph,
            std::move(owned_batch),
            dist_func,
            pruning_config);

        auto t1 = std::chrono::high_resolution_clock::now();
        _build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            t1 - t0).count();

        const layer_id_t top_level_id =
            _graph->top_occupied_level_id();
        ARTEA_INFO(fmt::format(
            "StackedRGraph built in {} ms: top_occupied_level={}, "
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
            if (h == 0) {
                // Level 0 is the base layer; no r-net covering radius.
                ARTEA_INFO(fmt::format(
                    "  highest_level_id={}: {} vertices ({:.2f}% of base), "
                    "base layer (no R)",
                    h, bucket.size(), ratio));
            } else {
                ARTEA_INFO(fmt::format(
                    "  highest_level_id={}: {} vertices ({:.2f}% of base), "
                    "R_{} = {:.6f}",
                    h, bucket.size(), ratio, h,
                    radius_at_paper_layer(l1_radius, beta,
                        static_cast<layer_num_t>(h))));
            }
        }
    }

    static void TearDownTestSuite() {
        _graph.reset();
    }

    static std::unique_ptr<stacked_rgraph::index_t>  _graph;
    static int64_t                                   _build_ms;
};

std::unique_ptr<stacked_rgraph::index_t>
    StackedRGraphTest::_graph = nullptr;
int64_t StackedRGraphTest::_build_ms = 0;

// ============================================================
//  Test cases
// ============================================================

TEST_F(StackedRGraphTest, HierarchyNonEmpty) {
    ASSERT_NE(_graph, nullptr);
    using HG = dynamic::hierarchical_graph_t;
    EXPECT_NE(_graph->top_occupied_level_id(),
              HG::unassigned_highest_level_id);

    // Level 0 must hold every vertex — it's the base layer.
    const auto& base_vecs =
        DataProvider::instance().get_dataset().get_base_vecs();
    EXPECT_EQ(_graph->get_num_vertices(),
              static_cast<vertex_num_t>(base_vecs.get_num_vecs()));
}

TEST_F(StackedRGraphTest, LayerPopulationShrinksUpward) {
    // Paper r-net is nested: L_{h+1} is a subset of L_h, so the set of
    // vertices participating at level h+1 must be a subset of those at
    // level h. We check |L_h| >= |L_{h+1}|, where
    //   |L_h| = sum(arena[h..max_restrict_level]).
    //
    // Note: the per-arena bucket sizes (highest_level_id == h) are NOT
    // required to be monotone — arena[h] = |L_h| - |L_{h+1}|, whose
    // shape depends on how fast r-net radii R_h dilate relative to
    // NN_L_h on the data, and in high-dimensional datasets like SIFT
    // those aren't monotone. Only the cumulative layer populations are.
    const layer_id_t max_h = _graph->max_restrict_level();

    std::vector<std::size_t> layer_pop(max_h + 1, 0);
    for (layer_id_t h = max_h; ; --h) {
        const std::size_t bucket_size =
            _graph->get_vids_with_highest_level(h).size();
        layer_pop[h] = bucket_size;
        if (h < max_h) layer_pop[h] += layer_pop[h + 1];
        if (h == 0) break;
    }

    for (layer_id_t h = 1; h <= max_h; ++h) {
        EXPECT_LE(layer_pop[h], layer_pop[h - 1])
            << "|L_" << static_cast<int>(h) << "|=" << layer_pop[h]
            << " should be <= |L_" << static_cast<int>(h - 1)
            << "|=" << layer_pop[h - 1];
    }
}

TEST_F(StackedRGraphTest, NeighborListsAreValid) {
    const auto& base_vecs =
        DataProvider::instance().get_dataset().get_base_vecs();
    const vertex_num_t n = static_cast<vertex_num_t>(base_vecs.get_num_vecs());
    const layer_id_t max_h = _graph->max_restrict_level();

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

TEST_F(StackedRGraphTest, Level0CapacityRespected) {
    // IndexFactory now builds L0 alongside the upper levels, so every
    // vertex's L0 slot can be filled up to its full 2 * max_nbr_size
    // capacity. The only invariant we still enforce is that no vertex
    // overruns that hard cap.
    const vertex_num_t cap_l0   = _graph->max_nbr_size(0);
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
        EXPECT_LE(cnt, cap_l0)
            << "vid=" << vid << " L0 count=" << cnt
            << " exceeds L0 capacity=" << cap_l0;
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
    const layer_id_t max_h = _graph->max_restrict_level();

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
//  Separation test: paper r-net requires any two vertices in L_h
//  to be at least R_h apart. Sample M vids from L_h, brute-force
//  each one's nearest neighbor within L_h, and count how many are
//  closer than R_h. A healthy r-net yields 0 violations (or a
//  tiny fraction from concurrency races). A degenerate build
//  where vertices piled into L_h without actually being isolated
//  shows a large violation rate — exactly what we want to catch.
// ============================================================

TEST_F(StackedRGraphTest, Separation) {
    const auto& base_vecs =
        DataProvider::instance().get_dataset().get_base_vecs();
    auto& dist_func = DataProvider::instance().get_dist_func();
    const layer_id_t max_h = _graph->max_restrict_level();

    ARTEA_INFO(fmt::format(
        "--- Separation (num_samples={}) ---",
        g_config.coverage_num_samples));

    for (layer_id_t cur_level = 1; cur_level <= max_h; ++cur_level) {
        std::vector<vertex_id_t> members;
        for (layer_id_t h = cur_level; h <= max_h; ++h) {
            const auto& b = _graph->get_vids_with_highest_level(h);
            members.insert(members.end(), b.begin(), b.end());
        }
        const vertex_num_t n_members =
            static_cast<vertex_num_t>(members.size());
        if (n_members < 2) continue;

        const distance_t R_h = radius_at_paper_layer(
            DataProvider::instance().get_l1_radius(),
            g_config.rnet_beta,
            static_cast<layer_num_t>(cur_level));

        const vertex_num_t num_samples = std::min<vertex_num_t>(
            g_config.coverage_num_samples, n_members);
        const vertex_num_t step =
            std::max<vertex_num_t>(1, n_members / num_samples);

        std::atomic<uint32_t> violations{0};
        std::atomic<uint32_t> tested{0};

        tbb::parallel_for(
            tbb::blocked_range<vertex_num_t>(0, num_samples),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                uint32_t local_v = 0;
                uint32_t local_t = 0;
                for (vertex_num_t i = r.begin(); i < r.end(); ++i) {
                    const vertex_num_t idx = i * step;
                    if (idx >= n_members) break;
                    const vertex_id_t a = members[idx];
                    const vec_ele_t* a_vec = base_vecs.get(a);

                    // Early terminate: we only care if ANY member is
                    // closer than R_h. For degenerate (over-populated)
                    // layers this avoids the full O(|L_h|) brute-force.
                    bool violated = false;
                    for (const vertex_id_t b : members) {
                        if (b == a) continue;
                        const distance_t d =
                            dist_func(a_vec, base_vecs.get(b));
                        if (d < R_h) { violated = true; break; }
                    }
                    ++local_t;
                    if (violated) ++local_v;
                }
                violations.fetch_add(local_v, std::memory_order_relaxed);
                tested.fetch_add(local_t, std::memory_order_relaxed);
            }
        );

        const uint32_t tv = violations.load();
        const uint32_t tt = tested.load();
        const float correct_rate = (tt > 0)
            ? (100.0f * (tt - tv) / tt) : 0.0f;

        ARTEA_INFO(fmt::format(
            "  level {}: |L|={}, R={:.4f}, correct separation = {}/{} = {:.2f}%",
            cur_level, n_members, R_h, tt - tv, tt, correct_rate));

        EXPECT_GE(correct_rate, 90.0f)
            << "Level " << static_cast<int>(cur_level)
            << " separation broken: only " << correct_rate
            << "% of sampled vertices have no L_"
            << static_cast<int>(cur_level)
            << " neighbor closer than R_" << static_cast<int>(cur_level);
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
//  Compact the internal hierarchical graph, validate the copy,
//  then measure search recall / throughput on the compact form.
// ============================================================

TEST_F(StackedRGraphTest, CompactGraphSearchRecallAndThroughput) {
    auto& provider = DataProvider::instance();
    const auto& dataset    = provider.get_dataset();
    const auto& base_vecs  = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt         = dataset.get_gt_vecs();
    auto&       dist_func  = provider.get_dist_func();

    // ---- Step 1: compact the dynamic hierarchical graph. ----
    const auto& dyn_hg = _graph->get_hierarchical_graph();
    auto t_compact_0 = std::chrono::high_resolution_clock::now();
    auto compact_hg = hierarchical_graph_compactor_t::compact_graph(dyn_hg);
    auto t_compact_1 = std::chrono::high_resolution_clock::now();
    const auto compact_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_compact_1 - t_compact_0).count();
    ARTEA_INFO(fmt::format("Hierarchical graph compacted in {} ms", compact_ms));

    // ---- Step 2: basic structural fidelity. ----
    ASSERT_EQ(compact_hg.get_num_vertices(), dyn_hg.get_num_vertices());
    ASSERT_EQ(compact_hg.max_nbr_size(),     dyn_hg.max_nbr_size());
    ASSERT_EQ(compact_hg.top_occupied_level_id(),
              dyn_hg.top_occupied_level_id());
    ASSERT_EQ(compact_hg.max_restrict_level(),
              dyn_hg.top_occupied_level_id());

    // Spot-check neighbor-list fidelity on a sampled subset of vertices
    // across every occupied level (full equivalence is covered by
    // test_hierarchical_graph.cpp::CompactorPreservesTopology).
    const layer_id_t top_level = compact_hg.max_restrict_level();
    for (layer_id_t h = 0; h <= top_level; ++h) {
        const auto& bucket = dyn_hg.get_vids_with_highest_level(h);
        const std::size_t sample_n = std::min<std::size_t>(bucket.size(), 64);
        for (std::size_t k = 0; k < sample_n; ++k) {
            const vertex_id_t vid = bucket[k];
            for (layer_id_t l = 0; l <= h; ++l) {
                const auto dyn_span = dyn_hg.fetch_layer_nbrs(vid, l);
                const auto cmp_span = compact_hg.fetch_layer_nbrs(vid, l);
                const vertex_num_t dyn_cnt = dyn_hg.num_valid_nbrs(vid, l);
                const vertex_num_t cmp_cnt = compact_hg.num_valid_nbrs(vid, l);
                ASSERT_EQ(dyn_cnt, cmp_cnt)
                    << "vid=" << vid << " level=" << static_cast<int>(l);
                for (vertex_num_t i = 0; i < dyn_cnt; ++i) {
                    ASSERT_EQ(dyn_span[i].get_vid(), cmp_span[i])
                        << "vid=" << vid
                        << " level=" << static_cast<int>(l)
                        << " i=" << i;
                }
            }
        }
    }

    // ---- Step 3: search on compact vs dynamic; recall + throughput. ----
    const uint32_t topk       = std::min<uint32_t>(10u, gt.get_vec_dim());
    const uint32_t queue_size = std::max<uint32_t>(g_config.search_nn_qs, topk);
    const uint32_t warmup_runs = 1;
    const uint32_t test_runs   = 3;
    const uint32_t num_queries = static_cast<uint32_t>(query_vecs.get_num_vecs());

    auto time_router = [&](auto& router, const auto& hg) {
        for (uint32_t w = 0; w < warmup_runs; ++w) {
            [[maybe_unused]] auto _ = router.batch_query(query_vecs, hg);
        }
        double total_us = 0.0;
        float  total_recall = 0.0f;
        knn_results_t last_results;
        recall_estimator_t re;
        for (uint32_t r = 0; r < test_runs; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            last_results = router.batch_query(query_vecs, hg);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_us += std::chrono::duration_cast<std::chrono::microseconds>(
                t1 - t0).count();
            total_recall += re.calculate_recall_at_k(
                last_results, gt, topk, num_queries);
        }
        const double avg_us = total_us / test_runs;
        const float  recall = total_recall / test_runs;
        const double qps    = num_queries * 1e6 / avg_us;
        return std::make_tuple(std::move(last_results), recall, avg_us, qps);
    };

    compact::hierarchical_graph_router_t compact_router(
        base_vecs, dist_func,
        /*topk=*/topk,
        /*search_nn_qs=*/g_config.search_nn_qs,
        /*candidate_queue_size=*/queue_size);
    compact_router.initialize();
    auto [compact_results, compact_recall, compact_us, compact_qps] =
        time_router(compact_router, compact_hg);

    ASSERT_EQ(compact_results.size(),
              static_cast<std::size_t>(num_queries) * topk);

    dynamic::hierarchical_graph_router_t dyn_router(
        _graph->get_vecs_storage(), dist_func,
        /*topk=*/topk,
        /*search_nn_qs=*/g_config.search_nn_qs,
        /*candidate_queue_size=*/queue_size);
    dyn_router.initialize();
    auto [dyn_results, dyn_recall, dyn_us, dyn_qps] =
        time_router(dyn_router, dyn_hg);

    ARTEA_INFO(fmt::format(
        "Search comparison (num_queries={}, topk={}, search_nn_qs={}, "
        "queue_size={}):",
        num_queries, topk, g_config.search_nn_qs, queue_size));
    ARTEA_INFO(fmt::format(
        "  compact : recall@{}={:.4f}, batch_latency={:.2f} ms, QPS={:.1f}",
        topk, compact_recall, compact_us / 1000.0, compact_qps));
    ARTEA_INFO(fmt::format(
        "  dynamic : recall@{}={:.4f}, batch_latency={:.2f} ms, QPS={:.1f}",
        topk, dyn_recall, dyn_us / 1000.0, dyn_qps));
    ARTEA_INFO(fmt::format(
        "  speedup : compact QPS / dynamic QPS = {:.2f}x",
        compact_qps / dyn_qps));

    EXPECT_GT(compact_recall, 0.0f);
    EXPECT_GT(dyn_recall, 0.0f);
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
        .default_value(100u).scan<'u', uint32_t>();

    program.add_argument("--scale-coeffs")
        .default_value(1.1f).scan<'g', float>()
        .help("RNG pruning scale coefficient (stacked_rgraph::pruning_config_t)");
    program.add_argument("--shifted-coeffs")
        .default_value(0.0f).scan<'g', float>()
        .help("RNG pruning shifted coefficient (stacked_rgraph::pruning_config_t)");

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
    g_config.scale_coeffs          = program.get<float>("--scale-coeffs");
    g_config.shifted_coeffs        = program.get<float>("--shifted-coeffs");
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
    std::cout << "scale_coeffs:   " << g_config.scale_coeffs   << "\n";
    std::cout << "shifted_coeffs: " << g_config.shifted_coeffs << "\n";
    std::cout << "==========================\n\n";

    DataProvider::instance().init();
    return RUN_ALL_TESTS();
}
