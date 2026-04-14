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
 * @FilePath: /Artea/unit_tests/test_hierarchical_graph.cpp
 * @Description: Coverage tests for dynamic::HierarchicalGraph and the
 *               dynamic → compact compactor. Focus is on the primitives
 *               that stacked_rgraph::IndexFactory relies on:
 *                 - add_vertices reserving contiguous vids,
 *                 - assign_layer claiming per-arena slots (parallel-safe),
 *                 - fetch_layer_nbrs slot layout and sentinel init,
 *                 - with_locked_nbrs serializing concurrent edge writes,
 *                 - bucket accessors (get_vids_with_highest_level,
 *                   top_occupied_level_id),
 *                 - compactor fidelity (dynamic topology → compact
 *                   topology byte-for-byte).
 *               A dataset is loaded only to fix a realistic vertex count;
 *               no actual build/search is performed here.
 */

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <unordered_set>
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
//  Global configuration
// ============================================================

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    uint32_t    num_vertices;          // scale for the fixture graph
    uint32_t    max_nbr_size;
    uint32_t    max_restrict_level;
    uint32_t    seed;
} g_config;

// ============================================================
//  DataProvider — loads a dataset once (used only for scale).
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
        const auto& base = _dataset->get_base_vecs();
        ARTEA_INFO(fmt::format(
            "Dataset loaded: {} vectors, {} dims (using first {} for tests)",
            base.get_num_vecs(), base.get_vec_dim(), g_config.num_vertices));
    }

    auto dataset_size() const -> uint32_t {
        return static_cast<uint32_t>(
            _dataset->get_base_vecs().get_num_vecs());
    }

    auto vectors() const -> const auto& {
        return _dataset->get_base_vecs();
    }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> _dataset;
};

// ============================================================
//  Helpers
// ============================================================

namespace {

using hg_t               = dynamic::hierarchical_graph_t;
using compact_hg_t       = compact::hierarchical_graph_t;
using compactor_t        = hierarchical_graph_compactor_t;

/** @brief Deterministically pick a highest_level_id for @p vid.
 *         Skewed so arena[0] dominates, decaying upward — mimics what
 *         a well-behaved r-net build produces and exercises every bucket. */
auto pick_highest_level(uint32_t vid, layer_id_t max_h, uint32_t seed)
    -> layer_id_t
{
    std::mt19937 rng(seed ^ (vid * 2654435761u));
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const float u = uniform(rng);
    // Geometric-like: P(h) ≈ 0.5^h until capped at max_h.
    layer_id_t h = 0;
    float threshold = 0.5f;
    while (h < max_h && u > threshold) {
        ++h;
        threshold += 0.5f * std::pow(0.5f, static_cast<float>(h));
    }
    return h;
}

}  // namespace

// ============================================================
//  Fixture
// ============================================================

class HierarchicalGraphTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        _num_vertices = std::min<uint32_t>(
            g_config.num_vertices,
            DataProvider::instance().dataset_size());
        _max_nbr      = g_config.max_nbr_size;
        _max_h        = static_cast<layer_id_t>(g_config.max_restrict_level);

        _graph = std::make_unique<hg_t>(
            /*max_restrict_level=*/_max_h,
            /*max_nbr_size=*/_max_nbr,
            /*total_vertices=*/_num_vertices);

        // Reserve ids, then parallel-assign layers — exactly the
        // two-phase pattern stacked_rgraph::IndexFactory uses.
        const vertex_id_t first = _graph->add_vertices(_num_vertices);
        ASSERT_EQ(first, 0u);
        _intended.resize(_num_vertices);
        for (vertex_id_t vid = 0; vid < _num_vertices; ++vid) {
            _intended[vid] = pick_highest_level(vid, _max_h, g_config.seed);
        }
        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, _num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    _graph->assign_layer(vid, _intended[vid]);
                }
            });
    }

    static void TearDownTestSuite() {
        _graph.reset();
        _intended.clear();
    }

    static std::unique_ptr<hg_t>    _graph;
    static std::vector<layer_id_t>  _intended;
    static vertex_num_t             _num_vertices;
    static vertex_num_t             _max_nbr;
    static layer_id_t               _max_h;
};

std::unique_ptr<hg_t>    HierarchicalGraphTest::_graph;
std::vector<layer_id_t>  HierarchicalGraphTest::_intended;
vertex_num_t             HierarchicalGraphTest::_num_vertices = 0;
vertex_num_t             HierarchicalGraphTest::_max_nbr      = 0;
layer_id_t               HierarchicalGraphTest::_max_h        = 0;

// ============================================================
//  Tests
// ============================================================

// ---- 1. Construction metadata: immutable graph-level properties ----
TEST_F(HierarchicalGraphTest, ConstructionMetadata) {
    EXPECT_EQ(_graph->max_restrict_level(), _max_h);
    EXPECT_EQ(_graph->max_nbr_size(), _max_nbr);
    // Level 0 is doubled; other levels are not.
    EXPECT_EQ(_graph->max_nbr_size(0),
              static_cast<vertex_num_t>(2 * _max_nbr));
    for (layer_id_t h = 1; h <= _max_h; ++h) {
        EXPECT_EQ(_graph->max_nbr_size(h), _max_nbr)
            << "upper level " << h << " should have max_nbr_size";
    }
    EXPECT_EQ(_graph->get_num_vertices(), _num_vertices);
}

// ---- 2. add_vertices on a fresh graph returns contiguous ids and
//         leaves rows unassigned until assign_layer runs. ----
TEST(HierarchicalGraphStandalone, AddVerticesBasics) {
    hg_t fresh(/*max_restrict_level=*/2, /*max_nbr_size=*/16,
               /*total_vertices=*/1024);
    EXPECT_EQ(fresh.get_num_vertices(), 0u);
    EXPECT_EQ(fresh.top_occupied_level_id(),
              hg_t::unassigned_highest_level_id);

    const vertex_id_t first_a = fresh.add_vertices(200);
    EXPECT_EQ(first_a, 0u);
    EXPECT_EQ(fresh.get_num_vertices(), 200u);

    const vertex_id_t first_b = fresh.add_vertices(50);
    EXPECT_EQ(first_b, 200u);
    EXPECT_EQ(fresh.get_num_vertices(), 250u);

    for (vertex_id_t vid = 0; vid < 250; ++vid) {
        EXPECT_FALSE(fresh.is_vertex_assigned(vid));
        EXPECT_EQ(fresh.get_highest_level_id(vid),
                  hg_t::unassigned_highest_level_id);
    }
    // No bucket populated yet.
    EXPECT_EQ(fresh.top_occupied_level_id(),
              hg_t::unassigned_highest_level_id);
}

// ---- 3. Parallel assign_layer produces unique slot_offsets per arena. ----
TEST_F(HierarchicalGraphTest, ParallelAssignLayerSlotUniqueness) {
    // (a) Every vid records the layer it was assigned to.
    for (vertex_id_t vid = 0; vid < _num_vertices; ++vid) {
        ASSERT_TRUE(_graph->is_vertex_assigned(vid));
        ASSERT_EQ(_graph->get_highest_level_id(vid), _intended[vid]);
    }

    // (b) Within each arena, no two vids share the same slot_offset.
    for (layer_id_t h = 0; h <= _max_h; ++h) {
        const auto& bucket = _graph->get_vids_with_highest_level(h);
        std::unordered_set<std::size_t> seen;
        seen.reserve(bucket.size());
        for (const vertex_id_t vid : bucket) {
            const auto off = _graph->get_slot_offset(vid);
            ASSERT_TRUE(seen.insert(off).second)
                << "duplicate slot_offset " << off
                << " inside arena h=" << h;
        }
        // And all offsets fit inside the pre-sized arena.
        const auto cap = _graph->get_arena_capacity_in_arena(h);
        for (const vertex_id_t vid : bucket) {
            const auto off = _graph->get_slot_offset(vid);
            // slot_offset is in nbr_t units; convert to slot units.
            const std::size_t slot_size =
                static_cast<std::size_t>(h + 2) *
                static_cast<std::size_t>(_max_nbr);
            EXPECT_LT(off / slot_size, static_cast<std::size_t>(cap))
                << "slot_offset out of bounds for arena h=" << h;
        }
    }

    // (c) Bucket sizes sum to num_vertices.
    vertex_num_t total = 0;
    for (layer_id_t h = 0; h <= _max_h; ++h) {
        total += static_cast<vertex_num_t>(
            _graph->get_vids_with_highest_level(h).size());
    }
    EXPECT_EQ(total, _num_vertices);

    // (d) top_occupied_level_id equals the largest h with
    //     any assignment.
    layer_id_t expected_top = 0;
    for (layer_id_t h = 0; h <= _max_h; ++h) {
        if (!_graph->get_vids_with_highest_level(h).empty()) {
            expected_top = h;
        }
    }
    EXPECT_EQ(_graph->top_occupied_level_id(), expected_top);
}

// ---- 4. fetch_layer_nbrs layout + sentinel init ----
TEST_F(HierarchicalGraphTest, SlotLayoutAndSentinelInit) {
    // Every assigned vid's slot must be fully sentinel-initialized
    // across every level in [0, highest_level_id]. Also: the per-level
    // spans must have the expected sizes (L0 = 2x, others = 1x).
    for (layer_id_t h = 0; h <= _max_h; ++h) {
        const auto& bucket = _graph->get_vids_with_highest_level(h);
        // Sample some vids per bucket to keep the test fast.
        const std::size_t sample_n = std::min<std::size_t>(bucket.size(), 256);
        for (std::size_t k = 0; k < sample_n; ++k) {
            const vertex_id_t vid = bucket[k];
            for (layer_id_t l = 0; l <= h; ++l) {
                const auto nbrs = _graph->fetch_layer_nbrs(vid, l);
                const std::size_t expected_len =
                    static_cast<std::size_t>(_max_nbr) *
                    (l == 0 ? 2 : 1);
                ASSERT_EQ(nbrs.size(), expected_len)
                    << "vid=" << vid << " level=" << l;
                for (const auto& nbr : nbrs) {
                    ASSERT_TRUE(nbr.is_invalid())
                        << "fresh slot should be sentinel-filled "
                           "(vid=" << vid << " level=" << l << ")";
                }
            }
            EXPECT_EQ(_graph->num_valid_nbrs(vid, 0), 0u);
            if (h > 0) EXPECT_EQ(_graph->num_valid_nbrs(vid, h), 0u);
        }
    }
}

// ---- 5. Intra-slot layout: writing at level l does not corrupt level l' ≠ l ----
TEST_F(HierarchicalGraphTest, PerLevelSlotsAreDisjoint) {
    // Pick a subset of vids that exist at multiple levels
    // (highest_level_id >= 1) and write distinct sentinels at each
    // level via fetch_layer_nbrs, then check cross-level isolation.
    std::vector<vertex_id_t> candidates;
    for (layer_id_t h = 1; h <= _max_h; ++h) {
        const auto& bucket = _graph->get_vids_with_highest_level(h);
        for (std::size_t k = 0; k < std::min<std::size_t>(bucket.size(), 64); ++k) {
            candidates.push_back(bucket[k]);
        }
    }
    ASSERT_FALSE(candidates.empty())
        << "need at least one vertex at level >= 1 for this test";

    // Write a synthetic nbr with vid == (vid * 31 + level + 1) at slot[0]
    // of each level, leaving slot[1..] as sentinel.
    for (const vertex_id_t vid : candidates) {
        const layer_id_t H = _graph->get_highest_level_id(vid);
        for (layer_id_t l = 0; l <= H; ++l) {
            auto span = _graph->fetch_layer_nbrs(vid, l);
            const vertex_id_t probe_vid =
                static_cast<vertex_id_t>((vid * 31u + l + 1u) & 0x7FFFFFFFu);
            span[0] = nbr_t::make_new_nbr(probe_vid, /*dist=*/1.0f);
            // Keep [1..] as invalid sentinels (they already are).
        }
    }

    // Read back per-level — probe must match what was written.
    for (const vertex_id_t vid : candidates) {
        const layer_id_t H = _graph->get_highest_level_id(vid);
        for (layer_id_t l = 0; l <= H; ++l) {
            const auto span = _graph->fetch_layer_nbrs(vid, l);
            const vertex_id_t expected =
                static_cast<vertex_id_t>((vid * 31u + l + 1u) & 0x7FFFFFFFu);
            ASSERT_FALSE(span[0].is_invalid());
            EXPECT_EQ(span[0].get_vid(), expected)
                << "vid=" << vid << " level=" << l
                << " — cross-level slot aliasing detected";
            EXPECT_EQ(_graph->num_valid_nbrs(vid, l), 1u);
        }
    }

    // Restore sentinels so later tests see a clean slot[0].
    for (const vertex_id_t vid : candidates) {
        const layer_id_t H = _graph->get_highest_level_id(vid);
        for (layer_id_t l = 0; l <= H; ++l) {
            auto span = _graph->fetch_layer_nbrs(vid, l);
            span[0] = nbr_t::make_invalid_nbr();
        }
    }
}

// ---- 6. with_locked_nbrs serializes concurrent writers ----
TEST_F(HierarchicalGraphTest, WithLockedNbrsIsThreadSafe) {
    // Pick a vid present at level 0 — arena[0] always exists.
    // We'll have every thread attempt to push a reverse-edge into the
    // SAME vid's level-0 slot, capped at slot.size(). At the end the
    // slot should contain exactly min(num_writes, slot_cap) valid
    // entries, with no torn writes.
    //
    // This mirrors IndexFactory::_insert_one's write_reverse_edges
    // concurrency pattern, where many threads may touch the same
    // popular vertex simultaneously.
    const vertex_id_t target_vid = 0;
    const vertex_num_t slot_cap =
        static_cast<vertex_num_t>(_graph->max_nbr_size(0));

    const std::size_t num_writers = 4096;
    std::atomic<std::size_t> successful_appends{0};

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_writers),
        [&](const tbb::blocked_range<std::size_t>& r) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                _graph->with_locked_nbrs(target_vid, 0,
                    [&](std::span<nbr_t> slot, vertex_num_t cnt) {
                        if (cnt < slot_cap) {
                            slot[cnt] = nbr_t::make_new_nbr(
                                static_cast<vertex_id_t>(i + 1),
                                static_cast<distance_t>(i));
                            if (cnt + 1 < slot_cap) {
                                slot[cnt + 1] = nbr_t::make_invalid_nbr();
                            }
                            successful_appends.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                    });
            }
        });

    const std::size_t appended = successful_appends.load();
    EXPECT_EQ(appended, std::min<std::size_t>(num_writers, slot_cap));

    // Valid prefix length equals number of successful appends.
    EXPECT_EQ(_graph->num_valid_nbrs(target_vid, 0),
              static_cast<vertex_num_t>(appended));

    // All appended entries have distinct payload vids (no write got
    // torn or overwrote a sibling because of a missing lock).
    const auto span = _graph->fetch_layer_nbrs(target_vid, 0);
    std::unordered_set<vertex_id_t> seen;
    seen.reserve(appended);
    for (std::size_t i = 0; i < appended; ++i) {
        ASSERT_FALSE(span[i].is_invalid()) << "tear at slot[" << i << "]";
        EXPECT_TRUE(seen.insert(span[i].get_vid()).second)
            << "duplicate or torn value at slot[" << i << "]";
    }

    // Restore sentinels so downstream tests see a clean slot.
    _graph->with_locked_nbrs(target_vid, 0,
        [&](std::span<nbr_t> slot, vertex_num_t /*cnt*/) {
            for (auto& n : slot) n = nbr_t::make_invalid_nbr();
        });
}

// ---- 7. Bucket membership matches intended highest_level_id ----
TEST_F(HierarchicalGraphTest, BucketsMatchAssignments) {
    for (layer_id_t h = 0; h <= _max_h; ++h) {
        const auto& bucket = _graph->get_vids_with_highest_level(h);
        for (const vertex_id_t vid : bucket) {
            EXPECT_EQ(_graph->get_highest_level_id(vid), h)
                << "vid=" << vid << " in bucket[" << h
                << "] but reports highest_level_id="
                << _graph->get_highest_level_id(vid);
        }
    }
}

// ---- 8. End-to-end workload mirroring IndexFactory's Step D ----
//        Threads concurrently pick a random vid and perform
//        locked-append + locked-overflow-prune into the same slot.
//        Asserts: the visible prefix is always sentinel-terminated,
//        no slot ever contains a sentinel followed by a valid entry,
//        and num_valid_nbrs is consistent with fetch_layer_nbrs.
TEST_F(HierarchicalGraphTest, IndexFactoryLikeWorkload) {
    // Only hammer level 1 — that's the level IndexFactory writes most
    // heavily via the L1-select-drives-L0-and-L1 fast path.
    //
    // If level 1 has no participants (which can happen with very small
    // max_h and tiny datasets), fall back to level 0.
    layer_id_t level = 1;
    if (_max_h < 1 ||
        _graph->get_vids_with_highest_level(1).empty())
    {
        level = 0;
    }

    // Collect every vid eligible at `level`.
    std::vector<vertex_id_t> eligible;
    for (layer_id_t h = level; h <= _max_h; ++h) {
        const auto& b = _graph->get_vids_with_highest_level(h);
        eligible.insert(eligible.end(), b.begin(), b.end());
    }
    if (eligible.empty()) {
        GTEST_SKIP() << "no vertices at level " << level;
    }

    const vertex_num_t slot_cap =
        static_cast<vertex_num_t>(_graph->max_nbr_size(level));

    const std::size_t num_ops = std::min<std::size_t>(
        eligible.size() * 4, 200'000u);

    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, num_ops),
        [&](const tbb::blocked_range<std::size_t>& r) {
            std::mt19937 rng(static_cast<uint32_t>(r.begin()) ^ 0x9E3779B9u);
            std::uniform_int_distribution<std::size_t> pick(
                0, eligible.size() - 1);
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                const vertex_id_t target = eligible[pick(rng)];
                const vertex_id_t payload =
                    static_cast<vertex_id_t>(i & 0x7FFFFFFFu);
                const distance_t dist = static_cast<distance_t>(i);

                _graph->with_locked_nbrs(target, level,
                    [&](std::span<nbr_t> slot, vertex_num_t cnt) {
                        const vertex_num_t cap =
                            static_cast<vertex_num_t>(slot.size());
                        if (cnt < cap) {
                            slot[cnt] = nbr_t::make_new_nbr(payload, dist);
                            if (cnt + 1 < cap) {
                                slot[cnt + 1] = nbr_t::make_invalid_nbr();
                            }
                        } else {
                            // Overflow path: replace the worst-distance
                            // entry if the new one is better. Mirrors
                            // the RNG-prune rebuild but is simpler and
                            // deterministic.
                            std::size_t worst_idx = 0;
                            distance_t worst = slot[0].get_distance();
                            for (std::size_t k = 1; k < cap; ++k) {
                                if (slot[k].get_distance() > worst) {
                                    worst = slot[k].get_distance();
                                    worst_idx = k;
                                }
                            }
                            if (dist < worst) {
                                slot[worst_idx] =
                                    nbr_t::make_new_nbr(payload, dist);
                            }
                        }
                    });
            }
        });

    // Invariants: for every eligible vid, the slot's valid prefix
    // equals num_valid_nbrs, followed either by end-of-span or by an
    // invalid sentinel — no "invalid then valid" zig-zag.
    for (const vertex_id_t vid : eligible) {
        const auto span = _graph->fetch_layer_nbrs(vid, level);
        const vertex_num_t cnt = _graph->num_valid_nbrs(vid, level);
        ASSERT_LE(cnt, slot_cap);
        for (vertex_num_t i = 0; i < cnt; ++i) {
            ASSERT_FALSE(span[i].is_invalid())
                << "gap in valid prefix at slot[" << i << "] for vid " << vid;
        }
        if (cnt < slot_cap) {
            ASSERT_TRUE(span[cnt].is_invalid())
                << "missing terminating sentinel for vid " << vid;
        }
    }
}

// ---- 9. Compactor fidelity: dynamic → compact preserves topology ----
TEST_F(HierarchicalGraphTest, CompactorPreservesTopology) {
    auto compact_graph = compactor_t::compact_graph(*_graph);

    // Structural metadata matches. Compactor trims max_restrict_level
    // to the source's top_occupied_level_id, so compare against that.
    EXPECT_EQ(compact_graph.max_restrict_level(),
              _graph->top_occupied_level_id());
    EXPECT_EQ(compact_graph.max_nbr_size(),
              _graph->max_nbr_size());
    EXPECT_EQ(compact_graph.get_num_vertices(),
              _graph->get_num_vertices());
    EXPECT_EQ(compact_graph.top_occupied_level_id(),
              _graph->top_occupied_level_id());

    // Per-vertex highest_level_id matches.
    const vertex_num_t check_n =
        std::min<vertex_num_t>(_graph->get_num_vertices(), 2000);
    for (vertex_id_t vid = 0; vid < check_n; ++vid) {
        EXPECT_EQ(compact_graph.get_highest_level_id(vid),
                  _graph->get_highest_level_id(vid));
    }

    // Buckets match (elements as sets — dynamic source is a
    // tbb::concurrent_vector, compact copy is a std::vector).
    // Iterate up to compact's (trimmed) max — trailing dynamic arenas
    // past top_occupied_level_id are empty by construction, so
    // nothing is lost.
    const layer_id_t compact_max_h = compact_graph.max_restrict_level();
    for (layer_id_t h = 0; h <= compact_max_h; ++h) {
        const auto& src_bucket =
            _graph->get_vids_with_highest_level(h);
        const auto dst_span =
            compact_graph.get_vids_with_highest_level(h);
        ASSERT_EQ(dst_span.size(), src_bucket.size())
            << "bucket size mismatch at h=" << h;
        std::unordered_set<vertex_id_t> src_set(
            src_bucket.begin(), src_bucket.end());
        for (const vertex_id_t v : dst_span) {
            EXPECT_TRUE(src_set.count(v))
                << "compact bucket[" << h << "] contains " << v
                << " not present in source";
        }
    }

    // Per-level neighbor vids match (nbr_t → vertex_id_t).
    // Sample vids to keep the test bounded.
    for (layer_id_t h = 0; h <= compact_max_h; ++h) {
        const auto& bucket = _graph->get_vids_with_highest_level(h);
        const std::size_t sample_n =
            std::min<std::size_t>(bucket.size(), 200);
        for (std::size_t k = 0; k < sample_n; ++k) {
            const vertex_id_t vid = bucket[k];
            for (layer_id_t l = 0; l <= h; ++l) {
                const auto dyn_span = _graph->fetch_layer_nbrs(vid, l);
                const auto cmp_span = compact_graph.fetch_layer_nbrs(vid, l);
                ASSERT_EQ(dyn_span.size(), cmp_span.size());
                const vertex_num_t dyn_cnt =
                    _graph->num_valid_nbrs(vid, l);
                const vertex_num_t cmp_cnt =
                    compact_graph.num_valid_nbrs(vid, l);
                EXPECT_EQ(dyn_cnt, cmp_cnt)
                    << "valid-count mismatch vid=" << vid << " l=" << l;
                for (vertex_num_t i = 0; i < dyn_cnt; ++i) {
                    EXPECT_EQ(dyn_span[i].get_vid(), cmp_span[i])
                        << "nbr vid mismatch vid=" << vid
                        << " l=" << l << " i=" << i;
                }
            }
        }
    }
}

// ============================================================
//  Layer ↔ RefiningGraph round-trip (extract → mutate → writeback)
// ============================================================

namespace {

// RefiningGraph is now an ordinary (non-CRTP) class, so we can use it
// directly without a derived stub.
using TestRefiningGraph = dynamic::RefiningGraph<index_traits_t>;

}  // namespace

// Walk every level h ∈ [0, top_occupied_level_id]. Extract → assert
// equivalence with the source slot → mutate one row → writeback → re-extract
// → assert the mutation round-tripped. Covers L0 (identity-mapped) and L1+
// (sparse-mapped) paths in one pass.
TEST_F(HierarchicalGraphTest, LayerRefiningGraphRoundTrip) {
    // Construct a sized-down VectorArray matching the fixture's vid space.
    // Values are not read by this test (fill/writeback never call dist_func),
    // so an uninitialized buffer of the right shape is sufficient.
    const auto& full_vecs = DataProvider::instance().vectors();
    vector_array_t vecs(_num_vertices, full_vecs.get_vec_dim());
    const layer_config_t layer_cfg(_max_nbr, /*reserved=*/_max_nbr);

    const layer_id_t top_h = _graph->top_occupied_level_id();
    ASSERT_NE(top_h, hg_t::unassigned_highest_level_id);

    for (layer_id_t h = 0; h <= top_h; ++h) {
        // ---- Build the (l2g, g2l) maps and construct the matching RG. ----
        auto [l2g, g2l] = _graph->collect_layer_vids(h);
        std::unique_ptr<TestRefiningGraph> rg;
        if (h == 0) {
            EXPECT_TRUE(l2g.empty()) << "L0 should be identity-mapped";
            EXPECT_TRUE(g2l.empty()) << "L0 should be identity-mapped";
            rg = std::make_unique<TestRefiningGraph>(vecs, layer_cfg);
        } else {
            // Participating-vid count must equal Σ |bucket[h..]|.
            vertex_num_t expected = 0;
            for (layer_id_t hh = h; hh <= _max_h; ++hh) {
                expected += static_cast<vertex_num_t>(
                    _graph->get_vids_with_highest_level(hh).size());
            }
            EXPECT_EQ(l2g.size(), expected);
            EXPECT_EQ(g2l.size(), _num_vertices);
            rg = std::make_unique<TestRefiningGraph>(
                vecs, layer_cfg, l2g, g2l);
            EXPECT_FALSE(rg->is_identity_mapped());
        }

        // ---- Fill from the layer. ----
        _graph->fill_refining_graph_from_layer(*rg, h);

        // For freshly-built indexes the slot is all-invalid (no edges
        // written by the fixture), so num_valid_nbrs == 0 and rg rows are
        // empty. Just sanity-check counts and identity round-trip.
        rg->parallel_for_each_vertex([&](const vertex_id_t /*local_vid*/, const vertex_id_t global_vid) {
            if (!_graph->is_vertex_assigned(global_vid)) return;
            const vertex_num_t hg_cnt = _graph->num_valid_nbrs(global_vid, h);
            const auto& rg_nbrs = rg->fetch_nbrs(global_vid);
            EXPECT_EQ(rg_nbrs.size(), hg_cnt)
                << "vid=" << global_vid << " h=" << h;
        });

        // ---- Mutate one row: pick the first local row, push a fake
        //      neighbor pair (vid_at(other), 1.0). Writeback, re-extract,
        //      assert the mutation survived. ----
        if (rg->get_num_vertices() < 2) continue;
        const vertex_id_t pivot_global = rg->vid_at(0);
        const vertex_id_t other_global = rg->vid_at(1);
        rg->fetch_nbrs(pivot_global).clear();
        rg->fetch_nbrs(pivot_global).emplace_back(
            other_global, distance_t{1}, /*is_new=*/true);

        _graph->writeback_layer_from_refining_graph(*rg, h);

        // Re-read the slot directly from hg.
        const auto written = _graph->fetch_layer_nbrs(pivot_global, h);
        ASSERT_GE(written.size(), 1u);
        EXPECT_FALSE(written[0].is_invalid());
        EXPECT_EQ(written[0].get_vid(), other_global);
        EXPECT_EQ(written[0].get_distance(), distance_t{1});
        if (written.size() >= 2) {
            EXPECT_TRUE(written[1].is_invalid())
                << "writeback must terminate the slot with a sentinel";
        }
    }
}

// ============================================================
//  main
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_hierarchical_graph");
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"));
    program.add_argument("--num-vertices")
        .default_value(100'000u).scan<'u', uint32_t>()
        .help("Number of vertices to simulate (bounded by dataset size).");
    program.add_argument("--max-nbr-size")
        .default_value(32u).scan<'u', uint32_t>();
    program.add_argument("--max-highest-level-id")
        .default_value(4u).scan<'u', uint32_t>()
        .help("Inclusive upper bound for per-vertex highest_level_id.");
    program.add_argument("--seed")
        .default_value(42u).scan<'u', uint32_t>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << "\n" << program;
        return 1;
    }

    g_config.config_path          = program.get<std::string>("--config");
    g_config.dataset_name         = program.get<std::string>("--dataset");
    g_config.num_vertices         = program.get<uint32_t>("--num-vertices");
    g_config.max_nbr_size         = program.get<uint32_t>("--max-nbr-size");
    g_config.max_restrict_level = program.get<uint32_t>("--max-highest-level-id");
    g_config.seed                 = program.get<uint32_t>("--seed");

    std::cout << "\n=== Test Configuration ===\n"
              << "Dataset:              " << g_config.dataset_name << "\n"
              << "num_vertices:         " << g_config.num_vertices << "\n"
              << "max_nbr_size:         " << g_config.max_nbr_size << "\n"
              << "max_restrict_level: " << g_config.max_restrict_level << "\n"
              << "seed:                 " << g_config.seed << "\n"
              << "==========================\n\n";

    DataProvider::instance().init();
    return RUN_ALL_TESTS();
}
