// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <artea/cpu/framework/type_context/default_context.hpp>
#include <gtest/gtest.h>
#include <tbb/global_control.h>
#include <tbb/info.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace artea::cpu;

namespace {
constexpr auto metric = DistanceMetricsT::EUCLIDEAN;
constexpr vec_dim_t dimension = 16;
using refining_graph_t = refiner_traits_t<metric, dimension>::dynamic::refining_graph_t;

std::vector<vertex_id_t> neighbor_ids(const nbr_arr_t& row) {
    std::vector<vertex_id_t> ids;
    for (const auto& neighbor : row) ids.push_back(neighbor.get_vid());
    return ids;
}

// Reverse-edge insertion appends to partially filled rows, so the bridge must
// accept unordered input. Equal distances also need the log merger's ID tie-break.
void check_refinement_input(layer_id_t level, vertex_num_t destination_capacity) {
    SCOPED_TRACE(::testing::Message() << "level=" << level << " capacity=" << destination_capacity);
    vector_array_t vectors(16, dimension);
    std::fill_n(vectors.get_all(), 16 * dimension, 0.0f);
    vectors.get(4)[0] = -1.0f;
    vectors.get(8)[0] = 1.0f;
    vectors.get(10)[0] = 2.0f;
    vectors.get(12)[0] = 3.0f;
    vectors.get(6)[0] = 4.0f;

    dynamic::hierarchical_graph_t graph(1, 8, 8, 16);
    graph.add_vertices(16);
    for (vertex_id_t v = 0; v < 16; ++v) {
        graph.assign_layer(v, v >= 2 && v <= 12 && v % 2 == 0 ? 1 : 0);
    }
    constexpr vertex_id_t pivot = 2;
    constexpr std::array<vertex_id_t, 4> input_ids{12, 8, 4, 10};
    dist_func_t<metric, dimension> distance;
    auto source = graph.fetch_level_nbrs(pivot, level);
    for (std::size_t i = 0; i < input_ids.size(); ++i) {
        source[i] = nbr_t(input_ids[i], distance(vectors.get(pivot), vectors.get(input_ids[i])));
    }

    layer_config_t config(destination_capacity);
    std::unique_ptr<refining_graph_t> refining;
    if (level == 0) {
        refining = std::make_unique<refining_graph_t>(vectors, config);
    } else {
        auto [local_to_global, global_to_local] = graph.collect_layer_vids(level);
        refining = std::make_unique<refining_graph_t>(
            vectors, config, std::move(local_to_global), std::move(global_to_local));
    }
    refiner_utils_t<metric, dimension>::fill_refining_graph_from_layer(graph, *refining, level);
    const auto& row = refining->fetch_nbrs(pivot);
    std::vector<vertex_id_t> expected{4, 8, 10, 12};
    expected.resize(std::min<std::size_t>(expected.size(), destination_capacity));
    ASSERT_EQ(neighbor_ids(row), expected);
    ASSERT_TRUE(std::is_sorted(row.begin(), row.end(), strict_nbr_comp_t{}));
    for (std::size_t i = 0; i < input_ids.size(); ++i) {
        EXPECT_EQ(source[i].get_vid(), input_ids[i]);  // Import does not mutate the source graph.
    }

    if (destination_capacity < input_ids.size()) return;
    log_table_t logs;
    logs.resize(refining->get_num_vertices());
    const auto local_pivot = refining->local_id_of(pivot);
    for (int iteration = 0; iteration < 2; ++iteration) {
        logs.write_log(local_pivot, 8, 1.0f);  // Existing neighbor at a tied distance.
        logs.write_log(local_pivot, 6, 4.0f);
        logs.apply_logs(local_pivot, *refining);
        EXPECT_EQ(neighbor_ids(row), (std::vector<vertex_id_t>{4, 8, 10, 12, 6}));
        EXPECT_TRUE(std::is_sorted(row.begin(), row.end(), strict_nbr_comp_t{}));
    }
}

TEST(GraphBuildRegressions, ImportedRowsStayUniqueAfterLogMerging) {
    for (layer_id_t level : {0u, 1u}) check_refinement_input(level, 8);
}

TEST(GraphBuildRegressions, ImportKeepsClosestNeighborsWhenDestinationIsSmaller) {
    for (layer_id_t level : {0u, 1u}) check_refinement_input(level, 2);
}

TEST(GraphBuildRegressions, WiderInsertionSearchRevisitsPreviouslyRejectedCandidates) {
    // One apex fixes the seed; append just one vertex to exercise the real
    // insertion path without depending on scheduling or randomized construction.
    using factory_t = stacked_rgraph::factory_t<metric, dimension>;
    const stacked_rgraph::rgraph_config_t<metric, dimension> config(
        2.0f, 0.0f, 0.0f, 0.01f, 30, 100, 100, 32, 64);
    // Reserve enough future vertices to permit a singleton L2 apex.
    stacked_rgraph::index_t<metric, dimension> index(10'000, config);
    vector_array_t initial(33, dimension);
    std::fill_n(initial.get_all(), 33 * dimension, 0.0f);
    initial.get(0)[0] = 100.0f;
    for (vertex_id_t v = 1; v <= 31; ++v) initial.get(v)[0] = static_cast<float>(v);
    index.append_vecs(std::move(initial));
    index.add_vertices(33);
    for (vertex_id_t v = 0; v < 33; ++v) index.assign_layer(v, v == 0 ? 2 : 1);

    dist_func_t<metric, dimension> distance;
    auto add_edge = [&](vertex_id_t from, std::size_t slot, vertex_id_t to) {
        index.fetch_level_nbrs(from, 1)[slot] = nbr_t(
            to, distance(index.get_base_vecs().get(from), index.get_base_vecs().get(to)));
    };
    for (vertex_id_t v = 1; v <= 31; ++v) add_edge(0, v - 1, v);
    add_edge(1, 0, 31);
    add_edge(31, 0, 32);

    // Descent's 30-wide queue rejects vertex 31 after marking it visited.
    // Selection's 100-wide queue must reconsider 31 to discover exact match 32.
    vector_array_t batch(1, dimension);
    std::fill_n(batch.get_all(), dimension, 0.0f);
    factory_t::add_vertices(index, std::move(batch), distance, false, false);
    ASSERT_GE(index.get_hierarchical_graph().get_highest_level_id(33), 1u);
    const auto neighbors = index.fetch_level_nbrs(33, 1);
    ASSERT_FALSE(neighbors[0].is_invalid());
    EXPECT_EQ(neighbors[0].get_vid(), 32u);
    EXPECT_FLOAT_EQ(neighbors[0].get_distance(), 0.0f);
}

// Test-only index substitution controls the old-top interleaving without
// putting hooks or sleeps in the production insertion path.
class GrowthRaceIndex : public stacked_rgraph::index_t<metric, dimension> {
    using base_t = stacked_rgraph::index_t<metric, dimension>;
public:
    using base_t::base_t;
    std::atomic<bool> armed{false};
    layer_id_t synchronized_level = 1;
    bool insert_on_l0 = false;
    mutable std::barrier<> snapshots_ready{2};
    mutable std::atomic<unsigned> stale_snapshots{0};
    mutable std::atomic<unsigned> refreshed_snapshots{0};
    std::atomic<unsigned> completed_publications{0};
    std::atomic<vertex_id_t> next_staged_vid{0};

    // Preallocate the whole batch exactly as the production batch entry point
    // does. Two explicit callers can then exercise its one-vertex work without
    // assuming TBB splits a two-element range into concurrent tasks.
    void stage_pair(vector_array_t&& batch) {
        next_staged_vid = get_num_vertices();
        base_t::append_vecs(std::move(batch));
        base_t::add_vertices(2);
    }

    void append_vecs(vector_array_t&& batch) {
        if (!armed.load()) base_t::append_vecs(std::move(batch));
        else EXPECT_EQ(batch.get_num_vecs(), 1u);
    }

    auto add_vertices(vertex_num_t count) -> vertex_id_t {
        if (!armed.load()) return base_t::add_vertices(count);
        EXPECT_EQ(count, 1u);
        return next_staged_vid.fetch_add(1);
    }

    auto get_top_level_view() const {
        auto view = base_t::get_top_level_view();
        if (armed.load()) {
            if (view.top_level_id == synchronized_level) {
                if (stale_snapshots.fetch_add(1) < 2) snapshots_ready.arrive_and_wait();
            } else {
                refreshed_snapshots.fetch_add(1);
            }
        }
        return view;
    }

    void publish_layer(vertex_id_t vid) {
        if (armed.load()) {
            const auto level = get_highest_level_id(vid);
            EXPECT_LT(base_t::get_top_level_view().top_level_id, level);
            for (layer_id_t l = insert_on_l0 ? 0u : 1u; l < level; ++l) {
                EXPECT_GT(num_valid_nbrs(vid, l), 0u)
                    << "New top was published before its lower-layer edges";
            }
            completed_publications.fetch_add(1);
        }
        base_t::publish_layer(vid);
    }
};

struct GrowthRaceTraits : graph_factory_traits_t<metric, dimension> {
    struct stacked_rgraph : artea::cpu::graph_factory_traits_t<metric, dimension>::stacked_rgraph {
        using index_t = GrowthRaceIndex;
    };
};

void check_concurrent_growth(layer_id_t initial_top, bool absorbed, bool insert_on_l0) {
    SCOPED_TRACE(::testing::Message() << "top=" << initial_top
                 << " absorbed=" << absorbed << " L0=" << insert_on_l0);
    using factory_t = stacked_rgraph::IndexFactory<GrowthRaceTraits>;
    const stacked_rgraph::rgraph_config_t<metric, dimension> config(
        2.0f, 0.0f, 0.0f, 0.01f, 30, 100, 100, 32, 64);
    GrowthRaceIndex index(10'000, config);
    index.insert_on_l0 = insert_on_l0;
    dist_func_t<metric, dimension> distance;
    vector_array_t initial(10, dimension);
    std::fill_n(initial.get_all(), 10 * dimension, 0.0f);
    factory_t::add_vertices(index, std::move(initial), distance, insert_on_l0, false);
    if (initial_top == 2) {
        vector_array_t apex(1, dimension);
        std::fill_n(apex.get_all(), dimension, 0.0f);
        apex.get(0)[0] = 0.25f;
        factory_t::add_vertices(index, std::move(apex), distance, insert_on_l0, false);
    }
    ASSERT_EQ(index.top_occupied_level_id(), initial_top);
    const vertex_id_t first = index.get_num_vertices();
    const layer_id_t new_level = initial_top + 1;
    vector_array_t batch(2, dimension);
    std::fill_n(batch.get_all(), 2 * dimension, 0.0f);
    batch.get(0)[0] = 1.0f;
    // Between R_h and R_(h+1): only the newly created layer can absorb it.
    batch.get(1)[0] = absorbed ? 1.0f + 1.5f * index.radius_at(initial_top) : 2.0f;
    index.synchronized_level = initial_top;
    index.stage_pair(std::move(batch));
    index.armed = true;
    auto insert_staged = [&] {
        vector_array_t placeholder(1, dimension);
        factory_t::add_vertices(index, std::move(placeholder), distance, insert_on_l0, false);
    };
    std::jthread left_worker(insert_staged);
    std::jthread right_worker(insert_staged);
    left_worker.join();
    right_worker.join();
    index.armed = false;

    ASSERT_EQ(index.stale_snapshots.load(), 2u);
    EXPECT_GE(index.refreshed_snapshots.load(), 1u);
    EXPECT_GE(index.completed_publications.load(), 1u);
    const auto a = index.get_highest_level_id(first);
    const auto b = index.get_highest_level_id(first + 1);
    if (absorbed) {
        EXPECT_EQ(std::min(a, b), initial_top);
        EXPECT_EQ(std::max(a, b), new_level);
        EXPECT_EQ(index.top_occupied_level_id(), new_level);
        auto [vertices, reverse_map] = index.get_hierarchical_graph().collect_layer_vids(new_level);
        EXPECT_EQ(vertices.size(), 1u);
    } else {
        ASSERT_GE(a, new_level);
        ASSERT_GE(b, new_level);
        const auto left = index.fetch_level_nbrs(first, new_level);
        const auto right = index.fetch_level_nbrs(first + 1, new_level);
        ASSERT_FALSE(left[0].is_invalid());
        ASSERT_FALSE(right[0].is_invalid());
        EXPECT_EQ(left[0].get_vid(), first + 1);
        EXPECT_EQ(right[0].get_vid(), first);
    }
}

TEST(GraphBuildRegressions, ConcurrentLayerCreatorsRetryAndConnect) {
    if (tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism) < 2)
        GTEST_SKIP() << "The controlled race requires two worker threads";
    for (layer_id_t level : {1u, 2u}) {
        for (bool insert_on_l0 : {false, true}) check_concurrent_growth(level, false, insert_on_l0);
    }
}

TEST(GraphBuildRegressions, GrowthRetryRecomputesTheCoveringLevel) {
    if (tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism) < 2)
        GTEST_SKIP() << "The controlled race requires two worker threads";
    for (layer_id_t level : {1u, 2u}) {
        for (bool insert_on_l0 : {false, true}) check_concurrent_growth(level, true, insert_on_l0);
    }
}

TEST(GraphBuildRegressions, PreparedTopIsHiddenAndCapturedEntryViewsStayStable) {
    dynamic::hierarchical_graph_t graph(3, 8, 8, 4);
    graph.add_vertices(4);
    EXPECT_TRUE(graph.get_top_level_view().vids.empty());
    graph.assign_layer(0, 1);
    const auto old_view = graph.get_top_level_view();
    {
        auto growth = graph.acquire_layer_growth_lock();
        graph.prepare_layer(1, 2);
        EXPECT_EQ(graph.top_occupied_level_id(), 1u);
        EXPECT_EQ(graph.get_top_level_view().vids.front(), 0u);
        graph.fetch_level_nbrs(1, 1)[0] = nbr_t(0, 1.0f);
        graph.publish_layer(1);
    }
    const auto new_view = graph.get_top_level_view();
    ASSERT_EQ(new_view.top_level_id, 2u);
    ASSERT_EQ(new_view.vids.size(), 1u);
    EXPECT_EQ(new_view.vids.front(), 1u);
    EXPECT_EQ(graph.fetch_level_nbrs(new_view.vids.front(), 1)[0].get_vid(), 0u);
    graph.assign_layer(2, 2);
    EXPECT_EQ(graph.get_top_level_view().vids.size(), 2u);
    EXPECT_EQ(new_view.vids.size(), 1u);  // Same-layer appends cannot extend a captured prefix.
    EXPECT_EQ(old_view.top_level_id, 1u);
    EXPECT_EQ(old_view.vids.size(), 1u);
    EXPECT_EQ(old_view.vids.front(), 0u);  // Promotion preserves the old entry range.
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const char* threads = std::getenv("OMP_NUM_THREADS");
    tbb::global_control concurrency(tbb::global_control::max_allowed_parallelism,
        threads ? std::stoul(threads) : tbb::info::default_concurrency());
    return RUN_ALL_TESTS();
}
