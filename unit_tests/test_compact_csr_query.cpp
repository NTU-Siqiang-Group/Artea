// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <filesystem>
#include <numeric>
#include <gtest/gtest.h>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea::cpu;

namespace {
constexpr auto metric = DistanceMetricsT::EUCLIDEAN_SQR;
constexpr vec_dim_t dimension = 128;
constexpr vertex_num_t vertex_count = 80;
constexpr vertex_num_t topk = 10;

// Fully connected rows allow an independent exact top-10 reference, including tied distances.
template <bool UpperBeam>
void check_queries() {
    vector_array_t vectors(vertex_count, dimension);
    dynamic::hierarchical_graph_t source(2, vertex_count, vertex_count, vertex_count);
    source.add_vertices(vertex_count);
    dist_func_t<metric, dimension> distance;
    for (vertex_id_t vid = 0; vid < vertex_count; ++vid) {
        std::fill_n(vectors.get(vid), dimension, 0.0f);
        vectors.get(vid)[0] = static_cast<float>(vid);
        source.assign_layer(vid, vid < 40 ? 1 : 0);
        for (layer_id_t level = 0; level <= source.get_highest_level_id(vid); ++level) {
            auto neighbors = source.fetch_level_nbrs(vid, level);
            const vertex_num_t participant_count = level == 0 ? vertex_count : 40;
            for (vertex_id_t neighbor = 0; neighbor < participant_count; ++neighbor) {
                neighbors[neighbor] = nbr_t(neighbor, static_cast<float>(neighbor));
            }
        }
    }
    auto graph = hierarchical_graph_compactor_t::compact_graph(source, vectors, distance);
    ASSERT_EQ(graph.top_occupied_level_id(), 1u);
    const auto directory = std::filesystem::path("temp/validation/compact-csr/query-tests");
    std::filesystem::create_directories(directory);
    const auto snapshot = directory / (UpperBeam ? "beam.graph" : "greedy.graph");
    hierarchical_graph_file_manager_t::snapshot(graph, snapshot.string());
    auto restored = hierarchical_graph_file_manager_t::restore(snapshot.string());
    hierarchical_graph_router_t<metric, dimension> router(vectors, distance, topk, vertex_count);
    router.initialize();
    vector_array_t queries(5, dimension);
    const std::array<float, 5> query_coordinates = {-2.0f, 12.25f, 39.5f, 60.75f, 90.0f};
    for (std::size_t query_index = 0; query_index < query_coordinates.size(); ++query_index) {
        auto* query = queries.get(static_cast<vertex_id_t>(query_index));
        std::fill_n(query, dimension, 0.0f);
        query[0] = query_coordinates[query_index];
        std::vector<vertex_id_t> expected(vertex_count);
        std::iota(expected.begin(), expected.end(), 0);
        std::sort(expected.begin(), expected.end(), [&](vertex_id_t left, vertex_id_t right) {
            const float left_distance = distance(query, vectors.get(left));
            const float right_distance = distance(query, vectors.get(right));
            return left_distance < right_distance || (left_distance == right_distance && left < right);
        });
        const auto original_results = router.template query<false, UpperBeam>(query, graph);
        const auto restored_results = router.template query<false, UpperBeam>(query, restored);
        const auto dynamic_results = router.template query<false, UpperBeam>(query, source);
        ASSERT_EQ(original_results.size(), topk);
        ASSERT_EQ(restored_results.size(), topk);
        ASSERT_EQ(dynamic_results.size(), topk);
        for (std::size_t rank = 0; rank < topk; ++rank) {
            EXPECT_EQ(original_results[rank].get_vid(), expected[rank]);
            EXPECT_EQ(restored_results[rank].get_vid(), original_results[rank].get_vid());
            EXPECT_EQ(dynamic_results[rank].get_vid(), original_results[rank].get_vid());
            EXPECT_FLOAT_EQ(original_results[rank].get_distance(),
                            distance(query, vectors.get(expected[rank])));
            EXPECT_FLOAT_EQ(restored_results[rank].get_distance(), original_results[rank].get_distance());
            EXPECT_FLOAT_EQ(dynamic_results[rank].get_distance(), original_results[rank].get_distance());
        }
    }
    const auto batch_results = router.template batch_query<false, UpperBeam>(queries, restored);
    ASSERT_EQ(batch_results.size(), query_coordinates.size() * topk);
    for (std::size_t query_index = 0; query_index < query_coordinates.size(); ++query_index) {
        const auto single_results = router.template query<false, UpperBeam>(queries.get(query_index), graph);
        for (std::size_t rank = 0; rank < topk; ++rank) {
            EXPECT_EQ(batch_results[query_index * topk + rank].get_vid(), single_results[rank].get_vid());
            EXPECT_FLOAT_EQ(batch_results[query_index * topk + rank].get_distance(),
                            single_results[rank].get_distance());
        }
    }
}

TEST(CompactCsrQuery, GreedyTop10MatchesExactReferenceAndRestoredBatch) { check_queries<false>(); }
TEST(CompactCsrQuery, UpperBeamTop10MatchesExactReferenceAndRestoredBatch) { check_queries<true>(); }

template <std::size_t BatchSize>
void check_neighbor_batches() {
    vector_array_t vectors(20, dimension);
    std::fill_n(vectors.get_all(), 20 * dimension, 0.0f);
    const std::array<vertex_id_t, 21> neighbor_ids = {
        2, 0, 3, 3, 1, 5, 7, 5, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    const std::array<vertex_id_t, 17> expected_ids = {
        2, 3, 5, 7, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    const std::array<std::size_t, 4> neighbor_counts = {0, 1, 5, neighbor_ids.size()};
    const std::array<std::size_t, 4> expected_counts = {0, 1, 2, expected_ids.size()};
    for (std::size_t case_index = 0; case_index < neighbor_counts.size(); ++case_index) {
        thread_local_bitmap_t visited(20);
        visited.set(0);
        visited.set(1);
        std::vector<vertex_id_t> actual_ids;
        const auto level_nbrs = std::span<const vertex_id_t>(neighbor_ids).first(neighbor_counts[case_index]);
        detail::process_prefetched_neighbors<BatchSize>(level_nbrs, vectors, visited,
                                                       [&](vertex_id_t vid) { actual_ids.push_back(vid); });
        const std::vector<vertex_id_t> expected(expected_ids.begin(),
                                                expected_ids.begin() + expected_counts[case_index]);
        EXPECT_EQ(actual_ids, expected);
        EXPECT_TRUE(visited.test(0));
        EXPECT_TRUE(visited.test(1));
        for (const vertex_id_t vid : actual_ids) EXPECT_TRUE(visited.test(vid));
    }
}

TEST(CompactCsrQuery, PrefetchBatch4HandlesOrderDuplicatesEmptyAndTail) { check_neighbor_batches<4>(); }
TEST(CompactCsrQuery, PrefetchBatch8HandlesOrderDuplicatesEmptyAndTail) { check_neighbor_batches<8>(); }
TEST(CompactCsrQuery, PrefetchBatch16HandlesOrderDuplicatesEmptyAndTail) { check_neighbor_batches<16>(); }

TEST(CompactCsrQuery, SentinelTerminatedNeighborsStopBeforeInvalidId) {
    vector_array_t vectors(5, dimension);
    thread_local_bitmap_t visited(5);
    const std::array<vertex_id_t, 4> neighbor_ids = {2, 3, base_traits_t::invalid_vertex_id, 4};
    const auto valid_neighbors = std::span<const vertex_id_t>(neighbor_ids) |
        std::views::take_while([](vertex_id_t vid) { return vid != base_traits_t::invalid_vertex_id; });
    std::vector<vertex_id_t> actual_ids;
    detail::process_prefetched_neighbors<8>(valid_neighbors, vectors, visited,
                                           [&](vertex_id_t vid) { actual_ids.push_back(vid); });
    EXPECT_EQ(actual_ids, (std::vector<vertex_id_t>{2, 3}));
    EXPECT_FALSE(visited.test(4));
}

TEST(CompactCsrQuery, NoncontiguousNeighborsKeepScalarVisitedMarkTiming) {
    vector_array_t vectors(5, dimension);
    thread_local_bitmap_t visited(5);
    const std::array<vertex_id_t, 3> neighbor_ids = {2, 3, 4};
    const auto projected_ids = neighbor_ids | std::views::transform([](vertex_id_t vid) { return vid; });
    static_assert(!std::ranges::contiguous_range<decltype(projected_ids)>);
    std::vector<vertex_id_t> actual_ids;
    detail::process_prefetched_neighbors<16>(projected_ids, vectors, visited, [&](vertex_id_t vid) {
        EXPECT_TRUE(visited.test(vid));
        if (vid < 4) EXPECT_FALSE(visited.test(vid + 1));
        actual_ids.push_back(vid);
    });
    EXPECT_EQ(actual_ids, (std::vector<vertex_id_t>{2, 3, 4}));
}
}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
