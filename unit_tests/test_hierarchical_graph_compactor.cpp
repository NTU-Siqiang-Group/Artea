// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "compaction_baseline_fixture.hpp"

namespace {
using namespace compaction_baseline;

void expect_topology(const Fixture& input, const compact_graph_t& graph) {
    ASSERT_EQ(graph.get_num_vertices(), input.highest_levels.size());
    EXPECT_EQ(graph.top_occupied_level_id(), input.spec.expected_top);
    EXPECT_EQ(graph.ul_max_nbr_size(), upper_neighbors);
    EXPECT_EQ(graph.bl_max_nbr_size(), bottom_neighbors);
    EXPECT_EQ(graph.entry_point_vid(), input.expected_entry);
    const auto buckets = input.expected_buckets();
    for (level_t h = 0; h < buckets.size(); ++h) {
        const auto actual = graph.get_vids_with_highest_level(h);
        EXPECT_EQ(row_t(actual.begin(), actual.end()), buckets[h]) << "bucket=" << h;
    }
    for (vid_t vid = 0; vid < input.highest_levels.size(); ++vid) {
        SCOPED_TRACE(::testing::Message() << "vid=" << vid);
        const level_t highest = std::min(input.highest_levels[vid], input.spec.expected_top);
        ASSERT_EQ(graph.get_highest_level_id(vid), highest);
        for (level_t h = 0; h <= highest; ++h) {
            SCOPED_TRACE(::testing::Message() << "level=" << h);
            const auto row = graph.fetch_layer_nbrs(vid, h);
            const auto& expected = input.neighbors[vid][h];
            ASSERT_EQ(row.size(), h == 0 ? bottom_neighbors : upper_neighbors);
            EXPECT_EQ(graph.num_valid_nbrs(vid, h), expected.size());
            for (std::size_t j = 0; j < row.size(); ++j) {
                EXPECT_EQ(row[j], j < expected.size() ? expected[j] : invalid_vid);
            }
        }
        // Compaction must leave the frozen source usable, including rows at
        // levels subsequently trimmed from the compact graph.
        EXPECT_EQ(input.source.get_highest_level_id(vid), input.highest_levels[vid]);
        for (level_t h = 0; h <= input.highest_levels[vid]; ++h) {
            const auto row = input.source.fetch_layer_nbrs(vid, h);
            const auto& expected = input.neighbors[vid][h];
            for (std::size_t j = 0; j < row.size(); ++j) {
                if (j < expected.size()) {
                    EXPECT_FALSE(row[j].is_invalid());
                    EXPECT_EQ(row[j].get_vid(), expected[j]);
                    EXPECT_FLOAT_EQ(row[j].get_distance(), float(j + 1));
                } else {
                    // Dynamic Neighbor packs its invalid marker into the ID;
                    // compact rows instead store the full vertex-ID sentinel.
                    EXPECT_TRUE(row[j].is_invalid());
                }
            }
        }
    }
    for (level_t h = 0; h < input.source_buckets.size(); ++h) {
        const auto& bucket = input.source.get_vids_with_highest_level(h);
        EXPECT_EQ(row_t(bucket.begin(), bucket.end()), input.source_buckets[h]);
        const auto claimed = input.source.get_num_claimed_slots_in_arena(h);
        EXPECT_GE(claimed, bucket.size());
        EXPECT_LE(claimed, input.source.get_arena_capacity_in_arena(h));
        if (bucket.empty()) EXPECT_EQ(claimed, 0u);
    }
}

auto dense_layout(compact_graph_t& graph) -> ::testing::AssertionResult {
    for (level_t h = 0; h <= graph.top_occupied_level_id(); ++h) {
        std::size_t slot = 0;
        for (const vid_t vid : graph.get_vids_with_highest_level(h)) {
            const auto actual = compact_offset(graph, vid);
            const auto expected = slot++ * slot_size(h);
            if (actual != expected) {
                return ::testing::AssertionFailure()
                    << "group=" << h << ", vid=" << vid << ", offset=" << actual
                    << ", dense offset=" << expected;
            }
        }
    }
    return ::testing::AssertionSuccess();
}

class CompactionTopology : public ::testing::TestWithParam<Case> {};

TEST_P(CompactionTopology, PreservesFrozenLogicalGraph) {
    const Fixture input(GetParam());
    const auto graph = input.compact();
    ASSERT_NO_FATAL_FAILURE(expect_topology(input, graph));
}

INSTANTIATE_TEST_SUITE_P(Synthetic, CompactionTopology, ::testing::ValuesIn(cases()),
    [](const ::testing::TestParamInfo<Case>& info) { return info.param.name; });

TEST(CompactionBaseline, EmptyGraphHasNoEntryOrOccupiedLayer) {
    const dynamic_graph_t source(3, upper_neighbors, bottom_neighbors, 0);
    const traits_t::vector_array_t vectors(0, 1);
    const auto graph = compactor_t::compact_graph(source, vectors, SquaredDistance{});
    EXPECT_EQ(graph.get_num_vertices(), 0u);
    EXPECT_EQ(graph.top_occupied_level_id(), compact_graph_t::invalid_level_id);
    EXPECT_EQ(graph.entry_point_vid(), invalid_vid);
    EXPECT_TRUE(graph.get_top_level_vids().empty());
    EXPECT_TRUE(graph.get_vids_with_highest_level(0).empty());
}

TEST(CompactionBaseline, SegmentedSourceRetainsCapacityAndReservationHoles) {
    const Fixture input(segmented_case(), true);
    auto graph = input.compact();
    ASSERT_NO_FATAL_FAILURE(expect_topology(input, graph));
    const std::size_t actual = input.source_buckets[1].size();
    const std::size_t claimed = input.source.get_num_claimed_slots_in_arena(1);
    const std::size_t capacity = input.source.get_arena_capacity_in_arena(1);
    EXPECT_EQ(actual, traits_t::slots_per_block + 8);
    EXPECT_EQ(claimed, traits_t::slots_per_block + 2 * chunk_slots);
    EXPECT_EQ(capacity, 2 * traits_t::slots_per_block);
    EXPECT_GT(claimed, actual);
    EXPECT_GT(capacity, claimed);

    std::vector<std::size_t> source_slots;
    for (const vid_t vid : input.source_buckets[1]) {
        source_slots.push_back(input.source.get_slot_offset(vid) / slot_size(1));
    }
    std::sort(source_slots.begin(), source_slots.end());
    EXPECT_EQ(source_slots[4], 4u);
    EXPECT_EQ(source_slots[5], chunk_slots); // slots 5..7 are unconsumed TLS reservations
    EXPECT_GE(source_slots.back(), traits_t::slots_per_block); // crosses a block boundary

    // These characterize the old layout, not invariants of the logical graph.
    // Replace them with the dense assertions when implementing step 02.
    for (vid_t vid = 0; vid < input.highest_levels.size(); ++vid) {
        if (input.highest_levels[vid] <= input.spec.expected_top) {
            EXPECT_EQ(compact_offset(graph, vid), input.source.get_slot_offset(vid));
        }
    }
    const auto& demoted = input.source_buckets[2];
    for (std::size_t i = 0; i < demoted.size(); ++i) {
        EXPECT_EQ(compact_offset(graph, demoted[i]), (capacity + i) * slot_size(1));
    }
    EXPECT_FALSE(dense_layout(graph));
}

// Intentionally excluded from the normal passing suite. Run it explicitly to
// demonstrate the old layout's failure; enable it after the step-02 repair.
TEST(CompactionBaseline, DISABLED_DenseSlotsFollowFinalBucketOrder) {
    const Fixture input(segmented_case(), true);
    auto graph = input.compact();
    ASSERT_TRUE(dense_layout(graph));
}

// Opt-in artifact capture: normal test runs do not write snapshots. Point the
// directory at temp/validation/compact-capacity/<experiment>/<run> to retain
// version-1 samples and the full logical/physical baseline before step 02.
TEST(CompactionBaseline, CaptureVersionOneSamplesWhenRequested) {
    const char* output = std::getenv("ARTEA_COMPACTION_BASELINE_DIR");
    if (output == nullptr || *output == '\0') {
        GTEST_SKIP() << "Set ARTEA_COMPACTION_BASELINE_DIR to capture pre-repair samples";
    }
    const std::filesystem::path directory(output);
    std::filesystem::create_directories(directory);
    auto specs = cases();
    specs.push_back(segmented_case());
    for (const auto& spec : specs) {
        const Fixture input(spec, spec.name == "segmented_tls_holes");
        auto graph = input.compact();
        ASSERT_NO_FATAL_FAILURE(expect_topology(input, graph));
        const auto snapshot = directory / (spec.name + ".v1.graph");
        file_manager_t::snapshot(graph, snapshot.string());

        std::ifstream binary(snapshot, std::ios::binary);
        uint32_t magic = 0, version = 0;
        binary.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        binary.read(reinterpret_cast<char*>(&version), sizeof(version));
        ASSERT_TRUE(binary.good());
        ASSERT_EQ(magic, 0x48475241u);
        ASSERT_EQ(version, 1u);

        nlohmann::json report = {
            {"case", spec.name}, {"snapshot_version", version},
            {"snapshot", snapshot.filename().string()},
            {"min_layer_cap", compactor_t::min_layer_cap},
            {"slots_per_block", traits_t::slots_per_block}, {"slot_chunk_size", chunk_slots},
            {"ul_max_nbr_size", upper_neighbors}, {"bl_max_nbr_size", bottom_neighbors},
            {"source_apex_counts", spec.apex_counts}, {"source_buckets", input.source_buckets},
            {"final_buckets", input.expected_buckets()}, {"final_top", spec.expected_top},
            {"entry_point_vid", graph.entry_point_vid()}, {"source_neighbors", input.neighbors},
            {"source_groups", nlohmann::json::array()}, {"vertices", nlohmann::json::array()},
            {"dense_slots_in_bucket_order", bool(dense_layout(graph))},
        };
        for (level_t h = 0; h < spec.apex_counts.size(); ++h) {
            report["source_groups"].push_back({
                {"highest_level", h}, {"actual_vertices", input.source_buckets[h].size()},
                {"capacity_slots", input.source.get_arena_capacity_in_arena(h)},
                {"claimed_high_water_slots", input.source.get_num_claimed_slots_in_arena(h)},
            });
        }
        for (vid_t vid = 0; vid < input.highest_levels.size(); ++vid) {
            report["vertices"].push_back({
                {"vid", vid}, {"coordinate", input.vectors.get(vid)[0]},
                {"source_highest_level", input.highest_levels[vid]},
                {"final_highest_level", graph.get_highest_level_id(vid)},
                {"source_offset", input.source.get_slot_offset(vid)},
                {"compact_offset", compact_offset(graph, vid)},
            });
        }
        std::ofstream json(directory / (spec.name + ".baseline.json"));
        json << report.dump(2) << '\n';
        ASSERT_TRUE(json.good());
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
