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
            const auto row = graph.fetch_level_nbrs(vid, h);
            const auto& expected = input.neighbors[vid][h];
            ASSERT_EQ(row.size(), expected.size());
            EXPECT_EQ(graph.num_valid_nbrs(vid, h), expected.size());
            for (std::size_t j = 0; j < row.size(); ++j) {
                EXPECT_EQ(row[j], expected[j]);
            }
        }
        // Compaction must leave the frozen source usable, including rows at
        // levels subsequently trimmed from the compact graph.
        EXPECT_EQ(input.source.get_highest_level_id(vid), input.highest_levels[vid]);
        for (level_t h = 0; h <= input.highest_levels[vid]; ++h) {
            const auto row = input.source.fetch_level_nbrs(vid, h);
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
        EXPECT_LE(claimed, input.source.get_slot_capacity_in_level(h));
        if (bucket.empty()) EXPECT_EQ(claimed, 0u);
    }
}

auto csr_layout(const compact_graph_t& graph) -> ::testing::AssertionResult {
    const level_t top = graph.get_num_vertices() == 0 ? 0 : graph.top_occupied_level_id();
    std::vector<bool> seen(graph.get_num_vertices(), false);
    std::size_t assigned_count = 0;
    std::uint64_t global_offset = 0;
    for (level_t highest_level = 0; highest_level <= top; ++highest_level) {
        const auto bucket = graph.get_vids_with_highest_level(highest_level);
        const auto offsets = graph.get_nbr_offsets(highest_level);
        if (offsets.size() != bucket.size() * (std::size_t(highest_level) + 1) + 1 ||
            offsets.front() != global_offset) {
            return ::testing::AssertionFailure() << "incorrect CSR row count/start: group=" << highest_level;
        }
        for (std::size_t local_vid = 0; local_vid < bucket.size(); ++local_vid) {
            const vid_t vid = bucket[local_vid];
            if (vid >= seen.size() || seen[vid]) {
                return ::testing::AssertionFailure() << "invalid or duplicated bucket vid=" << vid;
            }
            seen[vid] = true;
            ++assigned_count;
            const auto& info = graph.get_vertex_info(vid);
            if (info.highest_level != highest_level || info.local_vid != local_vid) {
                return ::testing::AssertionFailure() << "incorrect vertex metadata: vid=" << vid;
            }
            for (level_t level_id = highest_level; ; --level_id) {
                const std::size_t row = local_vid * (std::size_t(highest_level) + 1) +
                                        highest_level - level_id;
                if (offsets[row] != global_offset || offsets[row + 1] < offsets[row] ||
                    offsets[row + 1] > graph.neighbor_ids().size()) {
                    return ::testing::AssertionFailure() << "invalid CSR offsets: vid=" << vid;
                }
                const auto neighbors = graph.fetch_level_nbrs(vid, level_id);
                if (neighbors.size() != offsets[row + 1] - offsets[row]) {
                    return ::testing::AssertionFailure() << "incorrect CSR span: vid=" << vid;
                }
                for (const vid_t neighbor : neighbors) {
                    if (neighbor >= graph.get_num_vertices()) {
                        return ::testing::AssertionFailure() << "invalid/sentinel neighbor=" << neighbor;
                    }
                }
                global_offset = offsets[row + 1];
                if (level_id == 0) break;
            }
        }
    }
    if (assigned_count != graph.get_num_vertices() || global_offset != graph.neighbor_ids().size()) {
        return ::testing::AssertionFailure() << "missing vertices or unused neighbor storage";
    }
    return ::testing::AssertionSuccess();
}

class CompactionTopology : public ::testing::TestWithParam<Case> {};

TEST_P(CompactionTopology, PreservesFrozenLogicalGraph) {
    const Fixture input(GetParam());
    auto graph = input.compact();
    ASSERT_NO_FATAL_FAILURE(expect_topology(input, graph));
    ASSERT_TRUE(csr_layout(graph));
}

INSTANTIATE_TEST_SUITE_P(Synthetic, CompactionTopology, ::testing::ValuesIn(cases()),
    [](const ::testing::TestParamInfo<Case>& info) { return info.param.name; });

TEST(CompactionBaseline, EmptyGraphHasNoEntryOrOccupiedLayer) {
    const dynamic_graph_t source(3, upper_neighbors, bottom_neighbors, 0);
    const traits_t::vector_array_t vectors(0, 1);
    auto graph = compactor_t::compact_graph(source, vectors, SquaredDistance{});
    EXPECT_EQ(graph.get_num_vertices(), 0u);
    EXPECT_EQ(graph.top_occupied_level_id(), compact_graph_t::invalid_level_id);
    EXPECT_EQ(graph.entry_point_vid(), invalid_vid);
    EXPECT_TRUE(graph.get_top_level_vids().empty());
    EXPECT_TRUE(graph.get_vids_with_highest_level(0).empty());
    EXPECT_TRUE(csr_layout(graph));
}

TEST(CompactionBaseline, SegmentedSourceHolesDoNotSurviveCompaction) {
    const Fixture input(segmented_case(), true);
    auto graph = input.compact();
    ASSERT_NO_FATAL_FAILURE(expect_topology(input, graph));
    const std::size_t actual = input.source_buckets[1].size();
    const std::size_t claimed = input.source.get_num_claimed_slots_in_arena(1);
    const std::size_t capacity = input.source.get_slot_capacity_in_level(1);
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

    ASSERT_TRUE(csr_layout(graph));
    EXPECT_EQ(graph.get_nbr_offsets(0).size(), input.source_buckets[0].size() + 1);
    EXPECT_EQ(graph.get_nbr_offsets(1).size(), (actual + input.source_buckets[2].size()) * 2 + 1);
    const auto& demoted = input.source_buckets[2];
    for (std::size_t i = 0; i < demoted.size(); ++i) {
        EXPECT_EQ(graph.get_vertex_info(demoted[i]).local_vid, actual + i);
    }
}

TEST(CompactionBaseline, CsrRowsFollowFinalBucketOrder) {
    const Fixture input(segmented_case(), true);
    auto graph = input.compact();
    ASSERT_TRUE(csr_layout(graph));
}

TEST(CompactionBaseline, ExceptionLeavesSourceUsableForAnotherCompaction) {
    const Fixture input(segmented_case(), true);
    const auto throwing_distance = [](const float*, const float*) -> float {
        throw std::runtime_error("injected centroid distance failure");
    };
    EXPECT_THROW(compactor_t::compact_graph(input.source, input.vectors, throwing_distance),
                 std::runtime_error);
    // A failure after target allocation/transcription must not consume or
    // corrupt the source. ASan/LSan also checks cleanup of the failed target.
    auto graph = input.compact();
    ASSERT_NO_FATAL_FAILURE(expect_topology(input, graph));
    ASSERT_TRUE(csr_layout(graph));
}

// Round-trip files live only in the repository's ignored validation directory.
auto validation_file(const std::string& name) -> std::filesystem::path {
    const auto directory = std::filesystem::path("temp/validation/compact-csr/unit-tests");
    std::filesystem::create_directories(directory);
    return directory / name;
}

auto binary_contents(const std::filesystem::path& path) -> std::string {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

TEST_P(CompactionTopology, VersionOneRoundTripPreservesNeighborOrderAndBytes) {
    const Fixture input(GetParam());
    auto graph = input.compact();
    const auto original_path = validation_file(GetParam().name + ".original.graph");
    const auto restored_path = validation_file(GetParam().name + ".restored.graph");
    file_manager_t::snapshot(graph, original_path.string());
    auto restored = file_manager_t::restore(original_path.string());
    ASSERT_TRUE(csr_layout(restored));
    EXPECT_EQ(restored.entry_point_vid(), graph.entry_point_vid());
    for (vid_t vid = 0; vid < graph.get_num_vertices(); ++vid) {
        ASSERT_EQ(restored.get_highest_level_id(vid), graph.get_highest_level_id(vid));
        for (level_t level = 0; level <= graph.get_highest_level_id(vid); ++level) {
            const auto expected = graph.fetch_level_nbrs(vid, level);
            const auto actual = restored.fetch_level_nbrs(vid, level);
            EXPECT_EQ(row_t(actual.begin(), actual.end()), row_t(expected.begin(), expected.end()));
        }
    }
    file_manager_t::snapshot(restored, restored_path.string());
    EXPECT_EQ(binary_contents(original_path), binary_contents(restored_path));
}

TEST(CompactCsr, ZeroNeighborsAndEmptyGroupHaveValidOffsets) {
    compact_graph_t graph(2, 3, 5, 2, {1, 0, 1});
    graph.get_vids_by_highest_level_mut() = {{1}, {}, {0}};
    graph.get_vertex_info_table_mut()[0] = {2, 0};
    graph.get_vertex_info_table_mut()[1] = {0, 0};
    graph.allocate_neighbors_from_row_counts();
    ASSERT_TRUE(csr_layout(graph));
    EXPECT_TRUE(graph.neighbor_ids().empty());
    EXPECT_EQ(graph.get_nbr_offsets(1).size(), 1u);
    for (level_t level = 0; level <= 2; ++level) EXPECT_TRUE(graph.fetch_level_nbrs(0, level).empty());
}

TEST(CompactCsr, CompactsNonemptySourceWithNoEdges) {
    constexpr vid_t vertex_count = 65;
    dynamic_graph_t source(1, upper_neighbors, bottom_neighbors, vertex_count);
    source.add_vertices(vertex_count);
    traits_t::vector_array_t vectors(vertex_count, 1);
    for (vid_t vid = 0; vid < vertex_count; ++vid) {
        source.assign_layer(vid, vid < compactor_t::min_layer_cap ? 1 : 0);
        vectors.get(vid)[0] = static_cast<float>(vid);
    }
    auto graph = compactor_t::compact_graph(source, vectors, SquaredDistance{});
    ASSERT_TRUE(csr_layout(graph));
    EXPECT_EQ(graph.top_occupied_level_id(), 1u);
    EXPECT_LT(graph.entry_point_vid(), compactor_t::min_layer_cap);
    EXPECT_TRUE(graph.neighbor_ids().empty());
    for (vid_t vid = 0; vid < vertex_count; ++vid) {
        for (level_t level = 0; level <= graph.get_highest_level_id(vid); ++level) {
            EXPECT_TRUE(graph.fetch_level_nbrs(vid, level).empty());
        }
    }
}

TEST(CompactCsr, RejectsInvalidCountsAndNeighborAllocationOverflow) {
    EXPECT_THROW(compact_graph_t(1, 3, 5, 2, {2}), std::invalid_argument);
    EXPECT_THROW(compact_graph_t(0, 3, 5, 2, {3}), std::length_error);
    compact_graph_t graph(0, 3, 5, 1, {1});
    graph.get_nbr_offsets_mut(0)[0] = std::numeric_limits<std::uint64_t>::max();
    EXPECT_THROW(graph.allocate_neighbors_from_row_counts(), std::length_error);
    compact_graph_t sum_overflow(0, 3, 5, 2, {2});
    const auto half_limit = artea::cpu::cache_aligned_container_t<vid_t>{}.max_size() / 2;
    sum_overflow.get_nbr_offsets_mut(0)[0] = half_limit + 1;
    sum_overflow.get_nbr_offsets_mut(0)[1] = half_limit + 1;
    EXPECT_THROW(sum_overflow.allocate_neighbors_from_row_counts(), std::length_error);
    static_assert(sizeof(compact_graph_t::VertexInfo) == 8);
}

TEST(CompactCsr, EmptyAndUnassignedRecordsRoundTrip) {
    for (const vid_t vertex_count : {0u, 3u}) {
        compact_graph_t graph(0, 3, 5, vertex_count, {0});
        graph.allocate_neighbors_from_row_counts();
        const auto path = validation_file("unassigned-" + std::to_string(vertex_count) + ".graph");
        file_manager_t::snapshot(graph, path.string());
        auto restored = file_manager_t::restore(path.string());
        EXPECT_EQ(restored.get_num_vertices(), vertex_count);
        EXPECT_TRUE(restored.neighbor_ids().empty());
        EXPECT_EQ(restored.entry_point_vid(), invalid_vid);
        for (vid_t vid = 0; vid < vertex_count; ++vid) {
            EXPECT_EQ(restored.get_vertex_info(vid).highest_level, compact_graph_t::invalid_level_id);
            EXPECT_EQ(restored.get_vertex_info(vid).local_vid, invalid_vid);
        }
    }
}

TEST(CompactCsr, RejectsCorruptAndTruncatedVersionOneFiles) {
    // One L0 vertex with a self edge; each field is a version-1 uint32 word.
    const std::vector<std::uint32_t> valid = {0x48475241u, 1, 1, 0, 3, 5, 0, 0, 1, 0};
    std::vector<std::vector<std::uint32_t>> corrupt;
    for (const auto [word, replacement] : std::vector<std::pair<std::size_t, std::uint32_t>>{
             {0, 0}, {1, 2}, {3, invalid_vid}, {6, 1}, {7, 1}, {8, 6}, {9, invalid_vid}}) {
        corrupt.push_back(valid);
        corrupt.back()[word] = replacement;
    }
    for (std::size_t words = 0; words < valid.size(); ++words) {
        corrupt.emplace_back(valid.begin(), valid.begin() + words);
    }
    const auto path = validation_file("corrupt.graph");
    for (const auto& words : corrupt) {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(std::uint32_t));
        output.close();
        EXPECT_THROW(file_manager_t::restore(path.string()), std::runtime_error);
    }
}

TEST(CompactCsr, RestoresRetainedPreCsrVersionOneSamplesWhenRequested) {
    const char* legacy_directory = std::getenv("ARTEA_COMPACTION_LEGACY_DIR");
    if (legacy_directory == nullptr) GTEST_SKIP() << "Set ARTEA_COMPACTION_LEGACY_DIR for legacy files";
    auto specs = cases();
    specs.push_back(segmented_case());
    for (const auto& spec : specs) {
        const Fixture input(spec, spec.name == "segmented_tls_holes");
        const auto legacy_path = std::filesystem::path(legacy_directory) / (spec.name + ".v1.graph");
        auto restored = file_manager_t::restore(legacy_path.string());
        ASSERT_TRUE(csr_layout(restored));
        EXPECT_EQ(restored.entry_point_vid(), input.expected_entry);
        for (vid_t vid = 0; vid < restored.get_num_vertices(); ++vid) {
            const level_t highest = std::min(input.highest_levels[vid], spec.expected_top);
            ASSERT_EQ(restored.get_highest_level_id(vid), highest);
            for (level_t level = 0; level <= highest; ++level) {
                const auto neighbors = restored.fetch_level_nbrs(vid, level);
                EXPECT_EQ(row_t(neighbors.begin(), neighbors.end()), input.neighbors[vid][level]);
            }
        }
        const auto roundtrip_path = validation_file(spec.name + ".legacy-roundtrip.graph");
        file_manager_t::snapshot(restored, roundtrip_path.string());
        EXPECT_EQ(binary_contents(legacy_path), binary_contents(roundtrip_path));
    }
}

// Opt-in artifact capture: normal test runs do not write snapshots. Point the
// directory at temp/validation/compact-capacity/<experiment>/<run> to retain
// version-1 samples and the current logical/physical layout. Use a fresh path
// after step 01 so the retained pre-repair samples are never overwritten.
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
        ASSERT_TRUE(csr_layout(graph));
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
            {"source_groups", nlohmann::json::array()}, {"compact_groups", nlohmann::json::array()},
            {"vertices", nlohmann::json::array()},
            {"csr_rows_in_bucket_order", bool(csr_layout(graph))},
        };
        for (level_t h = 0; h < spec.apex_counts.size(); ++h) {
            report["source_groups"].push_back({
                {"highest_level", h}, {"actual_vertices", input.source_buckets[h].size()},
                {"capacity_slots", input.source.get_slot_capacity_in_level(h)},
                {"claimed_high_water_slots", input.source.get_num_claimed_slots_in_arena(h)},
            });
        }
        for (level_t h = 0; h <= spec.expected_top; ++h) {
            report["compact_groups"].push_back({
                {"highest_level", h}, {"actual_vertices", graph.get_vids_with_highest_level(h).size()},
                {"offset_elements", graph.get_nbr_offsets(h).size()},
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
