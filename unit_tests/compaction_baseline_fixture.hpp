// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <artea/cpu/framework/type_traits/base_traits.hpp>
#include <artea/cpu/framework/type_traits/index_traits.hpp>
#include <artea/cpu/index/compact_structure/hierarchical_graph.hpp>
#include <artea/cpu/index/compactor/hierarchical_graph_compactor.hpp>
#include <artea/cpu/index/dynamic_structure/hierarchical_graph.hpp>
#include <artea/cpu/index/persistence/hierarchical_graph_file_manager.hpp>

// Shared synthetic input for the baseline and subsequent dense-compaction tests.
// No dataset, SIMD distance kernel, or contiguous dynamic arena is assumed.
namespace compaction_baseline {

using traits_t = artea::cpu::IndexTraits<artea::cpu::BaseTraits<uint32_t, float>>;
using dynamic_graph_t = traits_t::dynamic::hierarchical_graph_t;
using compact_graph_t = traits_t::compact::hierarchical_graph_t;
using compactor_t = artea::cpu::HierarchicalGraphCompactor<traits_t>;
using file_manager_t = artea::cpu::HierarchicalGraphFileManager<traits_t>;
using vid_t = traits_t::vertex_id_t;
using level_t = traits_t::layer_id_t;
using row_t = std::vector<vid_t>;
using topology_t = std::vector<std::vector<row_t>>;
constexpr vid_t invalid_vid = traits_t::invalid_vertex_id;
constexpr std::size_t chunk_slots = artea::cpu::dynamic::LevelGroupArena<traits_t>::slot_chunk_size;
constexpr vid_t upper_neighbors = 3;
constexpr vid_t bottom_neighbors = 5;

struct Case {
    std::string name;
    std::vector<vid_t> apex_counts;
    level_t expected_top;
};

inline auto cases() -> std::vector<Case> {
    const vid_t floor = compactor_t::min_layer_cap;
    return {
        {"single_l0", {1}, 0},
        {"l0_only", {5}, 0},
        {"retained_layers", {5, floor + 1, floor + 1}, 2},
        {"threshold_top", {5, floor + 1, floor}, 2},
        {"multiple_trim", {5, floor + 1, 2, 2}, 1},
        {"trim_to_l0", {5, 3, 2, 2}, 0},
        {"empty_middle_group", {5, 0, floor + 1}, 2},
    };
}

inline auto segmented_case() -> Case {
    return {"segmented_tls_holes", {5, static_cast<vid_t>(traits_t::slots_per_block + 8), 2}, 1};
}

struct SquaredDistance {
    auto operator()(const float* left, const float* right) const -> float {
        const float delta = left[0] - right[0];
        return delta * delta;
    }
};

struct Fixture {
    Case spec;
    dynamic_graph_t source;
    traits_t::vector_array_t vectors;
    std::vector<level_t> highest_levels;
    std::vector<row_t> source_buckets;
    topology_t neighbors;
    vid_t expected_entry = invalid_vid;

    explicit Fixture(Case input, bool tls_holes = false)
        : spec(std::move(input)),
          source(static_cast<level_t>(spec.apex_counts.size() - 1),
                 upper_neighbors, bottom_neighbors, 1),
          vectors(std::accumulate(spec.apex_counts.begin(), spec.apex_counts.end(), vid_t{0}), 1),
          highest_levels(vectors.get_num_vecs()),
          source_buckets(spec.apex_counts.size()), neighbors(vectors.get_num_vecs()) {
        source.add_vertices(vectors.get_num_vecs());
        vid_t begin = 0;
        for (level_t h = 0; h < spec.apex_counts.size(); ++h) {
            const vid_t end = begin + spec.apex_counts[h];
            auto assign = [&](vid_t vid) { source.assign_layer(vid, h); };
            if (tls_holes && h == 1) {
                // A distinct thread consumes five of its eight reserved slots.
                // Joining freezes its unused tail before the main thread claims
                // enough slots to cross a data-block boundary. No scheduling
                // race or fixed performance thread limit is needed.
                std::jthread worker([&] {
                    for (vid_t vid = begin + 5; vid > begin; ) assign(--vid);
                });
                worker.join();
                for (vid_t vid = end; vid > begin + 5; ) assign(--vid);
            } else {
                // Descending IDs make bucket order observably different from
                // restore's ascending-ID order.
                for (vid_t vid = end; vid > begin; ) assign(--vid);
            }
            for (vid_t vid = begin; vid < end; ++vid) highest_levels[vid] = h;
            const auto& bucket = source.get_vids_with_highest_level(h);
            source_buckets[h] = row_t(bucket.begin(), bucket.end());
            begin = end;
        }

        // Final top-bucket coordinates sum exactly to zero, with a unique
        // centroid representative at coordinate zero. All arithmetic here is
        // small integer arithmetic, so parallel reductions do not change it.
        const auto top = expected_buckets().back();
        expected_entry = top.front();
        for (vid_t vid = 0; vid < vectors.get_num_vecs(); ++vid) vectors.get(vid)[0] = float(vid + 1);
        vectors.get(expected_entry)[0] = 0.0f;
        std::size_t i = 1;
        if (top.size() % 2 == 0) {
            vectors.get(top[i++])[0] = 1.0f;
            vectors.get(top[i++])[0] = 2.0f;
            vectors.get(top[i++])[0] = -3.0f;
        }
        for (int coordinate = 4; i < top.size(); ++coordinate) {
            vectors.get(top[i++])[0] = float(coordinate);
            vectors.get(top[i++])[0] = float(-coordinate);
        }

        // Preserve ordered valid prefixes independently of physical offsets.
        // Include empty, full, and partially filled rows at both degree limits.
        for (vid_t vid = 0; vid < vectors.get_num_vecs(); ++vid) {
            neighbors[vid].resize(highest_levels[vid] + 1);
            for (level_t h = 0; h <= highest_levels[vid]; ++h) {
                row_t candidates;
                for (vid_t other = vectors.get_num_vecs(); other > 0; ) {
                    --other;
                    if (other != vid && highest_levels[other] >= h) candidates.push_back(other);
                }
                const std::size_t capacity = h == 0 ? bottom_neighbors : upper_neighbors;
                const std::size_t count = std::min<std::size_t>((vid + h) % (capacity + 1), candidates.size());
                auto row = source.fetch_layer_nbrs(vid, h);
                for (std::size_t j = 0; j < count; ++j) {
                    const vid_t other = candidates[(vid + h + j) % candidates.size()];
                    neighbors[vid][h].push_back(other);
                    row[j] = traits_t::nbr_t(other, float(j + 1));
                }
            }
        }
    }

    auto expected_buckets() const -> std::vector<row_t> {
        std::vector<row_t> buckets(spec.expected_top + 1);
        for (level_t h = 0; h < source_buckets.size(); ++h) {
            auto& bucket = buckets[std::min(h, spec.expected_top)];
            bucket.insert(bucket.end(), source_buckets[h].begin(), source_buckets[h].end());
        }
        return buckets;
    }

    auto compact() const -> compact_graph_t {
        return compactor_t::compact_graph(source, vectors, SquaredDistance{});
    }
};

inline auto slot_size(level_t h) -> std::size_t {
    return bottom_neighbors + std::size_t(h) * upper_neighbors;
}

// Read via the public neighbor view; do not require a new layout accessor.
inline auto compact_offset(compact_graph_t& graph, vid_t vid) -> std::size_t {
    const level_t h = graph.get_highest_level_id(vid);
    return graph.fetch_layer_nbrs(vid, h).data() - graph.arena_base(h);
}

}  // namespace compaction_baseline
