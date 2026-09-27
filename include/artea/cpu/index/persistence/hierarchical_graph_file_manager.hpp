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
 * @FilePath: /Artea/include/artea/cpu/index/persistence/hierarchical_graph_file_manager.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Snapshot / restore for compact::HierarchicalGraph.
 *
 *               Single-file binary (.graph) format. All integer fields
 *               are little-endian uint32. Only valid neighbors are
 *               persisted. Restore rebuilds grouped CSR offsets and the global
 *               neighbor array without changing the version-1 wire format.
 *
 *               Wire format (no padding):
 *                 [Header — 7 × uint32 = 28 bytes]
 *                   magic              : uint32 = 0x48475241  ('HGRA')
 *                   version            : uint32 = 1
 *                   num_vertices       : uint32
 *                   top_level_id       : uint32 (0 for an empty graph)
 *                   ul_max_nbr_size    : uint32
 *                   bl_max_nbr_size    : uint32
 *                   entry_point_vid    : uint32
 *                 [Per-vid records, vid = 0 .. num_vertices-1]
 *                   highest_level_id   : uint32
 *                   if highest_level_id == invalid_level_id:
 *                       (no further payload for this vid)
 *                   else for level_id = highest_level_id down to 0:
 *                       valid_nbr_count : uint32
 *                       valid_nbr_vids[valid_nbr_count] : uint32 each
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief File manager for compact::HierarchicalGraph snapshot and restore.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class HierarchicalGraphFileManager {

    using vertex_num_t  = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t   = typename IndexTraitsT::vertex_id_t;
    using layer_num_t   = typename IndexTraitsT::layer_num_t;
    using layer_id_t    = typename IndexTraitsT::layer_id_t;
    using compact_hg_t  = typename IndexTraitsT::compact::hierarchical_graph_t;

    static_assert(std::is_same_v<vertex_id_t,  uint32_t> &&
                  std::is_same_v<vertex_num_t, uint32_t> &&
                  std::is_same_v<layer_id_t,   uint32_t> &&
                  std::is_same_v<layer_num_t,  uint32_t>,
                  "Wire format assumes all id / count integer fields are "
                  "uint32_t. If any widens, bump k_file_version and rev "
                  "the wire format.");

public:
    /**
     * @brief Snapshot a compact hierarchical graph to a single binary file.
     * @param compact_hg Source compact graph.
     * @param bin_path   Destination file path. Parent dirs are created.
     */
    static auto snapshot(
        const compact_hg_t& compact_hg,
        const std::string&  bin_path
    ) -> void {
        std::error_code ec;
        std::filesystem::create_directories(
            std::filesystem::path(bin_path).parent_path(), ec);

        std::ofstream ofs(bin_path, std::ios::binary | std::ios::trunc);
        if (!ofs.is_open()) {
            ARTEA_ERROR(fmt::format("HierarchicalGraphFileManager::snapshot: failed to open {}", bin_path));
        }

        const uint32_t     magic              = k_file_magic;
        const uint32_t     version            = k_file_version;
        const vertex_num_t num_vertices       = compact_hg.get_num_vertices();
        // Empty graphs keep the version-1 header's L0 placeholder instead of the invalid level sentinel.
        const layer_id_t   top_level_id =
            num_vertices == 0 ? layer_id_t{0} : compact_hg.top_occupied_level_id();
        const vertex_num_t ul_max_nbr_size    = compact_hg.ul_max_nbr_size();
        const vertex_num_t bl_max_nbr_size    = compact_hg.bl_max_nbr_size();
        const vertex_id_t  entry_point_vid    = compact_hg.entry_point_vid();

        ofs.write(reinterpret_cast<const char*>(&magic),              sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&version),            sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&num_vertices),       sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&top_level_id),       sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&ul_max_nbr_size),    sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&bl_max_nbr_size),    sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&entry_point_vid),    sizeof(uint32_t));

        constexpr layer_id_t  unassigned_highest_level_sentinel = compact_hg_t::invalid_level_id;

        for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
            const layer_id_t highest_level_id = compact_hg.get_highest_level_id(vid);
            ofs.write(reinterpret_cast<const char*>(&highest_level_id), sizeof(uint32_t));
            if (highest_level_id == unassigned_highest_level_sentinel) continue;

            // Walk levels [highest_level_id .. 0], preserving the version-1 record order.
            for (layer_id_t level_id = highest_level_id; ; --level_id) {
                auto level_nbrs = compact_hg.fetch_level_nbrs(vid, level_id);
                const uint32_t valid_nbr_count = static_cast<uint32_t>(level_nbrs.size());
                ofs.write(reinterpret_cast<const char*>(&valid_nbr_count),
                          sizeof(uint32_t));
                if (valid_nbr_count > 0) {
                    ofs.write(reinterpret_cast<const char*>(level_nbrs.data()), 
                        static_cast<std::streamsize>(valid_nbr_count * sizeof(vertex_id_t)));
                }
                if (level_id == 0) break;
            }
        }

        if (!ofs.good()) {
            ARTEA_ERROR(fmt::format(
                "HierarchicalGraphFileManager::snapshot: write failure on {}",
                bin_path));
        }
    }

    /**
     * @brief Restore a compact hierarchical graph from a snapshot file.
     * @param bin_path Source file path.
     * @return A freshly-constructed compact::HierarchicalGraph.
     */
    static auto restore(const std::string& bin_path) -> compact_hg_t {
        std::ifstream ifs(bin_path, std::ios::binary);
        if (!ifs.is_open()) {
            ARTEA_ERROR(fmt::format(
                "HierarchicalGraphFileManager::restore: failed to open {}",
                bin_path));
        }

        uint32_t     magic              = 0;
        uint32_t     version            = 0;
        vertex_num_t num_vertices       = 0;
        layer_id_t   top_level_id       = 0;
        vertex_num_t ul_max_nbr_size    = 0;
        vertex_num_t bl_max_nbr_size    = 0;
        vertex_id_t  entry_point_vid    = 0;

        ifs.read(reinterpret_cast<char*>(&magic),              sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&version),            sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&num_vertices),       sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&top_level_id),       sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&ul_max_nbr_size),    sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&bl_max_nbr_size),    sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&entry_point_vid),    sizeof(uint32_t));

        if (!ifs.good()) {
            ARTEA_ERROR(fmt::format(
                "HierarchicalGraphFileManager::restore: header read failed: {}",
                bin_path));
        }
        if (magic != k_file_magic) {
            ARTEA_ERROR(fmt::format(
                "HierarchicalGraphFileManager::restore: bad magic {:#x} in {}",
                magic, bin_path));
        }
        if (version != k_file_version) {
            ARTEA_ERROR(fmt::format(
                "HierarchicalGraphFileManager::restore: unsupported "
                "version {} in {}", version, bin_path));
        }

        if (top_level_id == compact_hg_t::invalid_level_id ||
            (entry_point_vid != compact_hg_t::invalid_vertex_id && entry_point_vid >= num_vertices)) {
            ARTEA_ERROR(fmt::format("HierarchicalGraphFileManager::restore: invalid header in {}", bin_path));
        }

        const std::size_t num_groups =
            static_cast<std::size_t>(top_level_id) + 1;

        // Empty graph: one end offset per empty group, no neighbor IDs.
        if (num_vertices == 0) {
            std::vector<std::size_t> empty_group_counts(num_groups, 0);
            compact_hg_t compact_hg(top_level_id,
                                    ul_max_nbr_size,
                                    bl_max_nbr_size,
                                    num_vertices,
                                    std::move(empty_group_counts));
            compact_hg.set_entry_point_vid(entry_point_vid);
            return compact_hg;
        }

        constexpr layer_id_t unassigned_highest_level_sentinel =
            compact_hg_t::invalid_level_id;

        // Single-pass read into staging buffers.
        std::vector<layer_id_t> highest_level_table(num_vertices);
        std::vector<std::vector<std::vector<vertex_id_t>>>
            per_vid_level_nbrs(num_vertices);
        std::vector<vertex_num_t> highest_level_histogram(num_groups, 0);

        for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
            layer_id_t highest_level_id = 0;
            ifs.read(reinterpret_cast<char*>(&highest_level_id),
                     sizeof(uint32_t));
            if (!ifs.good()) {
                ARTEA_ERROR(fmt::format(
                    "HierarchicalGraphFileManager::restore: truncated at "
                    "vid {} in {}", vid, bin_path));
            }
            highest_level_table[vid] = highest_level_id;
            if (highest_level_id == unassigned_highest_level_sentinel) continue;

            if (highest_level_id > top_level_id) {
                ARTEA_ERROR(fmt::format(
                    "HierarchicalGraphFileManager::restore: vid {} has "
                    "highest_level_id {} > top_level_id {} in {}",
                    vid, highest_level_id, top_level_id, bin_path));
            }

            highest_level_histogram[highest_level_id] += 1;
            per_vid_level_nbrs[vid].resize(
                static_cast<std::size_t>(highest_level_id) + 1);

            for (layer_id_t level_id = highest_level_id; ; --level_id) {
                uint32_t valid_nbr_count = 0;
                ifs.read(reinterpret_cast<char*>(&valid_nbr_count),
                         sizeof(uint32_t));
                if (!ifs.good()) {
                    ARTEA_ERROR(fmt::format(
                        "HierarchicalGraphFileManager::restore: truncated "
                        "at vid {} level {} in {}",
                        vid, level_id, bin_path));
                }
                const vertex_num_t level_capacity =
                    (level_id == 0) ? bl_max_nbr_size : ul_max_nbr_size;
                if (valid_nbr_count > level_capacity) {
                    ARTEA_ERROR(fmt::format(
                        "HierarchicalGraphFileManager::restore: vid {} "
                        "level {} count {} > capacity {} in {}",
                        vid, level_id, valid_nbr_count,
                        level_capacity, bin_path));
                }
                per_vid_level_nbrs[vid][level_id].resize(valid_nbr_count);
                if (valid_nbr_count > 0) {
                    ifs.read(
                        reinterpret_cast<char*>(
                            per_vid_level_nbrs[vid][level_id].data()),
                        static_cast<std::streamsize>(
                            valid_nbr_count * sizeof(vertex_id_t)));
                    if (!ifs.good()) {
                        ARTEA_ERROR(fmt::format(
                            "HierarchicalGraphFileManager::restore: "
                            "truncated reading nbrs of vid {} level {} in {}",
                            vid, level_id, bin_path));
                    }
                }
                for (const vertex_id_t neighbor_vid : per_vid_level_nbrs[vid][level_id]) {
                    if (neighbor_vid >= num_vertices) {
                        ARTEA_ERROR(fmt::format(
                            "HierarchicalGraphFileManager::restore: invalid neighbor {} in {}",
                            neighbor_vid, bin_path));
                    }
                }
                if (level_id == 0) break;
            }
        }

        const std::vector<std::size_t> vertex_counts_per_group(highest_level_histogram.begin(),
                                                               highest_level_histogram.end());
        compact_hg_t compact_hg(top_level_id, ul_max_nbr_size, bl_max_nbr_size,
                               num_vertices, vertex_counts_per_group);
        auto& buckets = compact_hg.get_vids_by_highest_level_mut();
        auto& vertex_info_table = compact_hg.get_vertex_info_table_mut();
        for (std::size_t highest_level = 0; highest_level < num_groups; ++highest_level) {
            buckets[highest_level].reserve(vertex_counts_per_group[highest_level]);
        }
        for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
            const layer_id_t highest_level = highest_level_table[vid];
            if (highest_level == unassigned_highest_level_sentinel) continue;
            auto& bucket = buckets[highest_level];
            const std::size_t local_vid = bucket.size();
            vertex_info_table[vid] = {highest_level, static_cast<vertex_id_t>(local_vid)};
            bucket.push_back(vid);
            auto row_counts = compact_hg.get_nbr_offsets_mut(highest_level);
            for (layer_id_t level_id = 0; level_id <= highest_level; ++level_id) {
                const std::size_t row_index = local_vid * (static_cast<std::size_t>(highest_level) + 1) +
                                              (highest_level - level_id);
                row_counts[row_index] = per_vid_level_nbrs[vid][level_id].size();
            }
        }
        compact_hg.allocate_neighbors_from_row_counts();
        for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
            const layer_id_t highest_level = highest_level_table[vid];
            if (highest_level == unassigned_highest_level_sentinel) continue;
            for (layer_id_t level_id = 0; level_id <= highest_level; ++level_id) {
                const auto& payload = per_vid_level_nbrs[vid][level_id];
                auto destination = compact_hg.fetch_level_nbrs_mut(vid, level_id);
                std::copy(payload.begin(), payload.end(), destination.begin());
            }
        }

        compact_hg.set_entry_point_vid(entry_point_vid);
        return compact_hg;
    }

private:
    static constexpr uint32_t k_file_magic   = 0x48475241u;  // "HGRA"
    static constexpr uint32_t k_file_version = 1;

};  // class HierarchicalGraphFileManager

}   // namespace cpu
}   // namespace artea
