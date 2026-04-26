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
 *               persisted; trailing sentinel cells in the in-memory
 *               arena are reconstructed at restore time by the
 *               compact::HierarchicalGraph constructor's
 *               std::fill(arena, invalid_vertex_id) pass.
 *
 *               Authoritative reference for the per-vertex in-arena
 *               layout used by snapshot's per-level loop and by
 *               restore's offset arithmetic:
 *                 compact_structure/hierarchical_graph.hpp ::
 *                 HierarchicalGraph::fetch_layer_nbrs.
 *
 *               Wire format (no padding):
 *                 [Header — 7 × uint32 = 28 bytes]
 *                   magic              : uint32 = 0x48475241  ('HGRA')
 *                   version            : uint32 = 1
 *                   num_vertices       : uint32
 *                   max_restrict_level : uint32
 *                   ul_max_nbr_size    : uint32
 *                   bl_max_nbr_size    : uint32
 *                   entry_point_vid    : uint32
 *                 [Per-vid records, vid = 0 .. num_vertices-1]
 *                   highest_level_id   : uint32
 *                   if highest_level_id == unassigned_highest_level_id:
 *                       (no further payload for this vid)
 *                   else for level_id = highest_level_id down to 0:
 *                       valid_nbr_count : uint32
 *                       valid_nbr_vids[valid_nbr_count] : uint32 each
 */

#pragma once

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
        const layer_num_t  max_restrict_level = compact_hg.max_restrict_level();
        const vertex_num_t ul_max_nbr_size    = compact_hg.ul_max_nbr_size();
        const vertex_num_t bl_max_nbr_size    = compact_hg.bl_max_nbr_size();
        const vertex_id_t  entry_point_vid    = compact_hg.entry_point_vid();

        ofs.write(reinterpret_cast<const char*>(&magic),              sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&version),            sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&num_vertices),       sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&max_restrict_level), sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&ul_max_nbr_size),    sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&bl_max_nbr_size),    sizeof(uint32_t));
        ofs.write(reinterpret_cast<const char*>(&entry_point_vid),    sizeof(uint32_t));

        constexpr vertex_id_t invalid_vertex_id_sentinel = compact_hg_t::invalid_vertex_id;
        constexpr layer_id_t  unassigned_highest_level_sentinel = compact_hg_t::unassigned_highest_level_id;

        for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
            const layer_id_t highest_level_id = compact_hg.get_highest_level_id(vid);
            ofs.write(reinterpret_cast<const char*>(&highest_level_id), sizeof(uint32_t));
            if (highest_level_id == unassigned_highest_level_sentinel) continue;

            // Walk levels [highest_level_id .. 0] matching in-arena layout.
            for (layer_id_t level_id = highest_level_id; ; --level_id) {
                auto level_nbrs = compact_hg.fetch_layer_nbrs(vid, level_id);
                uint32_t valid_nbr_count = 0;
                for (vertex_id_t nbr_vid : level_nbrs) {
                    if (nbr_vid == invalid_vertex_id_sentinel) break;
                    ++valid_nbr_count;
                }
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
        layer_num_t  max_restrict_level = 0;
        vertex_num_t ul_max_nbr_size    = 0;
        vertex_num_t bl_max_nbr_size    = 0;
        vertex_id_t  entry_point_vid    = 0;

        ifs.read(reinterpret_cast<char*>(&magic),              sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&version),            sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&num_vertices),       sizeof(uint32_t));
        ifs.read(reinterpret_cast<char*>(&max_restrict_level), sizeof(uint32_t));
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

        const std::size_t num_groups =
            static_cast<std::size_t>(max_restrict_level) + 1;

        // Empty graph: zero-capacity arenas, entry point sentinel.
        if (num_vertices == 0) {
            std::vector<std::size_t> empty_arena_caps(num_groups, 0);
            compact_hg_t compact_hg(max_restrict_level,
                                    ul_max_nbr_size,
                                    bl_max_nbr_size,
                                    num_vertices,
                                    std::move(empty_arena_caps));
            compact_hg.set_entry_point_vid(entry_point_vid);
            return compact_hg;
        }

        constexpr layer_id_t unassigned_highest_level_sentinel =
            compact_hg_t::unassigned_highest_level_id;

        // Single-pass read into staging buffers.
        std::vector<layer_id_t> highest_level_table(num_vertices);
        std::vector<std::vector<std::vector<vertex_id_t>>>
            per_vid_layer_nbrs(num_vertices);
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

            if (highest_level_id > max_restrict_level) {
                ARTEA_ERROR(fmt::format(
                    "HierarchicalGraphFileManager::restore: vid {} has "
                    "highest_level_id {} > max_restrict_level {} in {}",
                    vid, highest_level_id, max_restrict_level, bin_path));
            }

            highest_level_histogram[highest_level_id] += 1;
            per_vid_layer_nbrs[vid].resize(
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
                per_vid_layer_nbrs[vid][level_id].resize(valid_nbr_count);
                if (valid_nbr_count > 0) {
                    ifs.read(
                        reinterpret_cast<char*>(
                            per_vid_layer_nbrs[vid][level_id].data()),
                        static_cast<std::streamsize>(
                            valid_nbr_count * sizeof(vertex_id_t)));
                    if (!ifs.good()) {
                        ARTEA_ERROR(fmt::format(
                            "HierarchicalGraphFileManager::restore: "
                            "truncated reading nbrs of vid {} level {} in {}",
                            vid, level_id, bin_path));
                    }
                }
                if (level_id == 0) break;
            }
        }

        // Derive per-arena capacity from the histogram. Mirrors the
        // compactor's own derivation:
        //   slot_size(group_id) = (group_id == 0)
        //                            ? bl_max_nbr_size
        //                            : group_id * ul_max_nbr_size
        //                              + bl_max_nbr_size
        //   arena_cap[group_id] = histogram[group_id]
        //                         * slot_size(group_id)
        std::vector<std::size_t> arena_vid_capacity_per_group(num_groups, 0);
        for (std::size_t group_id = 0; group_id < num_groups; ++group_id) {
            const std::size_t slot_size = (group_id == 0)
                ? static_cast<std::size_t>(bl_max_nbr_size)
                : (group_id * static_cast<std::size_t>(ul_max_nbr_size)
                   + static_cast<std::size_t>(bl_max_nbr_size));
            arena_vid_capacity_per_group[group_id] =
                static_cast<std::size_t>(highest_level_histogram[group_id])
                * slot_size;
        }

        compact_hg_t compact_hg(max_restrict_level,
                                ul_max_nbr_size,
                                bl_max_nbr_size,
                                num_vertices,
                                std::move(arena_vid_capacity_per_group));

        // Populate vids_by_highest_level by scanning highest_level_table
        // in vid order. The slot index inside arena[group_id] is the
        // position in this list.
        auto& vids_by_highest_level = compact_hg.get_vids_by_highest_level_mut();
        for (std::size_t group_id = 0; group_id < num_groups; ++group_id) {
            vids_by_highest_level[group_id].reserve(
                highest_level_histogram[group_id]);
        }
        for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
            const layer_id_t highest_level_id = highest_level_table[vid];
            if (highest_level_id == unassigned_highest_level_sentinel) continue;
            vids_by_highest_level[highest_level_id].push_back(vid);
        }

        // Write valid prefixes into arenas; trailing cells stay
        // invalid_vertex_id from the constructor's std::fill.
        auto& vertex_info_table = compact_hg.get_vertex_info_table_mut();
        for (std::size_t group_id = 0; group_id < num_groups; ++group_id) {
            const std::size_t slot_size = (group_id == 0)
                ? static_cast<std::size_t>(bl_max_nbr_size)
                : (group_id * static_cast<std::size_t>(ul_max_nbr_size)
                   + static_cast<std::size_t>(bl_max_nbr_size));
            vertex_id_t* arena_base =
                compact_hg.arena_base(static_cast<layer_id_t>(group_id));

            for (std::size_t slot_idx = 0;
                 slot_idx < vids_by_highest_level[group_id].size();
                 ++slot_idx)
            {
                const vertex_id_t vid =
                    vids_by_highest_level[group_id][slot_idx];
                const std::size_t slot_offset = slot_idx * slot_size;

                vertex_info_table[vid].highest_level_id =
                    static_cast<layer_id_t>(group_id);
                vertex_info_table[vid].slot_offset = slot_offset;

                // Per-level write: for level_id in [0..group_id], the
                // in-slot offset is (group_id - level_id) *
                // ul_max_nbr_size; for group_id == 0 the single L0
                // segment sits at offset 0.
                for (layer_id_t level_id = static_cast<layer_id_t>(group_id);
                     ;
                     --level_id)
                {
                    const auto& level_payload =
                        per_vid_layer_nbrs[vid][level_id];
                    if (!level_payload.empty()) {
                        const std::size_t in_slot_offset = (group_id == 0)
                            ? std::size_t{0}
                            : (static_cast<std::size_t>(group_id) - level_id)
                              * static_cast<std::size_t>(ul_max_nbr_size);
                        std::memcpy(
                            arena_base + slot_offset + in_slot_offset,
                            level_payload.data(),
                            level_payload.size() * sizeof(vertex_id_t));
                    }
                    if (level_id == 0) break;
                }
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
