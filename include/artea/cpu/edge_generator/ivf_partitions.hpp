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
 * @FilePath: /Artea/include/artea/cpu/edge_generator/ivf_partitions.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: IVF (Inverted File) partitions using CSR format for efficient partition-based operations.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Policy for IVF partitions construction strategy.
 */
enum class IVFConstructPolicyT {
    serial,      ///< Serial construction (single-threaded)
    parallel     ///< Parallel construction (multi-threaded with TBB)
};

/**
 * @brief IVF partitions using CSR (Compressed Sparse Row) format.
 * @tparam EdgeGeneratorTraitsT The edge generator traits type.
 *
 * @details CSR Structure:
 * - Stores vertex IDs grouped by partition ID
 * - _partition_offsets: CSR offsets array (size = num_partitions + 1)
 *   - _partition_offsets[i] = start offset of partition i's vertices
 *   - _partition_offsets[i+1] = end offset of partition i's vertices
 * - _partition_vids: Flattened array storing all vertex IDs
 *
 * Example with 3 partitions:
 *   Partition 0 has vertices [5, 12, 23]
 *   Partition 1 has vertices [1, 7]
 *   Partition 2 has vertices [3, 9, 15, 20]
 *
 *   _partition_vids = [5, 12, 23, 1, 7, 3, 9, 15, 20]
 *   _partition_offsets = [0, 3, 5, 9]
 */
template <typename EdgeGeneratorTraitsT>
class IVFPartitions {
    using vertex_id_t = typename EdgeGeneratorTraitsT::vertex_id_t;
    using vertex_num_t = typename EdgeGeneratorTraitsT::vertex_num_t;
    using part_id_t = typename EdgeGeneratorTraitsT::part_id_t;
    using part_num_t = typename EdgeGeneratorTraitsT::part_num_t;
    using csr_vids_t = typename EdgeGeneratorTraitsT::csr_vids_t;
    using cache_aligned_offset_t = cache_aligned_container_t<vertex_num_t>;

public:
    IVFPartitions() = default;

    /**
     * @brief Construct IVF partitions from partition assignments using policy-based strategy selection.
     * @tparam IVFPolicy Construction policy (serial or parallel)
     * @tparam ContainerT Container type with vector-like interface (e.g., std::vector, std::span)
     * @param part_ids Container where part_ids[i] is the partition ID for vertex i
     * @param num_partitions Total number of partitions
     */
    template <IVFConstructPolicyT IVFPolicy = IVFConstructPolicyT::parallel, typename ContainerT>
    auto from_partition_ids(const ContainerT& part_ids, const part_num_t num_partitions) -> void {
        if constexpr (IVFPolicy == IVFConstructPolicyT::serial) {
            _from_partition_ids_serial(part_ids, num_partitions);
        } else {
            _from_partition_ids_parallel(part_ids, num_partitions);
        }
    }

    /**
     * @brief Get vertex IDs for a specific partition.
     * @param part_id The partition ID
     * @return A span of vertex IDs in this partition
     */
    __attribute__((always_inline))
    auto get_partition_vids(const part_id_t part_id) const -> std::span<const vertex_id_t> {
        return std::span<const vertex_id_t>(
            _partition_vids.data() + _partition_offsets[part_id],
            _partition_offsets[part_id + 1] - _partition_offsets[part_id]
        );
    }

    /**
     * @brief Get the number of vertices in a specific partition.
     * @param part_id The partition ID
     * @return The number of vertices in this partition
     */
    __attribute__((always_inline))
    auto get_partition_size(const part_id_t part_id) const -> vertex_num_t {
        return _partition_offsets[part_id + 1] - _partition_offsets[part_id];
    }

    /**
     * @brief Get the total number of partitions.
     * @return The number of partitions
     */
    __attribute__((always_inline))
    auto get_num_partitions() const -> part_num_t {
        return static_cast<part_num_t>(_partition_offsets.size() - 1);
    }

    /**
     * @brief Get the partition offsets array.
     * @return Reference to the offsets array
     */
    __attribute__((always_inline))
    auto get_partition_offsets() const -> const cache_aligned_offset_t& {
        return _partition_offsets;
    }

    /**
     * @brief Get the flattened vertex IDs array.
     * @return Reference to the vertex IDs array
     */
    __attribute__((always_inline))
    auto get_partition_vids() const -> const csr_vids_t& {
        return _partition_vids;
    }

private:
    /**
     * @brief Serial implementation: single-threaded CSR construction.
     * @tparam ContainerT Container type with vector-like interface
     * @param part_ids Container where part_ids[i] is the partition ID for vertex i
     * @param num_partitions Total number of partitions
     *
     * @note Simple and straightforward implementation:
     *       1. Count vertices per partition
     *       2. Build offsets (prefix sum)
     *       3. Populate vertex IDs into partitions
     *
     * Time complexity: O(N + P) where N = num_vertices, P = num_partitions
     * Space complexity: O(N + P)
     */
    template <typename ContainerT>
    auto _from_partition_ids_serial(const ContainerT& part_ids, const part_num_t num_partitions) -> void {
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(part_ids.size());

        // Count vertices per partition
        std::vector<vertex_num_t> partition_counts(num_partitions, 0);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            partition_counts[part_ids[i]]++;
        }

        // Build offsets (prefix sum)
        _partition_offsets.resize(num_partitions + 1);
        _partition_offsets[0] = 0;
        for (part_num_t p = 0; p < num_partitions; ++p) {
            _partition_offsets[p + 1] = _partition_offsets[p] + partition_counts[p];
        }

        // Allocate and populate vertex IDs
        _partition_vids.resize(num_vertices);
        std::vector<vertex_num_t> write_positions(_partition_offsets.begin(), _partition_offsets.end() - 1);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            part_id_t p = part_ids[i];
            _partition_vids[write_positions[p]++] = static_cast<vertex_id_t>(i);
        }
    }

    /**
     * @brief Parallel implementation: multi-threaded CSR construction using chunk-based 1D prefix sum.
     * @tparam ContainerT Container type with vector-like interface
     * @param part_ids Container where part_ids[i] is the partition ID for vertex i
     * @param num_partitions Total number of partitions
     *
     * @note This method builds the CSR structure in parallel using 1D prefix sum approach:
     *       Pass 1: Count vertices per partition per chunk (parallel, lock-free)
     *       Pass 2: Compute 1D prefix sum to get write positions for each chunk
     *       Pass 3: Scatter vertex IDs to their partitions (parallel, lock-free)
     *
     * Memory overhead: O(num_chunks * num_partitions) which is negligible compared to O(num_vertices)
     * Peak memory: Exactly 1N (no intermediate buffers, no data duplication)
     */
    template <typename ContainerT>
    auto _from_partition_ids_parallel(const ContainerT& part_ids, const part_num_t num_partitions) -> void {
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(part_ids.size());

        // Divide data into chunks (4x threads for load balancing)
        const uint32_t num_threads = tbb::this_task_arena::max_concurrency();
        const uint32_t num_chunks = num_threads * 4;
        const size_t chunk_size = (num_vertices + num_chunks - 1) / num_chunks;

        // chunk_counts[chunk_idx][p]: number of vertices in chunk chunk_idx belonging to partition p
        std::vector<std::vector<vertex_num_t>> chunk_counts(
            num_chunks,
            std::vector<vertex_num_t>(num_partitions, 0)
        );

        // [Pass 1] Parallel counting: each chunk counts its partition frequencies (lock-free)
        tbb::parallel_for(
            static_cast<uint32_t>(0),
            num_chunks,
            [&](uint32_t chunk_idx) {
                vertex_num_t chunk_start = static_cast<vertex_num_t>(chunk_idx * chunk_size);
                vertex_num_t chunk_end = std::min(static_cast<vertex_num_t>(chunk_start + chunk_size), num_vertices);

                auto& local_counts = chunk_counts[chunk_idx];
                for (vertex_num_t i = chunk_start; i < chunk_end; ++i) {
                    local_counts[part_ids[i]]++;
                }
            }
        );

        // Compute global offsets and per-chunk write positions (1D prefix sum)
        _partition_offsets.assign(num_partitions + 1, 0);

        // chunk_write_pos[chunk_idx][p]: write position for chunk chunk_idx writing to partition p
        std::vector<std::vector<vertex_num_t>> chunk_write_pos(
            num_chunks,
            std::vector<vertex_num_t>(num_partitions, 0)
        );

        vertex_num_t accumulated_pos = 0;
        for (part_num_t p = 0; p < num_partitions; ++p) {
            // Record the starting offset for partition p
            _partition_offsets[p] = accumulated_pos;
            // Record the starting write position for each chunk in this partition
            for (uint32_t c = 0; c < num_chunks; ++c) {
                chunk_write_pos[c][p] = accumulated_pos;
                accumulated_pos += chunk_counts[c][p];
            }
        }
        // Final offset
        _partition_offsets[num_partitions] = accumulated_pos;

        // Allocate final global array (single allocation, exact size)
        _partition_vids.resize(num_vertices);

        // [Pass 2] Parallel scatter: each chunk writes vertex IDs to their partitions (lock-free)
        tbb::parallel_for(
            static_cast<uint32_t>(0),
            num_chunks,
            [&](uint32_t chunk_idx) {
                vertex_num_t chunk_start = static_cast<vertex_num_t>(chunk_idx * chunk_size);
                vertex_num_t chunk_end = std::min(static_cast<vertex_num_t>(chunk_start + chunk_size), num_vertices);
                auto local_write_pos = chunk_write_pos[chunk_idx];

                for (vertex_num_t i = chunk_start; i < chunk_end; ++i) {
                    part_id_t p = part_ids[i];
                    vertex_num_t dest_idx = local_write_pos[p]++;
                    _partition_vids[dest_idx] = static_cast<vertex_id_t>(i);
                }
            }
        );
    }

    /** @brief CSR offsets into _partition_vids for each partition (size = num_partitions + 1) */
    cache_aligned_offset_t _partition_offsets;

    /** @brief Flattened array of vertex IDs grouped by partition */
    csr_vids_t _partition_vids;

};  // class IVFPartitions

}   // namespace cpu
}   // namespace artea