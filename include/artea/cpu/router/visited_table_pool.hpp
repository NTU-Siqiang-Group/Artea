// Copyright 2025 Weitang Ye
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
 * @FilePath: /Artea/include/artea/cpu/router/visited_table_pool.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <artea/cpu/utils/parallel.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {

/**
 * @brief A pool of thread-local visited tables for parallel graph search.
 *
 * Design Philosophy:
 * During batch_query, each TBB worker thread needs its own visited table
 * to track which vertices have been explored in the current query's beam search.
 * This pool lazily constructs one VisitedTableT per TBB thread via
 * tbb::enumerable_thread_specific, and reuses them across queries by clearing
 * between invocations. No locks, no contention — each thread owns its table.
 *
 * @tparam RouterTraitsT The router traits type, which provides vertex_num_t.
 * @tparam VisitedTableT The visited table type (must satisfy VisitedTable concept).
 */
template <typename RouterTraitsT, VisitedTable VisitedTableT = typename RouterTraitsT::thread_local_bitmap_t>
class VisitedTablePool {

    using visited_table_t = VisitedTableT;
    using vertex_num_t = typename RouterTraitsT::vertex_num_t;

public:

    VisitedTablePool() = default;

    explicit VisitedTablePool(const vertex_num_t num_vertices)
        : _num_vertices(num_vertices),
          _pool([num_vertices]() { return visited_table_t(num_vertices); })
    {}

    /**
     * @brief Acquire this thread's visited table.
     * @return Reference to the calling thread's VisitedTableT.
     */
    __attribute__((always_inline))
    auto acquire() -> visited_table_t& {
        auto& table = _pool.local();
        return table;
    }

    /**
     * @brief Force all TBB worker threads to construct their tables now.
     *
     * Launches a parallel_for with one task per thread slot. Each task calls
     * _pool.local(), which triggers the lazy construction for that thread.
     * Call this once after construction (e.g., in initialize_impl) so that
     * the first real batch_query pays no allocation cost.
     */
    auto warmup() -> void {
        const int num_threads = tbb_max_num_threads();
        tbb::parallel_for(
            tbb::blocked_range<int>(0, num_threads, 1),
            [&](const tbb::blocked_range<int>&) {
                auto& table = _pool.local();
                table.clear();
            }
        );
    }

    /** @brief Get the number of bits each table tracks. */
    __attribute__((always_inline))
    auto num_vertices() const -> vertex_num_t { return _num_vertices; }

private:

    /** @brief Number of vertices (bits) per table. */
    vertex_num_t _num_vertices = 0;

    /** @brief Thread-local table storage. One table per TBB worker thread. */
    tbb::enumerable_thread_specific<visited_table_t> _pool;

};  // class VisitedTablePool

}   // namespace cpu
}   // namespace artea
