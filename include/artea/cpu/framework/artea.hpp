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
 * @FilePath: /Artea/include/artea/cpu/framework/artea.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

// Including the file is all you need.

#pragma once

/** @note: this file is only provided for project user,
 *         do not include this file in internal implementation. */

#include <artea/common/logger.hpp>

#include <artea/cpu/framework/base_traits.hpp>
#include <artea/cpu/framework/computer_traits.hpp>
#include <artea/cpu/framework/buffer_traits.hpp>
#include <artea/cpu/framework/router_traits.hpp>
#include <artea/cpu/framework/updater_traits.hpp>
#include <artea/cpu/framework/constructor_traits.hpp>

#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/containers/vector_dataset.hpp>
#include <artea/cpu/containers/locked_buffer.hpp>
#include <artea/cpu/containers/tbb_buffer.hpp>
#include <artea/cpu/containers/word_aligned_bitmap.hpp>

#include <artea/cpu/index/neighbor.hpp>
#include <artea/cpu/index/index_graph.hpp>
#include <artea/cpu/index/graph_initializer.hpp>
#include <artea/cpu/index/graph_constructor.hpp>

#include <artea/cpu/propagation/nbr_log_table.hpp>
#include <artea/cpu/propagation/neighbor_updater.hpp>
#include <artea/cpu/propagation/rng_updater.hpp>
#include <artea/cpu/propagation/propagate_engine.hpp>

#include <artea/cpu/router/vector_router.hpp>
#include <artea/cpu/router/bruteforce_router.hpp>
#include <artea/cpu/router/proximity_graph_router.hpp>

#include <artea/cpu/utils/bit_ops.hpp>
#include <artea/cpu/utils/clear_cache.hpp>
#include <artea/cpu/utils/nbr_arr_checker.hpp>
#include <artea/cpu/utils/parallel.hpp>
#include <artea/cpu/utils/random_seq.hpp>
#include <artea/cpu/utils/simd_distance.hpp>
#include <artea/cpu/utils/vector_sampler.hpp>
#include <artea/cpu/utils/recall_estimator.hpp>