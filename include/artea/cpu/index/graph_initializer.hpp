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

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/utils/simd_distance.hpp>
#include <artea/cpu/index/index_graph.hpp>
#include <artea/cpu/containers/vector_dataset.hpp>
#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/propagation/propagate_engine.hpp>
#include <artea/cpu/propagation/rng_updater.hpp>
#include <artea/common/definitions.hpp>

namespace artea {
namespace cpu {

template <typename type_context_t>
class GraphInitializer {

    using vertex_num_t = typename type_context_t::vertex_num_t;
    using vertex_id_t = typename type_context_t::vertex_id_t;
    using vec_ele_t = typename type_context_t::vec_ele_t;
    using index_graph_t = typename type_context_t::index_graph_t;

public:
    static random_initialize(
        index_graph_t& index_graph
    ) -> void {


    }   // random_initialize

};  // class GraphInitializer

}   // namespace cpu
}   // namespace artea