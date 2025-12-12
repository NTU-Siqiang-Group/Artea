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
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t
>
class GraphInitializer {

public:
    static random_initialize(
        IndexGraph<vertex_num_t, vec_ele_t, graph_direction_t::HIBRID>& index_graph
    ) -> void {


    }   // random_initialize

};  // class GraphInitializer

}   // namespace cpu
}   // namespace artea