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

#include <cstddef>
#include <cstdint>
#include <vector>
#include <functional>
#include <algorithm>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/utils/parallel.hpp>
#include <artea/cpu/router/candidate_queue_concept.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {

template <
    typename RouterTraitsT,
    CandidateQueue CandidateQueueImpl = typename RouterTraitsT::std_candidate_queue_t,
    VisitedTable VisitedTableImpl = typename RouterTraitsT::thread_local_bitmap_t
>   requires CandidateQueue<CandidateQueueImpl> && VisitedTable<VisitedTableImpl>
class HierarchicalGraphRouter :
    public RouterTraitsT::template vector_router_t<
        HierarchicalGraphRouterImpl<RouterTraitsT, CandidateQueueImpl, VisitedTableImpl>>
{

    using candidate_queue_t = CandidateQueueImpl;
    using visited_table_t = VisitedTableImpl;
    using visited_table_pool_t = typename RouterTraitsT::template visited_table_pool_t<VisitedTableImpl>;
    using vertex_num_t = typename RouterTraitsT::vertex_num_t;
    using vertex_id_t = typename RouterTraitsT::vertex_id_t;
    using vec_ele_t = typename RouterTraitsT::vec_ele_t;
    using distance_t = typename RouterTraitsT::distance_t;
    using dist_func_t = typename RouterTraitsT::dist_func_t;
    using vector_array_t = typename RouterTraitsT::vector_array_t;
    using idlist_array_t = typename RouterTraitsT::idlist_array_t;
    using flat_search_graph_t = typename RouterTraitsT::flat_search_graph_t;
    using hierarchical_search_graph_t = typename RouterTraitsT::hierarchical_search_graph_t;


};  // class HierarchicalGraphRouter

}   // namespace cpu
}   // namespace artea