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

template <typename RefinerTraitsT>
class SpeculativeUpdater :
    public RefinerTraitsT::template neighbor_updater_t<SpeculativeUpdater<RefinerTraitsT>> {

    using vertex_id_t = typename RefinerTraitsT::vertex_id_t;
    using vertex_num_t = typename RefinerTraitsT::vertex_num_t;
    using vec_ele_t = typename RefinerTraitsT::vec_ele_t;
    using distance_t = typename RefinerTraitsT::distance_t;
    using ratio_t = typename RefinerTraitsT::ratio_t;
    using vector_array_t = typename RefinerTraitsT::vector_array_t;
    using bnbr_t = typename RefinerTraitsT::bnbr_t;
    using bnbr_arr_t = typename RefinerTraitsT::bnbr_arr_t;
    using log_table_t = typename RefinerTraitsT::log_table_t;
    using dist_func_t = typename RefinerTraitsT::dist_func_t;
    using base_class_t = typename RefinerTraitsT::template neighbor_updater_t<TriangleUpdater<RefinerTraitsT>>;

    SpeculativeUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_data,
        log_table_t& log_table
    ) : base_class_t(dist_func, vecs_data, log_table) {}

};  //  class SpeculativeUpdater

