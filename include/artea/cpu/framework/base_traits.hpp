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
 * @FilePath: /Artea/include/artea/cpu/framework/base_traits.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstddef>
#include <vector>
#include <utility>
#include <limits>
#include <cmath>
#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/containers/thread_local_bitmap.hpp>
#include <artea/cpu/containers/version_tag_table.hpp>
#include <artea/cpu/containers/word_aligned_bitmap.hpp>

namespace artea {
namespace cpu {

/* ------ Forward Declarations ------ */
template <typename BaseTraitsT> struct Neighbor;
template <typename BaseTraitsT> class NbrLogTable;
template <typename BaseTraitsT> class VectorDataset;
template <typename BaseTraitsT> class VectorSampler;
template <typename BaseTraitsT> class NbrArrChecker;
template <typename BaseTraitsT> class RandomSeq;
template <typename BaseTraitsT> struct NeighborComparator;
template <typename BaseTraitsT> struct StrictNeighborComparator;
template <typename BaseTraitsT> struct NeighborIdComparator;
template <typename BaseTraitsT> struct NeighborDistanceComparator;
template <typename BaseTraitsT> struct VertexSubset;
template <typename T, typename ContainerT, typename Compare> class FourAryHeap;

/* ------ Enumerations ------ */
enum class PruningConditionT;

/* ------ Base Traits Definition ------ */
template <typename VertexNumT, typename VecEleT, bool ProfilingMode = false>
struct BaseTraits {

private:
    /** ------ Basic Type ------ **/
    using base_traits_t = BaseTraits<VertexNumT, VecEleT, ProfilingMode>;

public:
    /** @brief vector dimensions. */
    using vec_dim_t = uint32_t;

    /** @brief vertex numbers. */
    using vertex_num_t = VertexNumT;

    /** @brief vertex identifiers. */
    using vertex_id_t = VertexNumT;

    /** @brief number of vectors. */
    using vec_num_t = VertexNumT;

    /** @brief vector identifiers. */
    using vec_id_t = VertexNumT;

    /** @brief vector elements. */
    using vec_ele_t = VecEleT;

    /** @brief distance values. */
    using distance_t = VecEleT;

    /** @brief number of clusters. */
    using cluster_num_t = VertexNumT;

    /** @brief cluster identifiers. */
    using cluster_id_t = VertexNumT;

    /** @brief partition numbers. */
    using part_num_t = VertexNumT;

    /** @brief partition identifiers. */
    using part_id_t = VertexNumT;

    /** @brief number of layers. */
    using layer_num_t = uint32_t;

    /** @brief layer identifiers. */
    using layer_id_t = uint32_t;

    /** @brief batch identifiers. */
    using batch_id_t = uint32_t;

    /** @brief number of hash functions. */
    using hash_num_t  = uint32_t;

    /** @brief iteration counts. */
    using iter_t = uint32_t;

    /** @brief predefined parameter ratio type */
    using ratio_t = float;

    /** @brief word-aligned bitmap. */
    using word_aligned_bitmap_t = WordAlignedBitmap;

    /** @brief thread-local bitmap */
    using thread_local_bitmap_t = ThreadLocalBitmap;

    /** @brief version-tag visited table */
    using version_tag_table_t = VersionTagTable;

    /** @brief neighbor entries. */
    using nbr_t = Neighbor<base_traits_t>;

    /** @brief neighbor comparator. */
    using nbr_comp_t = NeighborComparator<base_traits_t>;

    /** @brief strict neighbor comparator. */
    using strict_nbr_comp_t = StrictNeighborComparator<base_traits_t>;

    /** @brief ID-only comparator. */
    using nbr_id_comp_t = NeighborIdComparator<base_traits_t>;

    /** @brief distance-only comparator. */
    using nbr_dist_comp_t = NeighborDistanceComparator<base_traits_t>;

    /** @brief neighbor arrays. */
    using nbr_arr_t = std::vector<nbr_t>;

    /** @brief CSR graph format. */
    using csr_graph_t = cache_aligned_container_t<vertex_id_t>;

    using vector_t = avx512_container_t<vec_ele_t>;

    /** @brief vector arrays. */
    using vector_array_t = VectorArray<vertex_num_t, vec_ele_t>;

    /** @brief id list array */
    using idlist_array_t = VectorArray<vertex_num_t, vec_id_t>;

    /** @brief vector datasets. */
    using vector_dataset_t = VectorDataset<base_traits_t>;

    /** @brief base vector arrays. */
    using base_vecs_t = vector_array_t;

    /** @brief query vector arrays. */
    using query_vecs_t = vector_array_t;

    /** @brief ground truth vector arrays. */
    using ground_truth_t = idlist_array_t;

    /** @brief vector samplers. */
    using vector_sampler_t = VectorSampler<base_traits_t>;

    /** @brief neighbor array checkers. */
    using nbr_arr_checker_t = NbrArrChecker<base_traits_t>;

    /** @brief random sequences generator. */
    using random_seq_t = RandomSeq<base_traits_t>;

    /** @brief vertex subset type. */
    using vertex_subset_t = VertexSubset<base_traits_t>;

    /** @brief pruning condition type for triangle inequality. */
    using pruning_condition_t = PruningConditionT;

    /** @brief profiling mode flag. */
    static constexpr bool profiling_mode = ProfilingMode;

    __attribute__((always_inline))
    static constexpr auto invalid_vertex_id_generator() -> vertex_id_t {
        return std::numeric_limits<vertex_id_t>::max();
    }
    static constexpr vertex_id_t invalid_vertex_id = invalid_vertex_id_generator();

    __attribute__((always_inline))
    static constexpr auto nan_distance_generator() -> distance_t {
        return std::numeric_limits<distance_t>::quiet_NaN();
    }
    static constexpr distance_t nan_distance = nan_distance_generator();

    __attribute__((always_inline))
    static constexpr auto max_distance_generator() -> distance_t {
        return std::numeric_limits<distance_t>::max();
    }
    static constexpr distance_t max_distance = max_distance_generator();

    __attribute__((always_inline))
    static constexpr auto min_distance_generator() -> distance_t {
        return std::numeric_limits<distance_t>::lowest();
    }
    static constexpr distance_t min_distance = min_distance_generator();

    __attribute__((always_inline))
    static constexpr auto is_nan_distance(const distance_t dist) -> bool {
        return std::isnan(dist);
    }

    template <typename T>
    static constexpr auto get_max_value() -> T {
        return std::numeric_limits<T>::max();
    }

};  // struct BaseTraits

}   // namespace cpu
}   // namespace artea
