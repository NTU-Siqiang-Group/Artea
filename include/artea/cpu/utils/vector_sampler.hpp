/*
 * @FilePath: /Artea/include/artea/cpu/utils/vector_sampler.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-01-22 15:23:42
 * @Date: 2025-11-22 11:11:23
 * @Description: Implements vector sampling logic with parallel random number generation.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/utils/random_seq.hpp>

namespace artea {
namespace cpu {

template <typename BaseTraitsT>
class VectorSampler {

public:

    using vertex_id_t = typename BaseTraitsT::vertex_id_t;
    using vertex_num_t = typename BaseTraitsT::vertex_num_t;
    using vec_ele_t = typename BaseTraitsT::vec_ele_t;
    using vector_array_t = typename BaseTraitsT::vector_array_t;
    using random_seq_t = typename BaseTraitsT::random_seq_t;

    VectorSampler() = default;
    ~VectorSampler() = default;

    /**
     * @brief Creates a random sample from the source array.
     *
     * @param source_arr The original vector array.
     * @param sampling_ratio The fraction of vectors to sample.
     * @return vector_array_t A new vector array containing sampled data.
     */
    auto sample(
        const vector_array_t& source_arr,
        const float sampling_ratio
    ) const -> vector_array_t {

        // Retrieve the total number of vertices from the source array
        vertex_num_t total_vertices = source_arr.get_num_vecs();

        // Calculate the number of samples required based on the ratio
        vertex_num_t sample_count = static_cast<vertex_num_t>(total_vertices * sampling_ratio);

        // Ensure we sample at least one element if ratio > 0 and input is not empty
        if (sample_count == 0 && total_vertices > 0 && sampling_ratio > 0) {
            sample_count = 1;
        }

        auto dim = source_arr.get_vec_dim();
        vector_array_t sampled_arr(sample_count, dim);

        vec_ele_t* dest_base = sampled_arr.get_all();
        const vec_ele_t* src_base = source_arr.get_all();

        std::vector<vertex_id_t> indices(sample_count);

        // Initialize the random sequence generator with the range [0, total_vertices)
        random_seq_t random_seq(total_vertices);

        // Generate random indices in parallel chunks to maximize throughput
        tbb::parallel_for(tbb::blocked_range<vertex_num_t>(0, sample_count),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                // Calculate the pointer to the current segment of the indices array
                vertex_id_t* current_segment_ptr = indices.data() + r.begin();
                // Generate random numbers for this specific segment using the thread-local stream
                random_seq.generate(current_segment_ptr, r.size());
            }
        );

        // Copy the actual vector data from source to destination in parallel
        tbb::parallel_for(tbb::blocked_range<vertex_num_t>(0, sample_count),
            [&](const tbb::blocked_range<vertex_num_t>& r) {
                for (auto i = r.begin(); i != r.end(); ++i) {
                    vertex_id_t src_idx = indices[i];
                    const vec_ele_t* src_ptr = src_base + static_cast<size_t>(src_idx) * dim;
                    vec_ele_t* dest_ptr = dest_base + static_cast<size_t>(i) * dim;
                    std::copy(src_ptr, src_ptr + dim, dest_ptr);
                }
            }
        );

        return sampled_arr;
    }

    /**
     * @brief Functor operator to enable direct invocation of the sampler.
     */
    __attribute__((always_inline))
    auto operator()(
        const vector_array_t& source_arr,
        const float sampling_ratio
    ) const -> vector_array_t {
        return sample(source_arr, sampling_ratio);
    }

};  // class VectorSampler

}   // namespace cpu
}   // namespace artea