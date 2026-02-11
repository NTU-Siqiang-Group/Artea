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
 * @FilePath: /Artea/include/artea/cpu/utils/radius_prober.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Probe distance distribution quantiles by sampling points and computing pairwise distances
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/utils/random_seq.hpp>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT>
class RadiusProber {

    using vec_num_t = typename ComputerTraitsT::vec_num_t;
    using vec_id_t = typename ComputerTraitsT::vec_id_t;
    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using distance_t = typename ComputerTraitsT::distance_t;
    using vector_array_t = typename ComputerTraitsT::vector_array_t;
    using random_seq_t = typename ComputerTraitsT::random_seq_t;
    using dist_func_t = typename ComputerTraitsT::dist_func_t;

public:
    /**
     * @brief Result containing the quantile radius value
     */
    struct ProbeResult {
        distance_t radius;                  // Estimated quantile radius value
        vec_num_t num_vecs_sampled;         // Number of vectors sampled
        vec_num_t num_distances_computed;   // Number of pairwise distances computed
        float quantile;                     // The quantile that was estimated
    };

    /**
     * @brief Construct a new RadiusProber object
     * @param dist_func Distance function for computing pairwise distances
     */
    RadiusProber(const dist_func_t& dist_func) : _dist_func(dist_func) {}

    /**
     * @brief Probe a quantile from the distance distribution
     *
     * This method samples vectors from the dataset, computes all pairwise distances
     * among the sampled vectors, and returns the specified quantile of those distances.
     *
     * @param base_vecs The dataset to probe
     * @param quantile Target quantile (e.g., 0.01 for 1%, 0.05 for 5%, 0.10 for 10%)
     * @param num_vecs_to_sample Number of vectors to sample from the dataset
     * @return ProbeResult containing the quantile radius and sampling information
     */
    auto probe(
        const vector_array_t& base_vecs,
        float quantile,
        vec_num_t num_vecs_to_sample
    ) -> ProbeResult {

        if (quantile <= 0.0f || quantile >= 1.0f) {
            throw std::invalid_argument("quantile must be in (0, 1)");
        }

        const vec_num_t total_vecs = base_vecs.get_num_vecs();
        if (total_vecs < 2) {
            throw std::invalid_argument("Dataset must contain at least 2 vectors");
        }

        if (num_vecs_to_sample < 2) {
            throw std::invalid_argument("num_vecs_to_sample must be at least 2");
        }

        if (num_vecs_to_sample > total_vecs) {
            num_vecs_to_sample = total_vecs;
        }

        // Sample vectors from the dataset
        vector_array_t sampled_vecs = sample_vecs(base_vecs, num_vecs_to_sample);

        // Compute all pairwise distances among sampled vectors
        std::vector<distance_t> pairwise_distances = compute_pairwise_distances(sampled_vecs);

        // Sort distances
        std::sort(pairwise_distances.begin(), pairwise_distances.end());

        // Extract quantile value
        vec_num_t quantile_index = static_cast<vec_num_t>(quantile * pairwise_distances.size());
        if (quantile_index >= pairwise_distances.size()) {
            quantile_index = pairwise_distances.size() - 1;
        }

        ProbeResult result;
        result.radius = pairwise_distances[quantile_index];
        result.num_vecs_sampled = num_vecs_to_sample;
        result.num_distances_computed = static_cast<vec_num_t>(pairwise_distances.size());
        result.quantile = quantile;

        return result;
    }

private:
    const dist_func_t& _dist_func;

    /**
     * @brief Sample vectors from the dataset
     *
     * @param base_vecs The dataset
     * @param num_vecs Number of vectors to sample
     * @return Vector array containing sampled vectors
     */
    auto sample_vecs(
        const vector_array_t& base_vecs,
        vec_num_t num_vecs
    ) -> vector_array_t {

        const vec_num_t total_vecs = base_vecs.get_num_vecs();
        const vec_num_t vec_dim = base_vecs.get_vec_dim();

        // Initialize random generator
        random_seq_t rand_gen(total_vecs);

        // Generate random indices
        std::vector<vec_id_t> sampled_indices(num_vecs);

        tbb::parallel_for(
            tbb::blocked_range<vec_num_t>(0, num_vecs),
            [&](const tbb::blocked_range<vec_num_t>& r) {
                rand_gen.generate(sampled_indices.data() + r.begin(), r.size());
            }
        );

        // Copy sampled vectors to new array
        vector_array_t sampled_vecs(num_vecs, vec_dim);

        tbb::parallel_for(
            tbb::blocked_range<vec_num_t>(0, num_vecs),
            [&](const tbb::blocked_range<vec_num_t>& r) {
                for (vec_num_t i = r.begin(); i != r.end(); ++i) {
                    vec_id_t source_idx = sampled_indices[i];
                    const vec_ele_t* source_vec = base_vecs.get(source_idx);
                    vec_ele_t* dest_vec = sampled_vecs.get(i);
                    std::copy(source_vec, source_vec + vec_dim, dest_vec);
                }
            }
        );

        return sampled_vecs;
    }

    /**
     * @brief Compute all pairwise distances among vectors
     *
     * @param vecs Vector array of vectors
     * @return Vector of all pairwise distances
     */
    auto compute_pairwise_distances(
        const vector_array_t& vecs
    ) -> std::vector<distance_t> {

        const vec_num_t num_vecs = vecs.get_num_vecs();

        // Calculate number of unique pairs: n*(n-1)/2
        const vec_num_t num_pairs = (num_vecs * (num_vecs - 1)) / 2;

        // Create thread-local storage for each thread's local distance array
        tbb::enumerable_thread_specific<std::vector<distance_t>> thread_local_dists;

        // Compute pairwise distances in parallel
        tbb::parallel_for(
            tbb::blocked_range<vec_num_t>(0, num_vecs - 1),
            [&](const tbb::blocked_range<vec_num_t>& r) {
                // Get this thread's local distance array
                auto& local_dists = thread_local_dists.local();

                for (vec_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* vec_i = vecs.get(i);

                    // Compute distances to all vectors j > i
                    for (vec_num_t j = i + 1; j < num_vecs; ++j) {
                        const vec_ele_t* vec_j = vecs.get(j);
                        local_dists.push_back(_dist_func(vec_i, vec_j));
                    }
                }
            }
        );

        // Concatenate all thread-local arrays into the final result in parallel
        std::vector<distance_t> distances;
        distances.resize(num_pairs);

        // Collect all thread-local arrays
        std::vector<const std::vector<distance_t>*> local_arrays;
        for (const auto& local_dists : thread_local_dists) {
            local_arrays.push_back(&local_dists);
        }

        // Calculate prefix sums to determine starting positions for each thread's data
        std::vector<vec_num_t> start_positions(local_arrays.size() + 1, 0);
        for (size_t i = 0; i < local_arrays.size(); ++i) {
            start_positions[i + 1] = start_positions[i] + static_cast<vec_num_t>(local_arrays[i]->size());
        }

        // Parallel copy from local arrays to final result
        // Each thread copies its own data to the correct position
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, local_arrays.size()),
            [&](const tbb::blocked_range<size_t>& r) {
                for (size_t i = r.begin(); i != r.end(); ++i) {
                    const auto& local_dists = *local_arrays[i];
                    std::copy(local_dists.begin(), local_dists.end(),
                              distances.begin() + start_positions[i]);
                }
            }
        );

        return distances;
    }

};  // class RadiusProber

}   // namespace cpu
}   // namespace artea
