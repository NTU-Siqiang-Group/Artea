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
 * @Description: Probe distance distribution quantiles by sampling independent vector pairs
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
#include <boost/math/distributions/normal.hpp>
#include <artea/common/logger.hpp>

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

    // Batch size for sampling to reduce memory usage
    static constexpr vec_num_t SAMPLING_BATCH_SIZE = 100000;

public:
    /**
     * @brief Result containing the quantile radius value
     */
    struct ProbeResult {
        distance_t radius;                  // Estimated quantile radius value
        vec_num_t num_dists_sampled;        // Number of independent distance samples
        float quantile;                     // The quantile that was estimated
    };

    /**
     * @brief Result containing multiple quantile radius values
     */
    struct MultiQuantileProbeResult {
        std::vector<float> quantiles;       // Quantile values
        std::vector<distance_t> radii;      // Corresponding radius values
        vec_num_t num_dists_sampled;        // Number of independent distance samples
    };

    /**
     * @brief Construct a new RadiusProber object
     * @param dist_func Distance function for computing pairwise distances
     */
    RadiusProber(const dist_func_t& dist_func) : _dist_func(dist_func) {}

    /**
     * @brief Compute required number of distance samples
     *
     * Uses the formula: m = Z^2 * (1-p) / (p * delta^2)
     * where Z is the critical value from standard normal distribution
     *
     * @param quantile Target quantile (e.g., 0.0005 for 0.05%, 0.001 for 0.1%)
     * @param confidence Confidence level (e.g., 0.95 for 95%, 0.99 for 99%)
     * @param relative_err Relative error (e.g., 0.1 for 10%, 0.2 for 20%)
     * @return Required number of distance samples
     */
    static auto compute_num_dists_sampled(
        float quantile,
        float confidence,
        float relative_err
    ) -> vec_num_t {
        if (quantile <= 0.0f || quantile >= 1.0f) {
            ARTEA_ERROR("quantile must be in (0, 1)");
        }
        if (confidence <= 0.0f || confidence >= 1.0f) {
            ARTEA_ERROR("confidence must be in (0, 1)");
        }
        if (relative_err <= 0.0f) {
            ARTEA_ERROR("relative_err must be positive");
        }

        // Calculate alpha and Z-value
        float alpha = 1.0f - confidence;
        boost::math::normal_distribution<float> normal(0.0f, 1.0f);
        float z_value = boost::math::quantile(normal, 1.0f - alpha / 2.0f);

        // Calculate required samples: m = Z^2 * (1-p) / (p * delta^2)
        float m = (z_value * z_value * (1.0f - quantile)) / (quantile * relative_err * relative_err);

        return static_cast<vec_num_t>(std::ceil(m));
    }

    /**
     * @brief Probe a quantile with automatic sample size calculation
     *
     * This overload automatically computes the required number of distance samples
     * based on the desired confidence level and relative error.
     *
     * @param base_vecs The dataset to probe
     * @param quantile Target quantile (e.g., 0.0005 for 0.05%, 0.001 for 0.1%)
     * @param confidence Confidence level (e.g., 0.95 for 95%, 0.99 for 99%)
     * @param relative_err Relative error (e.g., 0.1 for 10%, 0.2 for 20%)
     * @return ProbeResult containing the quantile radius and sampling information
     */
    auto probe(
        const vector_array_t& base_vecs,
        float quantile,
        float confidence,
        float relative_err
    ) -> ProbeResult {
        vec_num_t num_distances = compute_num_dists_sampled(quantile, confidence, relative_err);
        return probe(base_vecs, quantile, num_distances);
    }

    /**
     * @brief Probe a quantile from the distance distribution
     *
     * This method samples independent pairs of vectors from the dataset by:
     * 1. Processing in batches to reduce memory usage
     * 2. For each batch: sampling batch_size vectors as first endpoints and batch_size as second endpoints
     * 3. Computing distances between corresponding pairs
     * This ensures distance samples are i.i.d. (independent and identically distributed).
     *
     * @param base_vecs The dataset to probe
     * @param quantile Target quantile (e.g., 0.0005 for 0.05%, 0.001 for 0.1%)
     * @param num_distances_to_sample Number of independent distance samples to compute
     * @return ProbeResult containing the quantile radius and sampling information
     */
    auto probe(
        const vector_array_t& base_vecs,
        float quantile,
        vec_num_t num_distances_to_sample
    ) -> ProbeResult {

        if (quantile <= 0.0f || quantile >= 1.0f) {
            ARTEA_ERROR("quantile must be in (0, 1)");
        }

        const vec_num_t total_vecs = base_vecs.get_num_vecs();
        if (total_vecs < 2) {
            ARTEA_ERROR("Dataset must contain at least 2 vectors");
        }

        if (num_distances_to_sample < 1) {
            ARTEA_ERROR("num_distances_to_sample must be at least 1");
        }

        // Allocate result vector for all distances
        std::vector<distance_t> distances;
        distances.reserve(num_distances_to_sample);

        // Process in batches to reduce memory usage
        vec_num_t remaining = num_distances_to_sample;
        while (remaining > 0) {
            vec_num_t batch_size = std::min(remaining, SAMPLING_BATCH_SIZE);

            // Sample two independent sets of vector indices for this batch
            std::vector<vec_id_t> vec_ids_1 = sample_vec_ids(total_vecs, batch_size);
            std::vector<vec_id_t> vec_ids_2 = sample_vec_ids(total_vecs, batch_size);

            // Compute distances for this batch
            std::vector<distance_t> batch_distances = compute_paired_distances(base_vecs, vec_ids_1, vec_ids_2);

            // Append to result
            distances.insert(distances.end(), batch_distances.begin(), batch_distances.end());

            remaining -= batch_size;
        }

        // Sort distances
        std::sort(distances.begin(), distances.end());

        // Extract quantile value
        vec_num_t quantile_index = static_cast<vec_num_t>(quantile * distances.size());
        if (quantile_index >= distances.size()) {
            quantile_index = distances.size() - 1;
        }

        ProbeResult result;
        result.radius = distances[quantile_index];
        result.num_dists_sampled = num_distances_to_sample;
        result.quantile = quantile;

        return result;
    }

    /**
     * @brief Probe multiple quantiles from the distance distribution
     *
     * This method samples independent pairs of vectors and computes multiple quantiles
     * in a single pass through the sorted distances.
     *
     * @param base_vecs The dataset to probe
     * @param confidence Confidence level for computing sample size (e.g., 0.95 for 95%, 0.99 for 99%)
     * @param relative_err Relative error for computing sample size (e.g., 0.1 for 10%, 0.2 for 20%)
     * @return MultiQuantileProbeResult containing all quantile radii and sampling information
     */
    auto probe_multi_quantiles(
        const vector_array_t& base_vecs,
        float confidence,
        float relative_err
    ) -> MultiQuantileProbeResult {
        // Define quantiles to probe
        std::vector<float> quantiles = {
            0.0001f, 0.0005f, 0.001f, 0.005f, 0.01f, 0.05f, 0.1f,
            0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f
        };

        // Use the smallest quantile to compute required sample size (most conservative)
        float min_quantile = *std::min_element(quantiles.begin(), quantiles.end());
        vec_num_t num_distances = compute_num_dists_sampled(min_quantile, confidence, relative_err);

        const vec_num_t total_vecs = base_vecs.get_num_vecs();
        if (total_vecs < 2) {
            ARTEA_ERROR("Dataset must contain at least 2 vectors");
        }

        // Allocate result vector for all distances
        std::vector<distance_t> distances;
        distances.reserve(num_distances);

        // Process in batches to reduce memory usage
        vec_num_t remaining = num_distances;
        while (remaining > 0) {
            vec_num_t batch_size = std::min(remaining, SAMPLING_BATCH_SIZE);

            // Sample two independent sets of vector indices for this batch
            std::vector<vec_id_t> vec_ids_1 = sample_vec_ids(total_vecs, batch_size);
            std::vector<vec_id_t> vec_ids_2 = sample_vec_ids(total_vecs, batch_size);

            // Compute distances for this batch
            std::vector<distance_t> batch_distances = compute_paired_distances(base_vecs, vec_ids_1, vec_ids_2);

            // Append to result
            distances.insert(distances.end(), batch_distances.begin(), batch_distances.end());

            remaining -= batch_size;
        }

        // Sort distances once
        std::sort(distances.begin(), distances.end());

        // Extract all quantile values in one pass
        MultiQuantileProbeResult result;
        result.quantiles = quantiles;
        result.radii.reserve(quantiles.size());
        result.num_dists_sampled = num_distances;

        for (float quantile : quantiles) {
            vec_num_t quantile_index = static_cast<vec_num_t>(quantile * distances.size());
            if (quantile_index >= distances.size()) {
                quantile_index = distances.size() - 1;
            }
            result.radii.push_back(distances[quantile_index]);
        }

        return result;
    }

private:
    const dist_func_t& _dist_func;

    /**
     * @brief Sample vector IDs uniformly at random
     *
     * @param total_vecs Total number of vectors in the dataset
     * @param num_samples Number of IDs to sample
     * @return Vector of sampled vector IDs
     */
    auto sample_vec_ids(
        vec_num_t total_vecs,
        vec_num_t num_samples
    ) -> std::vector<vec_id_t> {

        // Initialize random generator
        random_seq_t rand_gen(total_vecs);

        // Generate random indices
        std::vector<vec_id_t> sampled_ids(num_samples);

        tbb::parallel_for(
            tbb::blocked_range<vec_num_t>(0, num_samples),
            [&](const tbb::blocked_range<vec_num_t>& r) {
                rand_gen.generate(sampled_ids.data() + r.begin(), r.size());
            }
        );

        return sampled_ids;
    }

    /**
     * @brief Compute distances between corresponding pairs of vectors
     *
     * @param base_vecs The dataset
     * @param vec_ids_1 First set of vector IDs
     * @param vec_ids_2 Second set of vector IDs
     * @return Vector of distances between vec_ids_1[i] and vec_ids_2[i]
     */
    auto compute_paired_distances(
        const vector_array_t& base_vecs,
        const std::vector<vec_id_t>& vec_ids_1,
        const std::vector<vec_id_t>& vec_ids_2
    ) -> std::vector<distance_t> {

        const vec_num_t num_pairs = static_cast<vec_num_t>(vec_ids_1.size());
        std::vector<distance_t> distances(num_pairs);

        tbb::parallel_for(
            tbb::blocked_range<vec_num_t>(0, num_pairs),
            [&](const tbb::blocked_range<vec_num_t>& r) {
                for (vec_num_t i = r.begin(); i != r.end(); ++i) {
                    const vec_ele_t* vec_1 = base_vecs.get(vec_ids_1[i]);
                    const vec_ele_t* vec_2 = base_vecs.get(vec_ids_2[i]);
                    distances[i] = _dist_func(vec_1, vec_2);
                }
            }
        );

        return distances;
    }

};  // class RadiusProber

}   // namespace cpu
}   // namespace artea
