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
 * @FilePath: /Artea/include/artea/cpu/utils/dataset_prober.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Probe dataset properties by sampling vertices and computing their
 *               top-K nearest neighbors against the full dataset.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/utils/random_seq.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Probe dataset geometric properties: k-NN distance quantile table and Local Intrinsic
 *        Dimensionality (LID).
 *
 * ## k-NN Distance Quantile Table
 *
 * Samples M vertices from the full dataset (N total), computes top-128 NN distances for each
 * via brute-force scan, then for each nn_rank (1..128) extracts the requested quantiles.
 * The result is a 128 x Q table where Q is the number of quantiles.
 *
 * ## Local Intrinsic Dimensionality (LID) Estimation
 *
 * LID measures the effective number of dimensions the data locally occupies. It captures
 * how the volume of a local ball grows with radius: in a d-dimensional space, the number
 * of points within radius r scales as N(r) ~ r^d. Real-world data often lies on a
 * lower-dimensional manifold embedded in high-dimensional space, so LID << ambient dimension.
 *
 * ### Levina-Bickel MLE Estimator
 *
 * For each sampled point x with k nearest neighbors at distances r_1 <= r_2 <= ... <= r_k,
 * the per-point LID estimate is:
 *
 *     LID(x) = [ (1/k) * sum_{i=1}^{k} ln(r_k / r_i) ]^{-1}
 *
 * The global LID is the average over all sampled points.
 *
 * ### Intuition: Shell Effect
 *
 * - **High LID**: In high-dimensional space, most neighbors cluster near the boundary r_k
 *   (shell effect). Ratios r_k/r_i ~ 1, log terms ~ 0, sum is small, so LID = 1/small is large.
 * - **Low LID**: In low-dimensional space, neighbors spread uniformly from center to boundary.
 *   Many r_i << r_k, log terms are large, sum is large, so LID = 1/large is small.
 *
 * ### Implementation Details
 *
 * - Distances r_i that are zero (duplicate vectors) are clamped to a small epsilon to avoid
 *   log(inf). These points contribute large log terms, pulling LID downward, which correctly
 *   reflects that duplicate data reduces effective dimensionality.
 * - The k used for LID estimation is MAX_K (128), the same as the NN table depth.
 *
 * @tparam ComputerTraitsT The computer traits type providing distance computation types.
 */
template <typename ComputerTraitsT>
class DatasetProber {

    using vec_num_t = typename ComputerTraitsT::vec_num_t;
    using vec_id_t = typename ComputerTraitsT::vec_id_t;
    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using distance_t = typename ComputerTraitsT::distance_t;
    using vector_array_t = typename ComputerTraitsT::vector_array_t;
    using random_seq_t = typename ComputerTraitsT::random_seq_t;
    using dist_func_t = typename ComputerTraitsT::dist_func_t;

    static constexpr uint32_t MAX_K = 128;
    static constexpr float EPSILON = 1e-6f;

public:

    /**
     * @brief Probe result containing the k-NN distance quantile table and LID estimate.
     */
    struct ProbeResult {
        std::vector<uint32_t> nn_ranks;                         // nn_ranks (1-based): 1..128
        std::vector<float> quantiles;                           // quantile values
        std::vector<std::vector<distance_t>> table;             // table[rank_idx][quantile_idx]
        vec_num_t num_samples;                                  // number of sampled vertices
        float lid;                                              // estimated Local Intrinsic Dimensionality
    };

    DatasetProber(const vector_array_t& base_vecs, const dist_func_t& dist_func)
        : _base_vecs(base_vecs), _dist_func(dist_func) {}

    /**
     * @brief Probe the dataset: compute the full nn_rank x quantile table and LID.
     *
     * @param quantiles Vector of target quantiles, each in (0, 1)
     * @param num_samples Number of vertices to sample
     * @return ProbeResult containing the table and LID estimate
     */
    auto probe(
        const std::vector<float>& quantiles,
        vec_num_t num_samples
    ) -> ProbeResult {

        // Compute top-K NN distances for all samples
        std::vector<distance_t> knn_table = _compute_knn_table(num_samples);

        // Build nn_ranks: 1..MAX_K
        std::vector<uint32_t> nn_ranks(MAX_K);
        for (uint32_t r = 0; r < MAX_K; ++r) {
            nn_ranks[r] = r + 1;
        }

        // Build quantile table: for each nn_rank, sort the column and extract quantiles
        std::vector<std::vector<distance_t>> table(MAX_K);

        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, MAX_K),
            [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t col = r.begin(); col != r.end(); ++col) {
                    std::vector<distance_t> column(num_samples);
                    for (vec_num_t i = 0; i < num_samples; ++i) {
                        column[i] = knn_table[i * MAX_K + col];
                    }

                    std::sort(column.begin(), column.end());

                    table[col].reserve(quantiles.size());
                    for (float q : quantiles) {
                        vec_num_t idx = static_cast<vec_num_t>(q * column.size());
                        if (idx >= column.size()) {
                            idx = column.size() - 1;
                        }
                        table[col].push_back(column[idx]);
                    }
                }
            }
        );

        // Compute LID using Levina-Bickel MLE estimator
        float lid = _compute_lid(knn_table, num_samples);

        ProbeResult result;
        result.nn_ranks = std::move(nn_ranks);
        result.quantiles = quantiles;
        result.table = std::move(table);
        result.num_samples = num_samples;
        result.lid = lid;
        return result;
    }

private:
    const vector_array_t& _base_vecs;
    const dist_func_t& _dist_func;

    /**
     * @brief Compute LID via Levina-Bickel MLE averaged over all sampled points.
     *
     * For each sample i with k-NN distances r_1..r_k (row i of knn_table):
     *   LID_i = [ (1/k) * sum_{j=1}^{k} ln(r_k / r_j) ]^{-1}
     *
     * Global LID = mean(LID_i) over all valid samples.
     *
     * @param knn_table Flat row-major table (num_samples x MAX_K), each row sorted ascending
     * @param num_samples Number of sampled vertices
     * @return Estimated global LID
     */
    auto _compute_lid(
        const std::vector<distance_t>& knn_table,
        vec_num_t num_samples
    ) -> float {

        // Use thread-local accumulators
        std::vector<double> lid_values(num_samples, 0.0);

        tbb::parallel_for(
            tbb::blocked_range<vec_num_t>(0, num_samples),
            [&](const tbb::blocked_range<vec_num_t>& r) {
                for (vec_num_t i = r.begin(); i != r.end(); ++i) {
                    const distance_t* row = knn_table.data() + static_cast<size_t>(i) * MAX_K;
                    distance_t r_k = row[MAX_K - 1];

                    // Skip degenerate cases where the k-th neighbor distance is zero
                    if (r_k <= EPSILON) {
                        lid_values[i] = 0.0;
                        continue;
                    }

                    double log_sum = 0.0;
                    for (uint32_t j = 0; j < MAX_K; ++j) {
                        distance_t r_j = std::max(row[j], static_cast<distance_t>(EPSILON));
                        log_sum += std::log(static_cast<double>(r_k) / static_cast<double>(r_j));
                    }

                    double avg_log = log_sum / static_cast<double>(MAX_K);
                    lid_values[i] = (avg_log > 1e-10) ? (1.0 / avg_log) : 0.0;
                }
            }
        );

        // Average over valid samples (LID > 0)
        double sum = 0.0;
        uint32_t valid_count = 0;
        for (vec_num_t i = 0; i < num_samples; ++i) {
            if (lid_values[i] > 0.0) {
                sum += lid_values[i];
                valid_count++;
            }
        }

        return (valid_count > 0) ? static_cast<float>(sum / valid_count) : 0.0f;
    }

    /**
     * @brief Sample vertices and compute their top-128 NN distances against the full dataset.
     *
     * For each sampled vertex, brute-force scans the entire dataset using a max-heap
     * of size MAX_K to maintain the K nearest neighbors.
     *
     * @param num_samples Number of vertices to sample
     * @return Flat vector of size num_samples * MAX_K (row-major), each row sorted ascending
     */
    auto _compute_knn_table(vec_num_t num_samples) -> std::vector<distance_t> {
        const vec_num_t total_vecs = _base_vecs.get_num_vecs();
        if (total_vecs < 2) {
            ARTEA_ERROR("Dataset must contain at least 2 vectors");
        }
        if (num_samples < 1) {
            ARTEA_ERROR("num_samples must be at least 1");
        }

        // Sample vertex IDs
        random_seq_t rand_gen(total_vecs);
        std::vector<vec_id_t> sample_ids(num_samples);
        tbb::parallel_for(
            tbb::blocked_range<vec_num_t>(0, num_samples),
            [&](const tbb::blocked_range<vec_num_t>& r) {
                rand_gen.generate(sample_ids.data() + r.begin(), r.size());
            }
        );

        // Flat table: num_samples x MAX_K
        std::vector<distance_t> knn_table(static_cast<size_t>(num_samples) * MAX_K);

        tbb::parallel_for(
            tbb::blocked_range<vec_num_t>(0, num_samples),
            [&](const tbb::blocked_range<vec_num_t>& r) {
                // Thread-local max-heap
                std::priority_queue<distance_t> heap;

                for (vec_num_t s = r.begin(); s != r.end(); ++s) {
                    const vec_id_t vid = sample_ids[s];
                    const vec_ele_t* query = _base_vecs.get(vid);

                    // Clear heap for reuse
                    while (!heap.empty()) heap.pop();

                    for (vec_num_t j = 0; j < total_vecs; ++j) {
                        if (j == vid) continue;
                        distance_t d = _dist_func(query, _base_vecs.get(j));
                        if (heap.size() < MAX_K) {
                            heap.push(d);
                        } else if (d < heap.top()) {
                            heap.pop();
                            heap.push(d);
                        }
                    }

                    // Extract heap into row, sorted ascending
                    distance_t* row = knn_table.data() + static_cast<size_t>(s) * MAX_K;
                    uint32_t count = static_cast<uint32_t>(heap.size());
                    // Fill from the end (heap pops max first)
                    for (uint32_t k = count; k > 0; --k) {
                        row[k - 1] = heap.top();
                        heap.pop();
                    }
                    // Fill remaining slots with max distance (if dataset has fewer than MAX_K vectors)
                    for (uint32_t k = count; k < MAX_K; ++k) {
                        row[k] = std::numeric_limits<distance_t>::max();
                    }
                }
            }
        );

        return knn_table;
    }

};  // class DatasetProber

}   // namespace cpu
}   // namespace artea
