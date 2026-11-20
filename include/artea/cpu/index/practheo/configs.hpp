// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>

namespace artea::cpu::practheo {

/** @brief Practheo-specific construction parameters. */
template <typename IndexTraitsT>
struct BuildConfig {
    using vec_num_t = typename IndexTraitsT::vec_num_t;
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using ratio_t = typename IndexTraitsT::ratio_t;

    BuildConfig(vec_num_t max_num_vecs, ratio_t sampling_rate0, ratio_t weight_promotion_probability = ratio_t{0.25},
                vertex_num_t num_nearest_nbrs = 64)
        : max_num_vecs(max_num_vecs), sampling_rate0(sampling_rate0),
          weight_promotion_probability(weight_promotion_probability), num_nearest_nbrs(num_nearest_nbrs) {
        update_weight_limits();
        sampling_rate1 = ratio_t{1} / log_max_num_vecs;
        validate_sampling_rates();
    }

    /** @brief Recompute the derived weight limits from the capacity and promotion probability. */
    auto update_weight_limits() -> void {
        if (max_num_vecs < 2) {
            throw std::invalid_argument("Practheo max_num_vecs must be at least 2");
        }
        if (!(weight_promotion_probability > 0 && weight_promotion_probability < 1)) {
            throw std::invalid_argument("Practheo weight_promotion_probability must be in (0, 1)");
        }

        const long double precise_log_max_num_vecs = std::logl(max_num_vecs);
        long double candidate_max_weight = std::ceil(
            precise_log_max_num_vecs / -std::logl(weight_promotion_probability));
        // Use the defining inequality to correct rounding at integer boundaries.
        if (max_num_vecs * std::pow(weight_promotion_probability, candidate_max_weight - 1) <= 1) {
            --candidate_max_weight;
        } else if (max_num_vecs * std::pow(weight_promotion_probability, candidate_max_weight) > 1) {
            ++candidate_max_weight;
        }
        if (candidate_max_weight > std::numeric_limits<vec_num_t>::max()) {
            throw std::overflow_error("Practheo max_weight is too large to represent");
        }
        log_max_num_vecs = static_cast<ratio_t>(precise_log_max_num_vecs);
        max_weight = static_cast<vec_num_t>(candidate_max_weight);
    }

    auto validate_sampling_rates() const -> void {
        if (!(sampling_rate0 >= 0 && sampling_rate0 <= 1)) {
            throw std::invalid_argument("Practheo sampling_rate0 must be in [0, 1]");
        }
        if (!(sampling_rate1 >= 0 && std::isfinite(sampling_rate1))) {
            throw std::invalid_argument("Practheo sampling_rate1 must be finite and nonnegative");
        }
    }

    /** @brief Inclusive upper bound on vertex weights. */
    auto max_vertex_weight() const -> vec_num_t { return max_weight; }

    /** @brief Maximum number of points the vector dataset may contain (at least 2). */
    vec_num_t max_num_vecs;

    /** @brief Natural logarithm of @c max_num_vecs. */
    ratio_t log_max_num_vecs;

    /** @brief Smallest H such that max_num_vecs * weight_promotion_probability^H is at most one. */
    vec_num_t max_weight;

    /** @brief Bernoulli probability of retaining each global nearest neighbor. */
    ratio_t sampling_rate0;

    /** @brief Retention probability for the pooled per-weight nearest neighbors, initially 1 / log_max_num_vecs and capped at 1 when sampling. */
    ratio_t sampling_rate1;

    /** @brief Probability in (0, 1) of advancing to the next weight, defaulting to 0.25. */
    ratio_t weight_promotion_probability;

    /** @brief Number of nearest neighbors to retrieve globally and from each weight group, excluding the query vertex. */
    vertex_num_t num_nearest_nbrs;
};

}  // namespace artea::cpu::practheo
