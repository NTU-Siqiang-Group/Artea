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
 * @FilePath: /Artea/include/artea/cpu/index/vertices_builder_config.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Configuration for vertices builder algorithm.
 */

#pragma once

namespace artea {
namespace cpu {

/**
 * @brief Configuration for greedy vertices builder (R-net based).
 * Supports builder pattern for flexible configuration.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
struct GreedyVerticesBuilderConfig {
    using distance_t = typename IndexTraitsT::distance_t;
    using ratio_t = typename IndexTraitsT::ratio_t;
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;

    /**
     * @brief Constructor for greedy vertices builder configuration.
     * @param min_radius Minimum radius for R-net (bottom layer radius).
     * @param beta_sq Beta squared coefficient for radius scaling between layers.
     * @param coverage_ratio Coverage ratio for R-net generation (recommend: 0.95).
     * @param confidence Confidence level for R-net generation (recommend: 0.95).
     * @param max_result_ratio Maximum result size ratio relative to current layer size (recommend: 0.2).
     * @param sampling_batch_size Batch size for sampling (recommend: 1024).
     * @param is_shuffle Whether to shuffle the input vectors before processing.
     */
    GreedyVerticesBuilderConfig(
        distance_t min_radius,
        ratio_t beta_sq,
        ratio_t coverage_ratio,
        ratio_t confidence,
        ratio_t max_result_ratio,
        vertex_num_t sampling_batch_size,
        bool is_shuffle
    ) :
        _min_radius(min_radius),
        _beta_sq(beta_sq),
        _coverage_ratio(coverage_ratio),
        _confidence(confidence),
        _max_result_ratio(max_result_ratio),
        _sampling_batch_size(sampling_batch_size),
        _is_shuffle(is_shuffle)
    {}

    // Builder pattern setters (chainable)
    auto min_radius(distance_t value) -> GreedyVerticesBuilderConfig& { _min_radius = value; return *this; }
    auto beta_sq(ratio_t value) -> GreedyVerticesBuilderConfig& { _beta_sq = value; return *this; }
    auto coverage_ratio(ratio_t value) -> GreedyVerticesBuilderConfig& { _coverage_ratio = value; return *this; }
    auto confidence(ratio_t value) -> GreedyVerticesBuilderConfig& { _confidence = value; return *this; }
    auto max_result_ratio(ratio_t value) -> GreedyVerticesBuilderConfig& { _max_result_ratio = value; return *this; }
    auto sampling_batch_size(vertex_num_t value) -> GreedyVerticesBuilderConfig& { _sampling_batch_size = value; return *this; }
    auto is_shuffle(bool value) -> GreedyVerticesBuilderConfig& { _is_shuffle = value; return *this; }

    // Const getters
    auto min_radius() const -> distance_t { return _min_radius; }
    auto beta_sq() const -> ratio_t { return _beta_sq; }
    auto coverage_ratio() const -> ratio_t { return _coverage_ratio; }
    auto confidence() const -> ratio_t { return _confidence; }
    auto max_result_ratio() const -> ratio_t { return _max_result_ratio; }
    auto sampling_batch_size() const -> vertex_num_t { return _sampling_batch_size; }
    auto is_shuffle() const -> bool { return _is_shuffle; }

private:
    /** @brief Minimum radius for R-net (bottom layer radius). */
    distance_t _min_radius;

    /** @brief Beta squared coefficient for radius scaling between layers. */
    ratio_t _beta_sq;

    /** @brief Coverage ratio for R-net generation (recommend: 0.95). */
    ratio_t _coverage_ratio;

    /** @brief Confidence level for R-net generation (recommend: 0.95). */
    ratio_t _confidence;

    /** @brief Maximum result size ratio relative to current layer size (recommend: 0.2). */
    ratio_t _max_result_ratio;

    /** @brief Batch size for sampling (recommend: 1024). */
    vertex_num_t _sampling_batch_size;

    /** @brief Whether to shuffle the input vectors before processing. */
    bool _is_shuffle;
};

/**
 * @brief Configuration for random vertices builder.
 * Supports builder pattern for flexible configuration.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
struct RandomVerticesBuilderConfig {
    using ratio_t = typename IndexTraitsT::ratio_t;

    /**
     * @brief Constructor for random vertices builder configuration.
     * @param random_result_ratio Result size ratio for random selection (recommend: 0.2).
     */
    explicit RandomVerticesBuilderConfig(ratio_t random_result_ratio) :
        _random_result_ratio(random_result_ratio)
    {}

    // Builder pattern setter (chainable)
    auto random_result_ratio(ratio_t value) -> RandomVerticesBuilderConfig& { _random_result_ratio = value; return *this; }

    // Const getter
    auto random_result_ratio() const -> ratio_t { return _random_result_ratio; }

private:
    /** @brief Result size ratio for random selection (recommend: 0.2). */
    ratio_t _random_result_ratio;
};

}   // namespace cpu
}   // namespace artea