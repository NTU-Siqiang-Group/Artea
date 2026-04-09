/*
 * @FilePath: /Artea/include/artea/cpu/utils/random_seq_nr.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-03-08 00:00:00
 * @Date: 2026-03-08 00:00:00
 * @Description: Random sequence generator without replacement (no duplicates).
 *               Single-threaded implementation using partial Fisher-Yates.
 */

#pragma once

#include <random>
#include <vector>
#include <algorithm>
#include <span>
#include <type_traits>
#include <fmt/format.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename BaseTraitsT>
class RandomSeqNR {

    using vec_num_t = typename BaseTraitsT::vec_num_t;
    using vec_id_t = typename BaseTraitsT::vec_id_t;

public:
    /**
     * @brief Construct a new RandomSeqNR object.
     *
     * The class carries no fixed range; the upper bound is supplied
     * per-call to @c generate, allowing one instance to serve multiple
     * ranges across its lifetime.
     *
     * @param seed Random seed for reproducibility (default: random_device).
     */
    RandomSeqNR(uint32_t seed = std::random_device{}())
        : _gen(seed) {}

    RandomSeqNR(const RandomSeqNR&) = delete;
    RandomSeqNR& operator=(const RandomSeqNR&) = delete;
    RandomSeqNR(RandomSeqNR&&) = default;
    RandomSeqNR& operator=(RandomSeqNR&&) = default;

    /**
     * @brief Sample @p sample_size distinct values from @c [0, upper_bound)
     *        in a uniformly random order.
     *
     * Implementation: partial Fisher-Yates over a reusable internal
     * scratch buffer. The first @p sample_size positions of the scratch
     * are shuffled and returned as a span; the storage is owned by this
     * instance and stays valid until the next call to @c generate.
     *
     * @param sample_size Number of distinct samples to produce. Must be
     *        @c <= upper_bound.
     * @param upper_bound Exclusive upper bound for the value range.
     * @return A span of @p sample_size distinct values from
     *         @c [0, upper_bound) in uniformly random order, backed by
     *         the internal scratch buffer.
     */
    auto generate(vec_num_t sample_size, vec_num_t upper_bound) -> std::span<const vec_id_t> {
        if (sample_size > upper_bound) {
            ARTEA_ERROR(fmt::format(
                "RandomSeqNR: sample_size ({}) cannot exceed upper_bound ({})",
                sample_size, upper_bound));
        }

        // Grow the scratch buffer monotonically; never shrink. The buffer
        // is reused across calls, so the O(upper_bound) initialization
        // amortizes when the same instance is called repeatedly with the
        // same (or smaller) upper_bound.
        if (_scratch.size() < upper_bound) {
            _scratch.resize(upper_bound);
        }
        for (vec_num_t i = 0; i < upper_bound; ++i) {
            _scratch[i] = static_cast<vec_id_t>(i);
        }

        // Partial Fisher-Yates: at iteration i, swap _scratch[i] with a
        // uniformly random element from [i, upper_bound). After
        // sample_size iterations, _scratch[0..sample_size) holds a
        // uniformly random sample of distinct values from
        // [0, upper_bound) in random order.
        for (vec_num_t i = 0; i < sample_size; ++i) {
            std::uniform_int_distribution<vec_num_t> dist(i, upper_bound - 1);
            const vec_num_t j = dist(_gen);
            std::swap(_scratch[i], _scratch[j]);
        }

        return std::span<const vec_id_t>(_scratch.data(), sample_size);
    }

private:
    /** @brief Random number generator. */
    std::mt19937 _gen;

    /**
     * @brief Reusable working buffer for partial Fisher-Yates. Grows
     *        monotonically with the largest @c upper_bound seen so far.
     *        Not part of the class's logical state — storage only.
     */
    std::vector<vec_id_t> _scratch;

};  // class RandomSeqNR

}   // namespace cpu
}   // namespace artea
