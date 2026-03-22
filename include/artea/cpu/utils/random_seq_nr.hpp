/*
 * @FilePath: /Artea/include/artea/cpu/utils/random_seq_nr.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-03-08 00:00:00
 * @Date: 2026-03-08 00:00:00
 * @Description: Random sequence generator without replacement (no duplicates).
 *               Single-threaded implementation using Fisher-Yates shuffle.
 */

#pragma once

#include <random>
#include <vector>
#include <algorithm>
#include <stdexcept>
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
     * @param num_vecs The total number of elements in the sequence (0 to num_vecs-1).
     * @param seed Random seed for reproducibility (default: random_device).
     */
    RandomSeqNR(const vec_num_t num_vecs, uint32_t seed = std::random_device{}())
        : _num_vecs(num_vecs), _gen(seed), _indices(num_vecs), _cursor(0) {
        // Initialize sequential indices
        for (vec_num_t i = 0; i < num_vecs; ++i) {
            _indices[i] = static_cast<vec_id_t>(i);
        }
        // Shuffle using Fisher-Yates algorithm
        std::shuffle(_indices.begin(), _indices.end(), _gen);
    }

    RandomSeqNR(const RandomSeqNR&) = delete;
    RandomSeqNR& operator=(const RandomSeqNR&) = delete;
    RandomSeqNR(RandomSeqNR&&) = default;
    RandomSeqNR& operator=(RandomSeqNR&&) = default;

    /**
     * @brief Generate a sample of specified size without replacement.
     * @param sample_size Number of elements to sample.
     * @return A span view of the sampled indices.
     * @note If cursor + sample_size exceeds total size, automatically reshuffles and resets.
     */
    std::span<const vec_id_t> generate(vec_num_t sample_size) {
        if (sample_size > _num_vecs) {
            ARTEA_ERROR("Sample size cannot exceed total number of vectors");
        }

        // Check if we need to reshuffle
        if (_cursor + sample_size > _num_vecs) {
            ARTEA_WARN(fmt::format(
                "RandomSeqNR: cursor ({}) + sample_size ({}) > num_vecs ({}), reshuffling",
                _cursor, sample_size, _num_vecs
            ));
            reshuffle();
            _cursor = 0;
        }

        // Return span from current cursor position
        std::span<const vec_id_t> result(_indices.data() + _cursor, sample_size);
        _cursor += sample_size;

        return result;
    }

    /**
     * @brief Get the shuffled index at position i.
     * @param i Position in the shuffled sequence (0 to num_vecs-1).
     * @return The shuffled index.
     */
    vec_id_t operator[](vec_num_t i) const {
        return _indices[i];
    }

    /**
     * @brief Get the total number of indices.
     */
    vec_num_t size() const {
        return _num_vecs;
    }

    /**
     * @brief Get the current cursor position.
     */
    vec_num_t cursor() const {
        return _cursor;
    }

    /**
     * @brief Reset the cursor to the beginning without reshuffling.
     */
    void reset_cursor() {
        _cursor = 0;
    }

    /**
     * @brief Reshuffle the sequence with a new random order and reset cursor.
     */
    void reshuffle() {
        std::shuffle(_indices.begin(), _indices.end(), _gen);
        _cursor = 0;
    }

private:
    /** @brief The total number of elements. */
    const vec_num_t _num_vecs;

    /** @brief Random number generator. */
    std::mt19937 _gen;

    /** @brief Shuffled indices. */
    std::vector<vec_id_t> _indices;

    /** @brief Current cursor position for sequential sampling. */
    vec_num_t _cursor;

};  // class RandomSeqNR

}   // namespace cpu
}   // namespace artea
