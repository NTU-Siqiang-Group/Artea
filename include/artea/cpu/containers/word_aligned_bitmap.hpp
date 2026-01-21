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
 * @FilePath: /Artea/include/artea/cpu/containers/word_aligned_bitmap.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>

#include <artea/common/definitions.hpp>
#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {

/**
 * @brief A high-performance non-atomic bitmap optimized for word-aligned parallel processing.
 *
 * Design Philosophy:
 * Instead of using atomic operations (slow) or locks (slower), this bitmap implies
 * a specific parallelization strategy:
 * * The work is partitioned by "Word Index" (chunks of 64 bits), not by individual bits.
 * * A single thread claims ownership of a whole uint64_t word.
 * * Bitwise operations are done in registers, and the result is written to memory once per word.
 */
class WordAlignedBitmap {

    using container_t = cache_aligned_container_t<uint64_t>;

public:
    using word_t = uint64_t;
    static constexpr size_t BITS_PER_WORD = 64;
    static constexpr size_t WORD_SHIFT = 6; // 2^6 = 64
    static constexpr word_t WORD_MASK = 0x3F;

    WordAlignedBitmap() = default;

    explicit WordAlignedBitmap(const size_t num_bits) {
        resize(num_bits);
    }

    /**
     * @brief Resize and clear the bitmap.
     * @param num_bits The new size in bits.
     */
    void resize(const size_t num_bits) {
        _num_bits = num_bits;
        // Ceil division: (num_bits + 63) / 64
        size_t num_words = (num_bits + BITS_PER_WORD - 1) >> WORD_SHIFT;
        _data.assign(num_words, 0);
    }

    /**
     * @brief Check if a specific bit is set. Read-only is always thread-safe.
     * @param bit_index The index of the bit to check.
     * @return true if the bit is set, false otherwise.
     */
    __attribute__((always_inline))
    bool test(const size_t bit_index) const {
        const size_t word_idx = bit_index >> WORD_SHIFT;
        const size_t bit_offset = bit_index & WORD_MASK;
        return (_data[word_idx] & (1ULL << bit_offset)) != 0;
    }

    /**
     * @brief Direct write to a specific word.
     * @param word_idx The index of the word to write.
     * @param mask The new mask to set for the word.
     * @warning: Safe ONLY if the caller guarantees no other thread writes to this word_idx.
     */
    __attribute__((always_inline))
    void set_word_mask(const size_t word_idx, const word_t mask) {
        // Simple assignment, zero overhead
        _data[word_idx] = mask;
    }

    /**
     * @brief Reads the entire word mask. Useful for fast batch checking.
     * @param word_idx The index of the word to read.
     * @return The word mask at the specified index.
     */
    __attribute__((always_inline))
    word_t get_word_mask(const size_t word_idx) const {
        return _data[word_idx];
    }

    /** @brief Get the total number of underlying words. Used for TBB range loop. */
    __attribute__((always_inline))
    size_t get_num_words() const {
        return _data.size();
    }

    /**
     * @brief Helper: Get the bit range [start, end) corresponding to a word index.
     * @param word_idx The index of the word.
     * @param start_vid Output parameter for the start bit index (inclusive).
     * @param end_vid Output parameter for the end bit index (exclusive).
     */
    __attribute__((always_inline))
    void get_range_from_word(const size_t word_idx, size_t& start_vid, size_t& end_vid) const {
        start_vid = word_idx << WORD_SHIFT;
        // Determine end_vid: usually start + 64, but capped at total bits for the last word
        end_vid = std::min(start_vid + BITS_PER_WORD, _num_bits);
    }

    /** @brief Fast clear. Can be parallelized by caller. */
    __attribute__((always_inline))
    void clear() {
        if (!_data.empty()) {
            std::memset(_data.data(), 0, _data.size() * sizeof(word_t));
        }
    }

    /** @brief Access raw data */
    __attribute__((always_inline))
    const container_t& data() const { return _data; }

private:

    /** @brief Underlying storage for the bitmap words. */
    container_t _data;

    /** @brief Total number of bits in the bitmap. */
    size_t _num_bits = 0;

};  // class WordAlignedBitmap

}   // namespace cpu
}   // namespace artea