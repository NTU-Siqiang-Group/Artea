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
 * @FilePath: /Artea/include/artea/cpu/containers/thread_local_bitmap.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <artea/cpu/containers/allocator.hpp>
#include <artea/cpu/router/visited_table_concept.hpp>

namespace artea {
namespace cpu {

/**
 * @brief A high-performance single-owner bitmap for thread-local use.
 *
 * Design Philosophy:
 * This bitmap is owned entirely by a single thread. No atomics, no locks,
 * no word-partitioning overhead. The raw pointer + size layout gives the
 * compiler maximum freedom to optimize: the pointer is stored in a register
 * across a hot loop, and each set/reset/test compiles down to a single
 * BTS/BTR/BT instruction on x86-64 (or equivalent shift+mask on other ISAs).
 *
 * Memory is allocated via _mm_malloc with cache-line alignment to avoid
 * false sharing when multiple threads each own their own bitmap.
 */
class ThreadLocalBitmap {

public:
    using word_t = uint64_t;
    static constexpr size_t BITS_PER_WORD = 64;
    static constexpr size_t WORD_SHIFT = 6;
    static constexpr word_t WORD_MASK = 0x3F;

    ThreadLocalBitmap() = default;

    explicit ThreadLocalBitmap(const size_t num_bits) {
        resize(num_bits);
    }

    ~ThreadLocalBitmap() {
        _mm_free(_data);
    }

    // Non-copyable: single-owner semantics.
    ThreadLocalBitmap(const ThreadLocalBitmap&) = delete;
    ThreadLocalBitmap& operator=(const ThreadLocalBitmap&) = delete;

    // Movable.
    ThreadLocalBitmap(ThreadLocalBitmap&& other) noexcept
        : _data(other._data), _num_words(other._num_words), _num_bits(other._num_bits)
    {
        other._data = nullptr;
        other._num_words = 0;
        other._num_bits = 0;
    }

    ThreadLocalBitmap& operator=(ThreadLocalBitmap&& other) noexcept {
        if (this != &other) {
            _mm_free(_data);
            _data = other._data;
            _num_words = other._num_words;
            _num_bits = other._num_bits;
            other._data = nullptr;
            other._num_words = 0;
            other._num_bits = 0;
        }
        return *this;
    }

    /**
     * @brief Resize and clear the bitmap.
     * @param num_bits The new size in bits.
     */
    void resize(const size_t num_bits) {
        _mm_free(_data);
        _num_bits = num_bits;
        _num_words = (num_bits + BITS_PER_WORD - 1) >> WORD_SHIFT;
        const size_t bytes = _num_words * sizeof(word_t);
        _data = static_cast<word_t*>(_mm_malloc(bytes, CACHE_LINE_SIZE));
        std::memset(_data, 0, bytes);
    }

    /**
     * @brief Set a specific bit to 1.
     * @param bit_index The index of the bit to set.
     */
    __attribute__((always_inline))
    void set(const size_t bit_index) {
        _data[bit_index >> WORD_SHIFT] |= (1ULL << (bit_index & WORD_MASK));
    }

    /**
     * @brief Clear a specific bit to 0.
     * @param bit_index The index of the bit to clear.
     */
    __attribute__((always_inline))
    void reset(const size_t bit_index) {
        _data[bit_index >> WORD_SHIFT] &= ~(1ULL << (bit_index & WORD_MASK));
    }

    /**
     * @brief Check if a specific bit is set.
     * @param bit_index The index of the bit to check.
     * @return true if the bit is set, false otherwise.
     */
    __attribute__((always_inline))
    bool test(const size_t bit_index) const {
        return (_data[bit_index >> WORD_SHIFT] & (1ULL << (bit_index & WORD_MASK))) != 0;
    }

    /**
     * @brief Set a bit and return its previous value. Useful for visited-set patterns:
     *        if (!bitmap.test_and_set(id)) { // first visit ... }
     * @param bit_index The index of the bit.
     * @return true if the bit was already set, false if it was 0 (and is now 1).
     */
    __attribute__((always_inline))
    bool test_and_set(const size_t bit_index) {
        word_t& word = _data[bit_index >> WORD_SHIFT];
        const word_t mask = 1ULL << (bit_index & WORD_MASK);
        const bool was_set = (word & mask) != 0;
        word |= mask;
        return was_set;
    }

    /**
     * @brief Clear all set bits using memset.
     */
    __attribute__((always_inline))
    void clear() {
        std::memset(_data, 0, _num_words * sizeof(word_t));
    }

    /** @brief Get the total number of bits. */
    __attribute__((always_inline))
    size_t num_bits() const { return _num_bits; }

    /** @brief Get the number of underlying words. */
    __attribute__((always_inline))
    size_t num_words() const { return _num_words; }

    /** @brief Access raw data pointer. */
    __attribute__((always_inline))
    const word_t* data() const { return _data; }

    __attribute__((always_inline))
    word_t* data() { return _data; }

    /**
     * @brief Find the index of the k-th set bit (0-indexed) using Broadword Programming.
     *
     * This operation is also known as "select" in succinct data structures.
     * Uses PDEP/PEXT instructions on x86-64 (BMI2) or efficient bit manipulation fallback.
     *
     * @param k The rank (0-indexed): 0 means first set bit, 1 means second set bit, etc.
     * @return The bit index of the k-th set bit, or num_bits() if fewer than k+1 bits are set.
     * @complexity O(num_words) in worst case, but typically O(1) with hardware support.
     *
     * Example: bitmap = 0b10010100, select(0) = 2, select(1) = 4, select(2) = 7
     */
    __attribute__((always_inline))
    size_t select(size_t k) const {
        size_t count = 0;

        for (size_t word_idx = 0; word_idx < _num_words; ++word_idx) {
            word_t word = _data[word_idx];
            const size_t popcount = __builtin_popcountll(word);

            if (count + popcount > k) {
                // The k-th set bit is in this word
                const size_t target_rank = k - count;

                // Find the target_rank-th set bit in this word using broadword programming
                size_t bit_pos = _select_in_word(word, target_rank);
                return (word_idx << WORD_SHIFT) + bit_pos;
            }

            count += popcount;
        }

        // Not enough set bits
        return _num_bits;
    }

    /**
     * @brief Find the index of the first set bit starting from start_bit (inclusive).
     *
     * @param start_bit The bit index to start searching from.
     * @return The bit index of the first set bit >= start_bit, or num_bits() if none found.
     * @complexity O(num_words) worst case, typically O(1) with sparse bitmaps.
     */
    __attribute__((always_inline))
    size_t find_first_set(size_t start_bit = 0) const {
        if (start_bit >= _num_bits) {
            return _num_bits;
        }

        size_t word_idx = start_bit >> WORD_SHIFT;
        size_t bit_offset = start_bit & WORD_MASK;

        // Check first word (may need to mask off lower bits)
        word_t word = _data[word_idx] & (~0ULL << bit_offset);
        if (word != 0) {
            return (word_idx << WORD_SHIFT) + __builtin_ctzll(word);
        }

        // Check remaining words
        for (++word_idx; word_idx < _num_words; ++word_idx) {
            word = _data[word_idx];
            if (word != 0) {
                return (word_idx << WORD_SHIFT) + __builtin_ctzll(word);
            }
        }

        return _num_bits;
    }

private:

    word_t* _data = nullptr;
    size_t _num_words = 0;
    size_t _num_bits = 0;

    /**
     * @brief Find the rank-th set bit (0-indexed) within a single 64-bit word.
     *
     * Uses Broadword Programming technique with PDEP instruction on BMI2-capable CPUs,
     * or efficient bit manipulation fallback.
     *
     * @param word The 64-bit word to search.
     * @param rank The target rank (0 = first set bit, 1 = second set bit, etc.).
     * @return The bit position (0-63) of the rank-th set bit.
     */
    __attribute__((always_inline))
    static size_t _select_in_word(word_t word, size_t rank) {
#if defined(__BMI2__) && defined(__x86_64__)
        // Use PDEP instruction for fast select on BMI2-capable CPUs
        // PDEP deposits bits from source into positions indicated by mask
        // By using (1ULL << rank) as source and word as mask, we get the position
        return __builtin_ctzll(_pdep_u64(1ULL << rank, word));
#else
        // Fallback: iterate through set bits
        for (size_t i = 0; i <= rank; ++i) {
            // Clear the lowest set bit
            size_t pos = __builtin_ctzll(word);
            if (i == rank) {
                return pos;
            }
            word &= word - 1;  // Clear lowest set bit
        }
        return 64;  // Should not reach here if rank is valid
#endif
    }

};  // class ThreadLocalBitmap

// Verify the bitmap satisfies the VisitedTable concept
static_assert(VisitedTable<ThreadLocalBitmap>, "ThreadLocalBitmap must satisfy VisitedTable concept");

}   // namespace cpu
}   // namespace artea
