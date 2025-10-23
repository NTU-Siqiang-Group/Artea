// bitmap.hpp
#pragma once

#include <artea/types.hpp>
#include <cstdint>

// --- Bitmap struct defined above should be placed here ---
/**
 * @brief A simple and efficient bitmap structure for managing a large number of bits.
 */
class Bitmap {

public:
    /**
     * @brief Constructs a Bitmap with the specified number of bits.
     * @param num_bits The total number of bits to be managed by the bitmap.
     */
    explicit Bitmap(std::size_t num_bits) : 
        _num_bits(num_bits), _num_words((num_bits + 63) >> 6) // _num_words((num_bits + 63) / 64) 
    {
        _data = new std::uint64_t[_num_words]();
    }

    ~Bitmap() {
        delete[] _data;
    }

    /**
     * @brief Sets the bit at the specified index to 1.
     * @param index The index of the bit to set.
     */
    __attribute__((always_inline))
    auto set_bit(std::size_t index) -> void {
        // _data[index / 64] |= (1ULL << (index % 64));
        _data[index >> 6] |= (1ULL << (index & 63));
    }

    /**
     * @brief Checks if the bit at the specified index is set (1).
     * @param index The index of the bit to check.
     * @return True if the bit is set, false otherwise.
     */
    __attribute__((always_inline))
    auto is_bit_set(std::size_t index) -> bool {
        // return (_data[index / 64] & (1ULL << (index % 64))) != 0;
        return (_data[index >> 6] & (1ULL << (index & 63))) != 0;
    }

private:
    /**
     * @brief Number of bits in the bitmap.
     */
    std::size_t _num_bits;
    /**
     * @brief Number of words (64-bit chunks) required to store the bits.
     */
    std::size_t _num_words;
    /**
     * @brief Array of 64-bit words to store the bits.
     */
    std::uint64_t* _data;

};  // class Bitmap