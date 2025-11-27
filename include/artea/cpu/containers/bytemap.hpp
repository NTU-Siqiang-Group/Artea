#pragma once

#include <cstdint>

#include <artea/cpu/containers/array.hpp>
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

// --- Bytemap struct defined above should be placed here ---
/**
 * @brief A simple and efficient bytemap structure.
 */
class Bytemap {

public:

    Bytemap() = default;

    /**
     * @brief Constructs a Bytemap with the specified number of bytes.
     * @param num_bytes The total number of bytes to be managed by the bytemap.
     */
    Bytemap(std::size_t num_bytes) : _num_bytes(num_bytes) {
        _data = Array<bool>::alloc(_num_bytes);
    }

    // --- Ownership Semantics ---
    Bytemap(const Bytemap& right) = delete;
    auto operator=(const Bytemap& right) -> Bytemap& = delete;
    Bytemap(Bytemap&& right) = default;
    auto operator=(Bytemap&& right) -> Bytemap& = default;
    ~Bytemap() = default;

    __attribute__((always_inline))
    auto num_bytes() const -> std::size_t {
        return _num_bytes;
    }

    __attribute__((always_inline))
    auto resize(std::size_t num_bytes) -> void {
        _num_bytes = num_bytes;
        _data.resize(num_bytes);
    }

    /**
     * @brief Sets the bype at the specified index to 1.
     * @param index The index of the byte to set.
     */
    __attribute__((always_inline))
    auto set_byte(std::size_t index) -> void {
        // _data[index / 64] |= (1ULL << (index % 64));
        _data[index] = 1;
    }

    /**
     * @brief Resets the byte at the specified index to 0.
     * @param index The index of the byte to reset.
     */
    __attribute__((always_inline))
    auto reset_byte(std::size_t index) -> void {
        _data[index] = 0;
    }

    /**
     * @brief Resets all bytes in the bytemap to 0.
     */
    __attribute__((always_inline))
    auto reset() -> void {
        _data.reset();
    }

private:

    /** @brief Number of bytes in the bytemap. */
    std::size_t _num_bytes;

    /** @brief Array of booleans to store the bytemap data. */
    Array<bool> _data;

};  // class Bytemap


}   // namespace cpu
}   // namespace artea