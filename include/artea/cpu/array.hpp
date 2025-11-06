/*
 * @FilePath: /yeweitang/Artea/include/artea/cpu/array.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2025-11-03
 * @Description: A flexible array structure with a strict resize policy (size == capacity),
 *               using unique_ptr for exclusive ownership and providing non-owning views.
 */

#pragma once

#include <memory>
#include <cstddef>
#include <immintrin.h>
#include <stdexcept>
#include <utility>
#include <algorithm> // For std::min
#include <cstring>   // For std::memmove
#include <omp.h>

#include <artea/types.hpp>
#include <artea/config.hpp>

namespace artea {
namespace cpu {

// Custom deleter for memory allocated by _mm_malloc.
struct AlignedDeleter {
    void operator()(void* p) const {
        _mm_free(p);
    }
};

/**
 * @brief A flexible array structure with a strict resize policy (size == capacity),
 *        using unique_ptr for exclusive ownership and providing non-owning views.
 * @note  The array structure can only manage base data types (POD types) and does not support
 *        complex types with custom constructors or destructors.
 */
template <typename T>
class Array 
{
    
public:
    using data_holder_t = std::unique_ptr<T[], AlignedDeleter>;

    static_assert(std::is_trivially_copyable_v<T>, "This Array Structure only supports POD types.");

    /**
     * @brief Default constructor. Creates an empty Array.
     */
    Array() : _data(nullptr), _length(0), _data_holder(nullptr), _num_element(0) {}

    Array(std::size_t length, std::size_t alignment = 64)
        : _data(nullptr), _length(length), _data_holder(nullptr), _num_element(0) {
        if (length > 0) {
            _data = static_cast<T*>(_mm_malloc(length * sizeof(T), alignment));
            if (!_data) {
                throw std::bad_alloc();
            }
            _data_holder.reset(_data);
        }
    }

    /**
     * @brief Constructor for creating a non-owning view of an existing Array. Views cannot be resized.
     */
    Array(T* data, std::size_t length)
        : _data(data), _length(length), _data_holder(nullptr), _num_element(0) {}

    // --- Ownership Semantics ---
    Array(const Array<T>& right) = delete;
    Array<T>& operator=(const Array<T>& right) = delete;
    Array(Array<T>&& right) noexcept = default;
    Array<T>& operator=(Array<T>&& right) noexcept = default;
    ~Array() = default;

    // --- Accessors ---
    __attribute__((always_inline))
    T& operator[](const std::size_t index) { return _data[index]; }

    __attribute__((always_inline))
    const T& operator[](const std::size_t index) const { return _data[index]; }

    __attribute__((always_inline))
    T* data() const { return _data; }

    __attribute__((always_inline))
    std::size_t size() const { return _length; }

    __attribute__((always_inline))
    std::size_t get_num_element() const { return _num_element; }

     __attribute__((always_inline))
    void set_num_element(std::size_t num_element) { _num_element = num_element; }
    
    __attribute__((always_inline))
    bool empty() const { return _length == 0; }
    
    __attribute__((always_inline))
    bool is_owning() const { return _data_holder != nullptr; }

    void swap(Array<T>& other) noexcept {
        std::swap(_data, other._data);
        std::swap(_length, other._length);
        std::swap(_data_holder, other._data_holder);
        std::swap(_num_element, other._num_element);
    }

    /**
     * @brief Releases ownership and clears the array.
     */
    __attribute__((always_inline))
    void clear() {
        if (!is_owning()) {
            throw std::runtime_error("Cannot clear a non-owning Array view.");
        }

        _data_holder.reset();
        _data = nullptr;
        _length = 0;
    }

    /**
     * @brief Changes the number of elements, reallocating memory to the exact new size.
     * @param new_length The new number of elements.
     */
    void resize(std::size_t new_length, std::size_t alignment = 64) {
        if (!is_owning()) {
            // A non-owning view cannot be resized.
            if (new_length != _length) {
                throw std::runtime_error("Cannot resize a non-owning Array view.");
            }
            return;
        }

        if (new_length == _length) {
            return; // No change needed.
        }

        if (new_length == 0) {
            clear();
            return;
        }

        // Always reallocate to the exact new size.
        T* new_ptr = static_cast<T*>(_mm_malloc(new_length * sizeof(T), alignment));
        if (!new_ptr) {
            throw std::runtime_error("Failed to allocate memory in Array::resize.");
        }
        
        // Determine how many elements to copy from the old array.
        std::size_t elements_to_move = std::min(_length, new_length);
        
        if (_data && elements_to_move > 0) {
            std::memmove(new_ptr, _data, elements_to_move * sizeof(T));
        }
        
        // Take ownership of the new memory and release the old one via unique_ptr's reset.
        _data = new_ptr;
        _length = new_length;
        _data_holder.reset(new_ptr);
    }

    /**
     * @brief Factory function to allocate a new, owning Array with aligned memory.
     */
    __attribute__((always_inline))
    static Array<T> alloc(std::size_t length, std::size_t alignment = 64) {
        if (length == 0) {
            return Array<T>();
        }
        
        T* ptr = static_cast<T*>(_mm_malloc(length * sizeof(T), alignment));
        if (!ptr) {
            throw std::runtime_error("Failed to allocate aligned memory in Array::alloc.");
        }
        
        return Array<T>(ptr, length, data_holder_t(ptr));
    }

private:
    /**
     * @brief Private constructor for internal factory use.
     */
    Array(T* data, std::size_t length, data_holder_t&& holder)
        : _data(data), _length(length), _data_holder(std::move(holder)) {}

    T* _data;
    std::size_t _length;
    data_holder_t _data_holder;
    std::size_t _num_element;

};  // class Array

} // namespace cpu
} // namespace artea
