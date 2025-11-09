/*
 * @FilePath: /Artea/include/artea/cpu/array.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2025-11-03
 * @Description: A flexible, dynamically growing array structure, using unique_ptr for 
 *               exclusive ownership and providing non-owning views.
 *               Now with full C++ Standard Library random access iterator support.
 */

#pragma once

#include <memory>
#include <cstddef>
#include <immintrin.h>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <mutex>

#include <tbb/concurrent_vector.h>

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
 * @brief A flexible, dynamically growing array structure using unique_ptr for exclusive ownership.
 * @note  The array structure can only manage base data types (POD types) and does not support
 *        complex types with custom constructors or destructors.
 */
template <typename T>
class Array 
{
    
public:
    // --- Iterator Support ---
    class iterator {
    public:
        // C++ standard iterator traits
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = T*;
        using reference         = T&;

        // Constructor
        iterator(pointer ptr = nullptr) : _ptr(ptr) {}

        // Dereferencing
        auto operator*() const -> reference { return *_ptr; }
        auto operator->() const -> pointer { return _ptr; }

        // Increment and decrement
        auto operator++() -> iterator& { ++_ptr; return *this; }
        auto operator++(int) -> iterator { iterator tmp = *this; ++_ptr; return tmp; }
        auto operator--() -> iterator& { --_ptr; return *this; }
        auto operator--(int) -> iterator { iterator tmp = *this; --_ptr; return tmp; }

        // Random access arithmetic
        auto operator+=(difference_type offset) -> iterator& { _ptr += offset; return *this; }
        auto operator+(difference_type offset) const -> iterator { return iterator(_ptr + offset); }
        friend auto operator+(difference_type offset, const iterator& it) -> iterator { return iterator(it._ptr + offset); }
        
        auto operator-=(difference_type offset) -> iterator& { _ptr -= offset; return *this; }
        auto operator-(const iterator& other) const -> iterator { return iterator(_ptr - other._ptr); }
        auto operator-(const iterator& other) const -> difference_type { return _ptr - other._ptr; }

        // Subscript operator
        auto operator[](difference_type offset) const -> reference { return _ptr[offset]; }

        // Comparison operators
        auto operator==(const iterator& other) const -> bool { return _ptr == other._ptr; }
        auto operator!=(const iterator& other) const -> bool { return _ptr != other._ptr; }
        auto operator<(const iterator& other) const -> bool { return _ptr < other._ptr; }
        auto operator>(const iterator& other) const -> bool { return _ptr > other._ptr; }
        auto operator<=(const iterator& other) const -> bool { return _ptr <= other._ptr; }
        auto operator>=(const iterator& other) const -> bool { return _ptr >= other._ptr; }

    private:
        pointer _ptr;
    };

    class const_iterator {
    public:
        // C++ standard iterator traits
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = const T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const T*;
        using reference         = const T&;

        // Constructor
        const_iterator(pointer ptr = nullptr) : _ptr(ptr) {}

        // Dereferencing
        auto operator*() const -> reference { return *_ptr; }
        auto operator->() const -> pointer { return _ptr; }

        // Increment and decrement
        auto operator++() -> const_iterator& { ++_ptr; return *this; }
        auto operator++(int) -> const_iterator { const_iterator tmp = *this; ++_ptr; return tmp; }
        auto operator--() -> const_iterator& { --_ptr; return *this; }
        auto operator--(int) -> const_iterator { const_iterator tmp = *this; --_ptr; return tmp; }

        // Random access arithmetic
        auto operator+=(difference_type offset) -> const_iterator& { _ptr += offset; return *this; }
        auto operator+(difference_type offset) const -> const_iterator { return const_iterator(_ptr + offset); }
        friend auto operator+(difference_type offset, const const_iterator& it) -> const_iterator { return const_iterator(it._ptr + offset); }
        
        auto operator-=(difference_type offset) -> const_iterator& { _ptr -= offset; return *this; }
        auto operator-(difference_type offset) const -> const_iterator { return const_iterator(_ptr - offset); }
        auto operator-(const const_iterator& other) const -> difference_type { return _ptr - other._ptr; }

        // Subscript operator
        auto operator[](difference_type offset) const -> reference { return _ptr[offset]; }

        // Comparison operators
        auto operator==(const const_iterator& other) const -> bool { return _ptr == other._ptr; }
        auto operator!=(const const_iterator& other) const -> bool { return _ptr != other._ptr; }
        auto operator<(const const_iterator& other) const -> bool { return _ptr < other._ptr; }
        auto operator>(const const_iterator& other) const -> bool { return _ptr > other._ptr; }
        auto operator<=(const const_iterator& other) const -> bool { return _ptr <= other._ptr; }
        auto operator>=(const const_iterator& other) const -> bool { return _ptr >= other._ptr; }

    private:
        pointer _ptr;
    };


public:
    using data_holder_t = std::unique_ptr<T[], AlignedDeleter>;

    static_assert(std::is_trivially_copyable_v<T>, "This Array Structure only supports POD types.");

    /**
     * @brief Default constructor. Creates an empty Array.
     */
    Array() : _data(nullptr), _size(0), _capacity(0), _data_holder(nullptr) {}
    
    /**
     * @brief Constructs an array with a given capacity, but size 0.
     */
    explicit Array(std::size_t capacity, std::size_t alignment = 64)
        : _data(nullptr), _size(0), _capacity(capacity), _data_holder(nullptr) {
        if (capacity > 0) {
            _data = static_cast<T*>(_mm_malloc(capacity * sizeof(T), alignment));
            if (!_data) {
                throw std::bad_alloc();
            }
            _data_holder.reset(_data);
        }
    }

    /**
     * @brief Constructor for creating a non-owning view of an existing Array. Views cannot be modified.
     */
    Array(T* data, std::size_t size)
        : _data(data), _size(size), _capacity(size), _data_holder(nullptr) {}

    // --- Ownership Semantics ---
    Array(const Array<T>& right) = delete;
    auto operator=(const Array<T>& right) -> Array<T>& = delete;
    Array(Array<T>&& right) noexcept = default;
    auto operator=(Array<T>&& right) noexcept -> Array<T>& = default;
    ~Array() = default;

    // --- Iterator Access ---
    auto begin() -> iterator { return iterator(_data); }
    auto end() -> iterator { return iterator(_data + _size); }
    auto begin() const -> const_iterator { return const_iterator(_data); }
    auto end() const -> const_iterator { return const_iterator(_data + _size); }
    auto cbegin() const -> const_iterator { return const_iterator(_data); }
    auto cend() const -> const_iterator { return const_iterator(_data + _size); }

    // --- Accessors ---
    __attribute__((always_inline))
    auto operator[](const std::size_t index) -> T& { return _data[index]; }

    __attribute__((always_inline))
    auto operator[](const std::size_t index) const -> const T& { return _data[index]; }

    __attribute__((always_inline))
    auto data() const -> T* { return _data; }

    __attribute__((always_inline))
    auto size() const -> std::size_t { return _size; }
    
    __attribute__((always_inline))
    auto capacity() const -> std::size_t { return _capacity; }

    __attribute__((always_inline))
    auto empty() const -> bool { return _size == 0; }
    
    __attribute__((always_inline))
    auto is_owning() const -> bool { return _data_holder != nullptr; }

    void swap(Array<T>& other) noexcept {
        std::swap(_data, other._data);
        std::swap(_size, other._size);
        std::swap(_capacity, other._capacity);
        std::swap(_data_holder, other._data_holder);
    }

    /**
     * @brief Releases ownership and clears the array. Size and capacity become 0.
     */
    __attribute__((always_inline))
    void clear() {
        if (!is_owning()) {
            _data = nullptr;
            _size = 0;
            _capacity = 0;
        } else {
            _data_holder.reset();
            _data = nullptr;
            _size = 0;
            _capacity = 0;
        }
    }
    
    /**
     * @brief Requests that the array capacity be at least enough to contain new_capacity elements.
     * @param new_capacity New capacity of the array.
     * @param alignment Alignment of the array.
     */
    void reserve(std::size_t new_capacity, std::size_t alignment = 64) {
        if (!is_owning()) {
            throw std::runtime_error("Cannot reserve capacity for a non-owning Array view.");
        }

        if (new_capacity > _capacity) {
            T* new_ptr = static_cast<T*>(_mm_malloc(new_capacity * sizeof(T), alignment));
            if (!new_ptr) {
                throw std::runtime_error("Failed to allocate memory in Array::reserve.");
            }
            
            if (_data && _size > 0) {
                std::memmove(new_ptr, _data, _size * sizeof(T));
            }
            
            _data = new_ptr;
            _capacity = new_capacity;
            _data_holder.reset(new_ptr);
        }
    }
    
    /**
     * @brief Resizes the array to contain new_size elements.
     *        If new_size is smaller than the current size, the content is reduced.
     *        If new_size is greater, the array is expanded, but new elements are uninitialized.
     * @param new_size The new size of the array.
     */
    void resize(std::size_t new_size) {
        if (!is_owning()) {
            if (new_size != _size) {
                 throw std::runtime_error("Cannot resize a non-owning Array view.");
            }
        } else {
            if (new_size > _capacity) {
                // Grow capacity, typically by doubling or to the required new_size.
                reserve(std::max(new_size, _capacity > 0 ? _capacity * 2 : (std::size_t)8));
            }
            _size = new_size;
        }
    }

    /**
     * @brief Adds an element to the end of the array.
     * @param value The value to append.
     * @warning If the array is not owning, the behavior is undefined.
     */
    void push_back(const T& value) {
        // if (!is_owning()) {
        //     throw std::runtime_error("Cannot push_back to a non-owning Array view.");
        // }
        if (_size >= _capacity) {
            reserve(_capacity > 0 ? _capacity * 2 : 8); // Double the capacity or start with 8.
        }
        _data[_size++] = value;
    }
    
    /**
     * @brief Inserts an element at a specified position.
     * @param pos Iterator to the position where the new element will be inserted.
     * @param value The value to insert.
     * @return An iterator pointing to the newly inserted element.
     * @warning If the array is not owning, the behavior is undefined.
     * @note The performance of this operation is linear in the distance to pos.
     */
    auto insert(const_iterator pos, const T& value) -> iterator {
        if (!is_owning()) {
            throw std::runtime_error("Cannot insert into a non-owning Array view.");
        }

        // Calculate insertion index
        difference_type index = pos - cbegin();
        if (index < 0 || (size_type)index > _size) {
            throw std::out_of_range("Insert iterator is out of range.");
        }

        // Ensure there is enough capacity
        if (_size >= _capacity) {
            size_type new_cap = _capacity > 0 ? _capacity * 2 : 8;
            reserve(new_cap);
        }

        // Get a non-const pointer to the insertion point (after potential reallocation)
        T* insert_ptr = _data + index;

        // Shift existing elements to the right
        if ((size_type)index < _size) {
            std::memmove(insert_ptr + 1, insert_ptr, (_size - index) * sizeof(T));
        }

        // Insert the new element
        *insert_ptr = value;
        _size++;
        
        return iterator(insert_ptr);
    }

    // --- Factory Functions ---
    /**
     * @brief Factory function to allocate a new, owning Array with aligned memory.
     *        The created array has its size equal to its capacity.
     */
    __attribute__((always_inline))
    static auto alloc(std::size_t size, std::size_t alignment = 64) -> Array<T> {
        // if (size == 0) {
        //     return Array<T>();
        // }
        
        T* ptr = static_cast<T*>(_mm_malloc(size * sizeof(T), alignment));
        if (!ptr) {
            throw std::runtime_error("Failed to allocate aligned memory in Array::alloc.");
        }
        
        // This private constructor is needed for the factory pattern
        return Array<T>(ptr, size, size, data_holder_t(ptr));
    }

    /**
     * @brief Factory function to create an Array view from a tbb::concurrent_vector.
     *        The created array is non-owning, and its size is equal to the vector's size.
     * @warning This function is not thread safe and can only be used in single-threaded context.
     */
    __attribute__((always_inline))
    static auto from(const tbb::concurrent_vector<T>& vec) -> Array<T> {
        return _from_tbb_concurrent_vector(vec);
    }

private:

    static auto _from_tbb_concurrent_vector(const tbb::concurrent_vector<T>& vec) -> Array<T> {
        const std::size_t size = vec.size();
        Array<T> new_array = Array<T>::alloc(size);
        std::copy(vec.begin(), vec.end(), new_array.begin());
        return new_array;
    }

    /**
     * @brief Private constructor for internal factory use.
     */
    Array(T* data, std::size_t size, std::size_t capacity, data_holder_t&& holder)
        : _data(data), _size(size), _capacity(capacity), _data_holder(std::move(holder)) {}

    T* _data;
    std::size_t _size;
    std::size_t _capacity;
    data_holder_t _data_holder;

};  // class Array

} // namespace cpu
} // namespace artea