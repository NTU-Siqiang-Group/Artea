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

#include <artea/definitions.hpp>
#include <artea/cpu/containers/allocator.hpp>

namespace artea {
namespace cpu {
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
    class iterator; // Forward declaration
    class const_iterator; // Forward declaration

    class iterator {

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = T*;
        using reference         = T&;

        friend class const_iterator; // Allow const_iterator to access _ptr

        iterator(pointer ptr = nullptr) : _ptr(ptr) {}

        __attribute__((always_inline))
        auto operator*() const -> reference { return *_ptr; }

        __attribute__((always_inline))
        auto operator->() const -> pointer { return _ptr; }

        __attribute__((always_inline))
        auto operator++() -> iterator& { ++_ptr; return *this; }

        __attribute__((always_inline))
        auto operator++(int) -> iterator { iterator tmp = *this; ++_ptr; return tmp; }

        __attribute__((always_inline))
        auto operator--() -> iterator& { --_ptr; return *this; }

        __attribute__((always_inline))
        auto operator--(int) -> iterator { iterator tmp = *this; --_ptr; return tmp; }

        __attribute__((always_inline))
        auto operator+=(difference_type offset) -> iterator& { _ptr += offset; return *this; }

        __attribute__((always_inline))
        auto operator+(difference_type offset) const -> iterator { return iterator(_ptr + offset); }

        __attribute__((always_inline))
        friend auto operator+(difference_type offset, const iterator& it) -> iterator { return iterator(it._ptr + offset); }

        __attribute__((always_inline))
        auto operator-=(difference_type offset) -> iterator& { _ptr -= offset; return *this; }

        __attribute__((always_inline))
        auto operator-(difference_type offset) const -> iterator { return iterator(_ptr - offset); }

        __attribute__((always_inline))
        auto operator-(const iterator& other) const -> difference_type { return _ptr - other._ptr; }

        __attribute__((always_inline))
        auto operator[](difference_type offset) const -> reference { return _ptr[offset]; }

        __attribute__((always_inline))
        auto operator==(const iterator& other) const -> bool { return _ptr == other._ptr; }

        __attribute__((always_inline))
        auto operator!=(const iterator& other) const -> bool { return _ptr != other._ptr; }

        __attribute__((always_inline))
        auto operator<(const iterator& other) const -> bool { return _ptr < other._ptr; }

        __attribute__((always_inline))
        auto operator>(const iterator& other) const -> bool { return _ptr > other._ptr; }

        __attribute__((always_inline))
        auto operator<=(const iterator& other) const -> bool { return _ptr <= other._ptr; }

        __attribute__((always_inline))
        auto operator>=(const iterator& other) const -> bool { return _ptr >= other._ptr; }

    private:
        pointer _ptr;
    };

    class const_iterator {

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = const T;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const T*;
        using reference         = const T&;

        const_iterator(pointer ptr = nullptr) : _ptr(ptr) {}

        const_iterator(const iterator& other) : _ptr(other._ptr) {}

        __attribute__((always_inline))
        auto operator*() const -> reference { return *_ptr; }

        __attribute__((always_inline))
        auto operator->() const -> pointer { return _ptr; }

        __attribute__((always_inline))
        auto operator++() -> const_iterator& { ++_ptr; return *this; }

        __attribute__((always_inline))
        auto operator++(int) -> const_iterator { const_iterator tmp = *this; ++_ptr; return tmp; }

        __attribute__((always_inline))
        auto operator--() -> const_iterator& { --_ptr; return *this; }

        __attribute__((always_inline))
        auto operator--(int) -> const_iterator { const_iterator tmp = *this; --_ptr; return tmp; }

        __attribute__((always_inline))
        auto operator+=(difference_type offset) -> const_iterator& { _ptr += offset; return *this; }

        __attribute__((always_inline))
        auto operator+(difference_type offset) const -> const_iterator { return const_iterator(_ptr + offset); }

        __attribute__((always_inline))
        friend auto operator+(difference_type offset, const const_iterator& it) -> const_iterator { return const_iterator(it._ptr + offset); }

        __attribute__((always_inline))
        auto operator-=(difference_type offset) -> const_iterator& { _ptr -= offset; return *this; }

        __attribute__((always_inline))
        auto operator-(difference_type offset) const -> const_iterator { return const_iterator(_ptr - offset); }

        __attribute__((always_inline))
        auto operator-(const const_iterator& other) const -> difference_type { return _ptr - other._ptr; }

        __attribute__((always_inline))
        auto operator[](difference_type offset) const -> reference { return _ptr[offset]; }

        __attribute__((always_inline))
        auto operator==(const const_iterator& other) const -> bool { return _ptr == other._ptr; }

        __attribute__((always_inline))
        auto operator!=(const const_iterator& other) const -> bool { return _ptr != other._ptr; }

        __attribute__((always_inline))
        auto operator<(const const_iterator& other) const -> bool { return _ptr < other._ptr; }

        __attribute__((always_inline))
        auto operator>(const const_iterator& other) const -> bool { return _ptr > other._ptr; }

        __attribute__((always_inline))
        auto operator<=(const const_iterator& other) const -> bool { return _ptr <= other._ptr; }

        __attribute__((always_inline))
        auto operator>=(const const_iterator& other) const -> bool { return _ptr >= other._ptr; }

    private:
        pointer _ptr;
    };

public:

    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;

    using data_holder_t = std::unique_ptr<T[], AlignedDeleter>;

    static_assert(std::is_trivially_copyable_v<T>, "This Array Structure only supports POD types.");

    Array() : _data(nullptr), _size(0), _capacity(0), _data_holder(nullptr) {}

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

    Array(T* data, std::size_t size)
        : _data(data), _size(size), _capacity(size), _data_holder(nullptr) {}

    Array(const Array<T>& right) = delete;
    auto operator=(const Array<T>& right) -> Array<T>& = delete;
    Array(Array<T>&& right) noexcept = default;
    auto operator=(Array<T>&& right) noexcept -> Array<T>& = default;
    ~Array() = default;

    __attribute__((always_inline))
    auto begin() -> iterator { return iterator(_data); }

    __attribute__((always_inline))
    auto end() -> iterator { return iterator(_data + _size); }

    __attribute__((always_inline))
    auto begin() const -> const_iterator { return const_iterator(_data); }

    __attribute__((always_inline))
    auto end() const -> const_iterator { return const_iterator(_data + _size); }

    __attribute__((always_inline))
    auto cbegin() const -> const_iterator { return const_iterator(_data); }

    __attribute__((always_inline))
    auto cend() const -> const_iterator { return const_iterator(_data + _size); }

    __attribute__((always_inline))
    auto operator[](const std::size_t index) -> T& { return _data[index]; }

    __attribute__((always_inline))
    auto operator[](const std::size_t index) const -> const T& { return _data[index]; }

    __attribute__((always_inline))
    auto front() -> T& { return _data[0]; }

    __attribute__((always_inline))
    auto front() const -> const T& { return _data[0]; }

    __attribute__((always_inline))
    auto back() -> T& { return _data[_size - 1]; }

    __attribute__((always_inline))
    auto back() const -> const T& { return _data[_size - 1]; }

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

    /** @brief Swap the contents with another Array
      * @param other The other Array to swap with.
     */
    void swap(Array<T>& other) noexcept {
        std::swap(_data, other._data);
        std::swap(_size, other._size);
        std::swap(_capacity, other._capacity);
        std::swap(_data_holder, other._data_holder);
    }


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

    /** @brief Reset all elements to zero.
     */
    __attribute__((always_inline))
    void reset() {
        if (_data && _size > 0) {
            std::memset(_data, 0, _size * sizeof(T));
        }
    }

    /** @brief Reserve capacity for the array.
      * @param new_capacity The new capacity to reserve.
      * @param alignment   The alignment for the allocated memory (default is 64 bytes).
      * @warning  If the array is a non-owning view, reserving more capacity than current
     */
    void reserve(std::size_t new_capacity, std::size_t alignment = 64) {
        // if (!is_owning()) {
        //     if (new_capacity > _capacity) {
        //         throw std::runtime_error("Cannot reserve capacity for a non-owning Array view.");
        //     }
        //     return; // No-op if new_capacity is not greater
        // }

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
            _data_holder.reset(new_ptr); // Releases old memory and takes ownership of new_ptr
        }
    }

    void resize(std::size_t new_size) {
        // if (!is_owning()) {
        //     if (new_size > _capacity) {
        //          throw std::runtime_error("Cannot resize a non-owning Array view beyond its capacity.");
        //     }
        //     _size = new_size; // Allow resizing within capacity
        //     return;
        // }

        if (new_size > _capacity) {
            reserve(std::max(new_size, _capacity > 0 ? _capacity * 2 : (std::size_t)8));
        }

        _size = new_size;
    }

    /** @brief Append an element to the end of the array.
      * @param value The value to append.
      * @warning  This function can only be used on owning Array instances.
    */
    void push_back(const T& value) {
        // if (!is_owning()) {
        //     throw std::runtime_error("Cannot push_back to a non-owning Array view.");
        // }
        if (_size >= _capacity) {
            reserve(_capacity > 0 ? _capacity * 2 : 8);
        }
        _data[_size++] = value;
    }

    /** @brief Insert an element at the specified position.
      * @param pos   The position to insert the element at.
      * @param value The value to insert.
      * @return An iterator pointing to the inserted element.
      * @warning  This function can only be used on owning Array instances.
      * @warning The complexity is linear in the distance to the end of the array.
                 Try to not use it frequently for performance consideration.
    */
    auto insert(const_iterator pos, const T& value) -> iterator {
        // if (!is_owning()) {
        //     throw std::runtime_error("Cannot insert into a non-owning Array view.");
        // }

        difference_type index = pos - cbegin();
        if (index < 0 || (std::size_t)index > _size) {
            throw std::out_of_range("Insert iterator is out of range.");
        }

        if (_size >= _capacity) {
            std::size_t new_cap = _capacity > 0 ? _capacity * 2 : 8;
            // Important: reserve might change _data, so we need to recalculate pointers
            reserve(new_cap);
        }

        T* insert_ptr = _data + index;

        if ((std::size_t)index < _size) {
            std::memmove(insert_ptr + 1, insert_ptr, (_size - index) * sizeof(T));
        }

        *insert_ptr = value;
        _size++;

        return iterator(insert_ptr);
    }

    /** @brief Allocate a new Array with specified size and alignment.
      * @param size      The size of the array to allocate.
      * @param alignment The alignment for the allocated memory (default is 64 bytes).
      * @return A new Array instance with allocated memory.
      * @note   The returned Array owns its memory and will manage its lifetime.
      * @warning  This function can only be used to create owning Array instances.
      * @warning size must be greater than zero.
     */
    __attribute__((always_inline))
    static auto alloc(std::size_t size, std::size_t alignment = 64) -> Array<T> {
        T* ptr = static_cast<T*>(_mm_malloc(size * sizeof(T), alignment));
        if (!ptr) {
            throw std::runtime_error("Failed to allocate aligned memory in Array::alloc.");
        }
        return Array<T>(ptr, size, size, data_holder_t(ptr));
    }

    /** @brief Populate the Array from a given container.
      * @param container The container to copy data from.
      * @note   The Array must be an owning instance to use this function.
     */
    template <typename container_t>
    __attribute__((always_inline))
    auto from(const container_t& container) -> void {
        // if (!is_owning()) {
        //     throw std::runtime_error("Cannot use 'from' on a non-owning Array view.");
        // }
        const std::size_t new_size = container.size();
        resize(new_size);
        std::copy(container.begin(), container.end(), begin());
    }

private:
    Array(T* data, std::size_t size, std::size_t capacity, data_holder_t&& holder)
        : _data(data), _size(size), _capacity(capacity), _data_holder(std::move(holder)) {}

    T* _data;
    std::size_t _size;
    std::size_t _capacity;
    data_holder_t _data_holder;

};  // class Array

} // namespace cpu
} // namespace artea