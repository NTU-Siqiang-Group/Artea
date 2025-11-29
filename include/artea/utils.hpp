/*
 * @FilePath: /Artea/include/artea/utils.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-15 14:15:57
 * @Date: 2025-11-13 19:59:17
 * @Description:
 */

#pragma once

#include <cstdint>
#include <utility>
#include <stdexcept>
#include <immintrin.h>

namespace artea {

/** @brief Custom deleter for memory allocated by _mm_malloc. */
struct AlignedDeleter {
    void operator()(void* p) const {
        _mm_free(p);
    }
};

/**
 * @brief An allocator that uses _mm_malloc and _mm_free for aligned memory allocation,
 *        suitable for SIMD operations (SSE, AVX, AVX-512).
 * @tparam T The type of the elements to be allocated.
 * @tparam Alignment The alignment requirement in bytes. Must be a power of two.
 */
template <typename T, std::size_t Alignment>
class AlignedAllocator {
public:
    // Ensure alignment is a power of two, a requirement for _mm_malloc.
    static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two.");

    using value_type = T;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    /**
     * @brief Allocates memory using _mm_malloc.
     * @param n The number of objects to allocate storage for.
     * @return A pointer to the aligned, allocated memory.
     * @throws std::bad_alloc if the allocation fails.
     */
    T* allocate(std::size_t n) {
        if (n > std::size_t(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }

        size_t bytes_to_allocate = n * sizeof(T);
        T* p = static_cast<T*>(_mm_malloc(bytes_to_allocate, Alignment));

        if (!p) {
            throw std::bad_alloc();
        }

        return p;
    }

    /**
     * @brief Deallocates memory using _mm_free.
     * @param p Pointer to the memory to deallocate.
     */
    void deallocate(T* p, std::size_t /*n*/) noexcept {
        _mm_free(p);
    }
};

/** @brief Comparison operators for AlignedAllocator */
template <typename T1, std::size_t A1, typename T2, std::size_t A2>
bool operator==(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept {
    return A1 == A2;
}

template <typename T1, std::size_t A1, typename T2, std::size_t A2>
bool operator!=(const AlignedAllocator<T1, A1>&, const AlignedAllocator<T2, A2>&) noexcept {
    return A1 != A2;
}



}   // namespace artea