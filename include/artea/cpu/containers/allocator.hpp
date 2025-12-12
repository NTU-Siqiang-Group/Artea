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
 * @FilePath: /Artea/include/artea/cpu/utils/allocator.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#pragma once

#include <cstdint>
#include <utility>
#include <stdexcept>
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

namespace artea {
namespace cpu {

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

/** @brief Custom deleter for memory allocated by _mm_malloc. */
struct AlignedDeleter {
    void operator()(void* p) const {
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

/* ------ MmapAllocator ------ */

/**
 * @brief Allocator using mmap for large memory pages or manual NUMA control.
 *        Memory is zero-initialized by the OS (implicitly) but physical pages
 *        are allocated on "first-touch".
 * @tparam T The type of elements.
 */
template <typename T>
class MmapAllocator {

public:
    using value_type = T;

    MmapAllocator() noexcept = default;

    template <typename U>
    MmapAllocator(const MmapAllocator<U>&) noexcept {}

    /**
     * @brief Allocates memory using mmap.
     * @param n Number of elements.
     * @return Pointer to allocated memory.
     */
    T* allocate(std::size_t n) {
        if (n > std::size_t(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }

        size_t bytes_to_allocate = n * sizeof(T);

        // Use mmap to allocate memory.
        // PROT_READ | PROT_WRITE: Read and write access.
        // MAP_PRIVATE | MAP_ANONYMOUS: Private memory, not backed by any file.
        // MAP_HUGETLB: Use huge pages if supported.
        // We do NOT use MAP_POPULATE to ensure First-Touch policy works during initialization.
        void* p = mmap(nullptr, bytes_to_allocate,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                       -1, 0);

        if (p == MAP_FAILED) {
            throw std::bad_alloc();
        }

        return static_cast<T*>(p);
    }

    /**
     * @brief Deallocates memory using munmap.
     * @param p Pointer to memory.
     * @param n Number of elements (needed for munmap).
     */
    void deallocate(T* p, std::size_t n) noexcept {
        size_t bytes_to_deallocate = n * sizeof(T);
        munmap(p, bytes_to_deallocate);
    }
};

template <typename T, typename U>
bool operator==(const MmapAllocator<T>&, const MmapAllocator<U>&) noexcept {
    return true;
}

template <typename T, typename U>
bool operator!=(const MmapAllocator<T>&, const MmapAllocator<U>&) noexcept {
    return false;
}

constexpr std::size_t AVX512_ALIGNMENT = 64;

constexpr std::size_t CACHE_LINE_SIZE = 64;

/** @brief a container with AVX-512 alignment */
template <typename T>
using avx512_container_t = std::vector<T, cpu::AlignedAllocator<T, AVX512_ALIGNMENT>>;

/** @brief a container with cache line alignment */
template <typename T>
using cache_aligned_container_t = std::vector<T, cpu::AlignedAllocator<T, CACHE_LINE_SIZE>>;

/** @brief a container with mmap allocator */
template <typename T>
using mmap_container_t = std::vector<T, cpu::MmapAllocator<T>>;

}   // namespace cpu
}   // namespace artea