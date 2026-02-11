/*
 * @FilePath: /Artea/include/artea/cpu/utils/random_seq.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-01-25 21:41:16
 * @Date: 2025-11-02 19:41:19
 * @Description: Modified to be thread-safe for parallel execution by using TBB thread-local storage.
 */

#pragma once

#include <random>
#include <stdexcept>
#include <type_traits>

#include <mkl.h>
#include <tbb/enumerable_thread_specific.h>

#include <artea/cpu/containers/vector_array.hpp>
#include <artea/cpu/containers/allocator.hpp>


namespace artea {
namespace cpu {

template <typename BaseTraitsT>
class RandomSeq {

    using vec_num_t = typename BaseTraitsT::vec_num_t;
    using vec_id_t = typename BaseTraitsT::vec_id_t;

    // --- Vectorized Implementation using the thread-local stream ---
    static_assert(sizeof(vec_id_t) == sizeof(int), "MKL vectorized integer generation requires a 32-bit type.");

public:
    /**
     * @brief Construct a new thread-safe RandomSeq object.
     * @param num_vecs The upper bound (exclusive) for the random numbers to be generated.
     */
    RandomSeq(const vec_num_t num_vecs) :
        _num_vecs(num_vecs),
        // Initialize the thread-local storage container.
        // The container will call this factory function once for each thread
        // that executes 'local()' function in 'generate()' method.
        _tl_streams([]() {
            VSLStreamStatePtr stream = nullptr;
            std::random_device rd;
            vslNewStream(&stream, VSL_BRNG_MT19937, (MKL_INT)rd());
            return stream;
        })
    {
    }

    ~RandomSeq() {
        // iterate through all streams created for all threads and delete each one.
        for (auto& stream : _tl_streams) {
            if (stream != nullptr) {
                vslDeleteStream(&stream);
            }
        }
    }

    RandomSeq(const RandomSeq&) = delete;
    RandomSeq& operator=(const RandomSeq&) = delete;
    RandomSeq(RandomSeq&&) = delete;
    RandomSeq& operator=(RandomSeq&&) = delete;

    /**
     * @brief Generate random numbers in a thread-safe manner.
     *        It automatically uses a random stream unique to the calling thread.
     *        This method can be called by single-threaded or multi-threaded code.
     * @param rand_ids Reference to the vector where the generated random numbers will be stored.
     * @param num_rand_ids The total number of random numbers to generate.
     * @note  Caller must ensure that the size of rand_ids is at least num_rand_ids.
     */
    auto generate(std::vector<vec_id_t>& rand_ids, const vec_num_t num_rand_ids) -> void {
        vec_id_t* rand_ids_ptr = rand_ids.data();

        // Get the MKL stream specific to the current thread.
        // If one doesn't exist yet for this thread, TBB creates it using our factory.
        VSLStreamStatePtr& local_stream = _tl_streams.local();

        viRngUniform(
            VSL_RNG_METHOD_UNIFORM_STD,
            local_stream, num_rand_ids,
            reinterpret_cast<int*>(rand_ids_ptr),
            0,
            static_cast<int>(_num_vecs)
        );
    }

    /**
     * @brief Generate random numbers in a thread-safe manner.
     *        It automatically uses a random stream unique to the calling thread.
     *        This method can be called by single-threaded or multi-threaded code.
     * @param rand_ids_ptr Pointer to the array where the generated random numbers will be stored.
     * @param num_rand_ids The total number of random numbers to generate.
     * @note  Caller must ensure that the memory pointed by rand_ids_ptr is large enough
     *        to hold num_rand_ids elements.
     */
    auto generate(vec_id_t* rand_ids_ptr, const vec_num_t num_rand_ids) -> void {
        // Get the MKL stream specific to the current thread.
        // If one doesn't exist yet for this thread, TBB creates it using our factory.
        VSLStreamStatePtr& local_stream = _tl_streams.local();

        viRngUniform(
            VSL_RNG_METHOD_UNIFORM_STD,
            local_stream, num_rand_ids,
            reinterpret_cast<int*>(rand_ids_ptr),
            0,
            static_cast<int>(_num_vecs)
        );
    }

    /**
     * @brief Generate random numbers in a thread-safe manner.
     *        It automatically uses a random stream unique to the calling thread.
     *        This method can be called by single-threaded or multi-threaded code.
     * @param rand_ids Reference to the cache-aligned container where the generated random numbers will be stored.
     * @param num_rand_ids The total number of random numbers to generate.
     * @note  Caller must ensure that the size of rand_ids is at least num_rand_ids.
     */
    auto generate(cache_aligned_container_t<vec_id_t>& rand_ids, const vec_num_t num_rand_ids) -> void {
        vec_id_t* rand_ids_ptr = rand_ids.data();

        // Get the MKL stream specific to the current thread.
        // If one doesn't exist yet for this thread, TBB creates it using our factory.
        VSLStreamStatePtr& local_stream = _tl_streams.local();

        viRngUniform(
            VSL_RNG_METHOD_UNIFORM_STD,
            local_stream, num_rand_ids,
            reinterpret_cast<int*>(rand_ids_ptr),
            0,
            static_cast<int>(_num_vecs)
        );
    }

private:
    /** @brief The upper bound for the random numbers. */
    const vec_num_t _num_vecs;

    /**
     * @brief Thread-local storage for MKL stream pointers.
     * Each thread gets its own VSLStreamStatePtr, managed by this container.
     */
    mutable tbb::enumerable_thread_specific<VSLStreamStatePtr> _tl_streams;

};  // class RandomSeq

}   // namespace cpu
}   // namespace artea