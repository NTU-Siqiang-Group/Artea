/*
 * @FilePath: /Artea/include/artea/cpu/utils/random_seq.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-27 11:41:38
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
#include <artea/definitions.hpp>

namespace artea {
namespace cpu {

template <typename vec_num_t>
class RandomSeq {

    using vec_id_t = vec_num_t;

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
     * @param rand_nbrs Reference to the vector where the generated random numbers will be stored.
     * @param num_rand_nbrs The total number of random numbers to generate.
     * @note  Caller must ensure that the size of rand_nbrs is at least num_rand_nbrs.
     */
    auto generate(std::vector<vec_id_t>& rand_nbrs, const vec_num_t num_rand_nbrs) -> void {
        vec_id_t* rand_nbrs_ptr = rand_nbrs.data();

        // Get the MKL stream specific to the current thread.
        // If one doesn't exist yet for this thread, TBB creates it using our factory.
        VSLStreamStatePtr& local_stream = _tl_streams.local();

        viRngUniform(
            VSL_RNG_METHOD_UNIFORM_STD,
            local_stream, num_rand_nbrs,
            reinterpret_cast<int*>(rand_nbrs_ptr),
            0,
            static_cast<int>(_num_vecs)
        );
    }

    /**
     * @brief Generate random numbers in a thread-safe manner.
     *        It automatically uses a random stream unique to the calling thread.
     *        This method can be called by single-threaded or multi-threaded code.
     * @param rand_nbrs_ptr Pointer to the array where the generated random numbers will be stored.
     * @param num_rand_nbrs The total number of random numbers to generate.
     * @note  Caller must ensure that the memory pointed by rand_nbrs_ptr is large enough
     *        to hold num_rand_nbrs elements.
     */
    auto generate(vec_id_t* rand_nbrs_ptr, const vec_num_t num_rand_nbrs) -> void {
        // Get the MKL stream specific to the current thread.
        // If one doesn't exist yet for this thread, TBB creates it using our factory.
        VSLStreamStatePtr& local_stream = _tl_streams.local();

        viRngUniform(
            VSL_RNG_METHOD_UNIFORM_STD,
            local_stream, num_rand_nbrs,
            reinterpret_cast<int*>(rand_nbrs_ptr),
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