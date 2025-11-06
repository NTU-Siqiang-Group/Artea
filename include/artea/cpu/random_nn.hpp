/*
 * @FilePath: /Artea/include/artea/cpu/random_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-05 19:18:18
 * @Date: 2025-11-02 19:41:19
 * @Description: Refactored to use generation type as a class template parameter.
 */

#pragma once

#include <random>
#include <stdexcept>
#include <type_traits>

#include <artea/types.hpp>
#include <artea/config.hpp>
#include <artea/cpu/array.hpp>
#include <artea/cpu/vector_array.hpp>

// This define (MKL_ENABLED) is set by CMake if Intel MKL is found
#ifdef MKL_ENABLED
#include "mkl.h"
#endif

namespace artea {
namespace cpu {

enum class RandomGenType {
    SERIAL,
    VECTORIZED  // use Intel MKL library for vectorized generation
};

template <
    typename vec_num_t, 
    typename vec_id_t = vec_num_t,
    RandomGenType GenType = RandomGenType::SERIAL
>
class RandomNN {

public:
    /**
     * @brief Construct a new RandomNN object.
     * @param num_vecs The upper bound (exclusive) for the random numbers to be generated.
     */
    RandomNN(const vec_num_t& num_vecs, bool enabled = false) : _num_vecs(num_vecs), _enabled(enabled) {
        // Initialize the appropriate generator based on the class template parameter
        if constexpr (GenType == RandomGenType::SERIAL) {
            std::random_device rd;
            _generator.seed(rd());
        } else if constexpr (GenType == RandomGenType::VECTORIZED) {
#ifdef MKL_ENABLED
            std::random_device rd;
            vslNewStream(&_mkl_stream, VSL_BRNG_MT19937, (MKL_INT)rd());
#else
            // This compile-time error prevents instantiation of the vectorized version without MKL
            static_assert(sizeof(vec_num_t) == 0, "Vectorized RandomNN requires Intel MKL. Compile with MKL support.");
#endif
        }
    }
    
    ~RandomNN() {
        // Clean up MKL stream only if it was created
        if constexpr (GenType == RandomGenType::VECTORIZED) {
#ifdef MKL_ENABLED
            if (_mkl_stream != nullptr) {
                vslDeleteStream(&_mkl_stream);
            }
#endif
        }
    }

    /**
     * @brief Generate random numbers and write them to the provided array.
     *        The implementation (serial or vectorized) is determined by the class's template parameter.
     * @param rand_nbrs Reference to the Array where the generated random numbers will be stored.
     * @param num_rand_nbrs The total number of random numbers to generate.
     */
    __attribute__((always_inline))
    auto generate(Array<vec_id_t>& rand_nbrs, const vec_num_t num_rand_nbrs) -> void {
        if (rand_nbrs.size() < num_rand_nbrs) {
            rand_nbrs.resize(num_rand_nbrs);
        }
        vec_id_t* rand_nbrs_ptr = rand_nbrs.data();

        if constexpr (GenType == RandomGenType::SERIAL) {
            // --- Serial Implementation ---
            std::uniform_int_distribution<vec_id_t> dist(0, _num_vecs - 1);
            for (vec_num_t i = 0; i < num_rand_nbrs; ++i) {
                rand_nbrs_ptr[i] = dist(_generator);
            }
        } else if constexpr (GenType == RandomGenType::VECTORIZED) {
            // --- Vectorized Implementation ---
#ifdef MKL_ENABLED
            static_assert(sizeof(vec_id_t) == sizeof(int), "MKL vectorized integer generation requires a 32-bit integer type.");
            // MKL's viRngUniform generates integers in the interval [a, b).
            // For a range of [0, _num_vecs - 1], the parameters are a=0 and b=_num_vecs.
            viRngUniform(VSL_RNG_METHOD_UNIFORM_STD, _mkl_stream, num_rand_nbrs, reinterpret_cast<int*>(rand_nbrs_ptr), 0, static_cast<int>(_num_vecs));
#endif
        }
    }

    /**
    * @brief Check if the random number generator is enabled.
    * @return true If the random number generator is enabled.
    * @return false If the random number generator is disabled.
    */
    __attribute__((always_inline))
    auto enabled() const -> bool {
        return _enabled;
    }

    /**
    * @brief Disable the random number generator.
    */
    __attribute__((always_inline))
    auto disable() -> void {
        _enabled = false;
    }

    /**
    * @brief Activate the random number generator.
    */
    __attribute__((always_inline))
    auto activate() -> void {
        _enabled = true;
    }

private:
    /** @brief The upper bound for the random numbers. */
    const vec_num_t& _num_vecs;
    /** @brief Whether the random number generator is enabled. */
    bool _enabled = false;

    // --- Member variables for different generator types ---
    // Both are declared, but only one will be initialized and used based on GenType.
    // The compiler will optimize away the unused one.

    /** @brief A single random number generator for serial execution. */
    std::mt19937 _generator;
    
#ifdef MKL_ENABLED
    /** @brief MKL stream for vectorized random number generation. */
    VSLStreamStatePtr _mkl_stream = nullptr;
#endif

};  // class RandomNN

}   // namespace cpu
}   // namespace artea