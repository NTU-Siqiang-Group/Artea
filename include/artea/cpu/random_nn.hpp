/*
 * @FilePath: /Artea/include/artea/cpu/random_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-02 20:45:27
 * @Date: 2025-11-02 19:41:19
 * @Description: 
 */

#pragma once

#include <cstdint>

#include <artea/types.hpp>
#include <artea/config.hpp>
#include <artea/cpu/vector_array.hpp>

template <
    typename vec_num_t, 
    typename vec_id_t = vec_num_t
>
class RandomNN {

public:
    /**
     * @brief Construct a new RandomNN object.
     * @param num_vecs The upper bound (exclusive) for the random numbers to be generated.
     */
    RandomNN(const vec_num_t& num_vecs) : _num_vecs(num_vecs) {
        // Initialize the random number generators for each thread.
        // A hardware-based random device to seed the master generator.
        std::random_device rd;
        std::mt19937 master_gen(rd());
        // A distribution to generate seeds for each thread's generator.
        std::uniform_int_distribution<unsigned int> seed_dist;
        
        _generators.resize(random_nn_threads);
        // Seed each thread's personal random number generator.
        for (uint32_t i = 0; i < random_nn_threads; ++i) {
            _generators[i].seed(seed_dist(master_gen));
        }
    }
    
    ~RandomNN() = default;

    /**
     * @brief Generate random numbers in parallel and write them to the provided array.
     * @param rand_nbrs Pointer to the array where the generated random numbers will be stored.
     * @param num_rand_nbrs The total number of random numbers to generate.
     */
    auto generate(vec_id_t* rand_nbrs, const vec_num_t num_rand_nbrs) -> void {    
        // A uniform distribution to generate random numbers in the range [0, _num_vecs - 1].
        std::uniform_int_distribution<vec_id_t> dist(0, _num_vecs - 1);
        
        #pragma omp parallel num_threads(random_nn_threads)
        {
            int thread_id = omp_get_thread_num();
            #pragma omp for
            for (vec_num_t i = 0; i < num_rand_nbrs; ++i) {
                rand_nbrs[i] = dist(_generators[thread_id]);
            }
        }
    }

private:
    /** @brief The upper bound for the random numbers. */
    const vec_num_t& _num_vecs;
    /** @brief A vector to hold a separate random number generator for each thread. */
    std::vector<std::mt19937> _generators;

};  // class RandomNN