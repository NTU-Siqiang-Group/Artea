#pragma once

#include <artea/types.hpp>
#include <random>
#include <cstdint>
#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cstddef>

namespace artea {
namespace cpu {

// 
template <uint32_t num_func_sets, uint32_t num_hash_funcs, uint32_t vec_dim>
class HashFuncsTable {


private:
    /**
     * @brief Array of hash functions.
     */
    std::uint64_t _hash_funcs[num_func_sets][num_hash_funcs][vec_dim];

};  // class HashFuncsTable

template <typename vertex_id_t>
class ReorderTable {

public:

    explicit ReorderTable(std::size_t num_vertices) 
        : _origin_ids(new vertex_id_t[num_vertices]) {
        // do nothing
    }

    ~ReorderTable() {
        delete[] _origin_ids;
    }

    __attribute__((always_inline))
    auto get_origin_id(vertex_id_t reordered_id) -> vertex_id_t {
        return _origin_ids[reordered_id];
    }

    __attribute__((always_inline))
    auto set_reordered_id(vertex_id_t reordered_id, vertex_id_t origin_id) -> void {
        _origin_ids[reordered_id] = origin_id;
    }

private:
    vertex_id_t* _origin_ids;

};  // class ReorderTable

template <typename vec_ele_t, vec_dim_t vec_dim>
class LSHBase {

public:

    /**
     * @brief Infers a result for a single input vector.
     * @param vec_data Pointer to the single vector's data.
     * @return The inferred bucket ID.
     */
    virtual auto infer(const vec_ele_t* vec_data) -> bucket_id_t = 0;

    /**
     * @brief Builds an index or model based on the entire provided dataset.
     * @param all_vec_data Pointer to the flat array of vector data.
     * @param num_vectors The total number of vectors in the dataset.
     */
    virtual auto build(const vec_ele_t* all_vec_data, const size_t num_vectors) -> void = 0;

};  // class LSHBase

/**
 * @brief UniBucketLSH maps each vector to a unified-size bucket using the "Projection-Sorted LSH" method.
 *
 * This is a batch (transductive) algorithm. You must call `build()` on the entire dataset
 * before querying the bucket ID for any individual vector using `get_bucket_id()`.
 * 
 * @tparam vec_ele_t The type of the vector elements (e.g., float).
 * @tparam vec_dim The dimension of the vector.
 * @tparam num_hash_funcs The number of hash functions (m), determining the hash code length.
 * @tparam bucket_size The fixed size of each bucket (block-sz).
 */
template <
    typename vec_ele_t, 
    vec_dim_t vec_dim, 
    uint32_t num_hash_funcs,
    uint32_t bucket_size
>
class UniBucketLSH : public LSHBase<vec_ele_t, vec_dim> {

    static_assert(num_hash_funcs > 0, "Number of hash functions must be greater than 0.");
    static_assert(bucket_size > 0, "Bucket size must be greater than 0.");

public:
    UniBucketLSH() {
        
    }

    void infer(const vec_ele_t* vec_data) -> bucket_id_t override {

    }
    
    /**
    * @brief Builds the LSH index for the entire dataset using the projection-sorting method.
    * @param all_vec_data A pointer to the flat array of vector data.
    * @param num_vectors The total number of vectors in the dataset.
    */
    void build(const vec_ele_t* all_vec_data, const size_t num_vectors) override {
        
    }

private:
    


};  // class UniBucketLSH





}   // namespace cpu
}   // namespace artea