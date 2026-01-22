/*
 * @FilePath: /Artea/include/artea/cpu/containers/vector_array.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-01-22 16:48:58
 * @Date: 2025-10-18 16:31:57
 * @Description:
 */


#pragma once

#include <vector>
#include <fstream>
#include <string>
#include <stdexcept>
#include <omp.h>
#include <utility> // For std::move
#include <artea/cpu/containers/allocator.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename VertexNumT, typename VecEleT>
class VectorArray {

    using vec_num_t = VertexNumT;
    using vec_dim_t = uint32_t;
    using vec_id_t = VertexNumT;
    using vec_ele_t = VecEleT;

public:
    /**
     * @brief Default constructor. Creates an empty VectorArray.
     */
    VectorArray() : _num_vecs(0), _vec_dim(0) {}

    /**
     * @brief Construct a new VectorArray object with a pre-defined size, using aligned memory.
     * @param num_vecs The total number of vectors the pool will manage.
     * @param dim The dimension of each vector.
     */
    VectorArray(vec_num_t num_vecs, vec_dim_t dim)
        : _num_vecs(num_vecs), _vec_dim(dim)
    {
        // if (num_vecs > 0 && dim > 0) {
        //     std::size_t total_elements = static_cast<std::size_t>(num_vecs) * dim;
        //     _storage.resize(total_elements);
        // }
        // else {
        //     logger.warn("VectorArray initialized with zero size or dimension.");
        // }

        std::size_t total_elements = static_cast<std::size_t>(num_vecs) * dim;
        _storage.resize(total_elements);
    }

    /**
     * @brief Construct a new VectorArray object from a .fvecs or .ivecs file.
     * @param fvecs_file_path The path to the .fvecs or .ivecs file.
     */
    VectorArray(const std::string& fvecs_file_path) : _num_vecs(0), _vec_dim(0) {
        from_vecs_file(fvecs_file_path);
    }
    ~VectorArray() = default;

    // A VectorArray can be large, so we delete the copy constructor and assignment
    // to prevent accidental, expensive deep copies. This makes ownership clear.
    VectorArray(const VectorArray&) = delete;
    VectorArray& operator=(const VectorArray&) = delete;

    // We enable move semantics, which will be efficient as it just moves the underlying std::vector object.
    VectorArray(VectorArray&&) noexcept = default;
    VectorArray& operator=(VectorArray&&) noexcept = default;

    // --- Accessors ---

    __attribute__((always_inline))
    auto get(vec_id_t vid) -> vec_ele_t* {
        // Access data through the underlying vector's data pointer.
        return _storage.data() + static_cast<std::size_t>(vid) * _vec_dim;
    }

    __attribute__((always_inline))
    auto get(vec_id_t vid) const -> const vec_ele_t* {
        return _storage.data() + static_cast<std::size_t>(vid) * _vec_dim;
    }

    __attribute__((always_inline))
    auto get_all() -> vec_ele_t* {
        return _storage.data();
    }

    __attribute__((always_inline))
    auto get_all() const -> const vec_ele_t* {
        return _storage.data();
    }

    __attribute__((always_inline))
    auto get_num_vecs() const -> vec_num_t {
        return _num_vecs;
    }

    __attribute__((always_inline))
    auto get_vec_dim() const -> vec_dim_t {
        return _vec_dim;
    }

    /**
     * @brief Changes the number of vectors stored in the array, reallocating memory.
     * @param new_num_vecs The new number of vectors.
     * @note If the dimension is 0, this function will throw. You must set a dimension first.
     */
    auto resize(const vec_num_t new_num_vecs) -> void {
        // if (_vec_dim == 0 && new_num_vecs > 0) {
        //     throw std::runtime_error("Cannot resize VectorArray with zero dimension.");
        // }
        std::size_t new_total_elements = static_cast<std::size_t>(new_num_vecs) * _vec_dim;
        // Reserve memory first to avoid over-subscription in case of reallocation.
        _storage.reserve(new_total_elements);
        _storage.resize(new_total_elements);
        _num_vecs = new_num_vecs;
    }

    /**
     * @brief Loads vector data from a file, replacing existing data.
     * @param fvecs_file_path The path to the .fvecs or .ivecs file.
     * @throw std::runtime_error If the file is invalid, corrupted, or dimensions are inconsistent.
     */
    auto from_vecs_file(const std::string& fvecs_file_path) -> void {

        std::ifstream temp_file(fvecs_file_path, std::ios::binary);
        if (!temp_file.is_open()) {
            throw std::runtime_error("Error: Could not open file " + fvecs_file_path);
        }

        // --- Determine dimension from the first vector ---
        int first_dim = 0;
        temp_file.read(reinterpret_cast<char*>(&first_dim), sizeof(int));

        if (temp_file.gcount() == 0) { // File is empty
            _storage.clear();
            _num_vecs = 0;
            _vec_dim = 0;
            return;
        }

        if (first_dim <= 0) {
            throw std::runtime_error("Error: Vector dimension read from file must be positive.");
        }

        // --- Determine the number of vectors from file size ---
        temp_file.seekg(0, std::ios::end);
        std::streamoff file_size = temp_file.tellg();
        temp_file.close();

        const std::streamoff record_size = sizeof(int) + static_cast<std::streamoff>(first_dim) * sizeof(vec_ele_t);
        if (record_size <= 0) { // Should not happen with positive dimension
            throw std::runtime_error("Error: Calculated record size is invalid.");
        }

        if (file_size % record_size != 0) {
            throw std::runtime_error("Error: File size indicates a malformed or incomplete file.");
        }

        vec_num_t num_vecs_in_file = static_cast<vec_num_t>(file_size / record_size);

        _num_vecs = num_vecs_in_file;
        _vec_dim = static_cast<vec_dim_t>(first_dim);

        _storage.reserve(static_cast<std::size_t>(_num_vecs) * _vec_dim);
        _storage.resize(static_cast<std::size_t>(_num_vecs) * _vec_dim);

        // --- Parallel read directly into the allocated memory ---
        bool error_flag = false;
        std::string error_message;

        #pragma omp parallel
        {

        std::ifstream input_file(std::string(fvecs_file_path), std::ios::binary);
        if (!input_file.is_open()) {
            #pragma omp critical
            {
                if (!error_flag) {
                    error_flag = true;
                    error_message = "Error: Could not open file " + std::string(fvecs_file_path);
                }
            }
        } else {
            #pragma omp for schedule(static)
            for (vec_num_t i = 0; i < _num_vecs; ++i) {
                if (error_flag) continue;

                std::streamoff offset = static_cast<std::streamoff>(i) * record_size;
                input_file.seekg(offset, std::ios::beg);

                int file_vec_dim = 0;
                input_file.read(reinterpret_cast<char*>(&file_vec_dim), sizeof(int));

                if (!input_file || file_vec_dim != _vec_dim) {
                    #pragma omp critical
                    {
                        if (!error_flag) {
                            error_flag = true;
                            if (!input_file) {
                                error_message = "Error: Corrupted file at vector index " + std::to_string(i);
                            } else {
                                error_message = "Error: Inconsistent vector dimension. Expected " + std::to_string(_vec_dim)
                                                + ", but file has " + std::to_string(file_vec_dim)
                                                + " at vector index " + std::to_string(i);
                            }
                        }
                    }
                    continue;
                }

                vec_ele_t* vec_start = _storage.data() + static_cast<std::size_t>(i) * _vec_dim;
                std::streamsize bytes_to_read = static_cast<std::streamsize>(_vec_dim) * sizeof(vec_ele_t);
                input_file.read(reinterpret_cast<char*>(vec_start), bytes_to_read);

                if (!input_file) {
                    #pragma omp critical
                    {
                        if (!error_flag) {
                            error_flag = true;
                            error_message = "Error: Truncated file. Failed to read vector data at index " + std::to_string(i);
                        }
                    }
                }
            }
        }

        } // End of parallel region.

        if (error_flag) {
            // If an error occurred, reset the object to a clean state before throwing.
            _storage.clear();
            _num_vecs = 0;
            _vec_dim = 0;
            throw std::runtime_error(error_message);
        }
    }

private:

    avx512_container_t<vec_ele_t> _storage;
    vec_num_t _num_vecs;
    vec_dim_t _vec_dim;

};

}   // namespace cpu
}   // namespace artea