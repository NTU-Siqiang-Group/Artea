/*
 * @FilePath: /Artea/include/artea/cpu/vector_array.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-10-23 17:13:43
 * @Date: 2025-10-18 16:31:57
 * @Description: 
 */

#pragma once

#include <fstream>      
#include <string>       
#include <stdexcept>
#include <omp.h>

#include <artea/types.hpp>

namespace artea {
namespace cpu {

template <
    typename vecs_num_t, 
    typename vec_ele_t,
    typename vec_id_t = vecs_num_t
>
class VectorArray {

public:
    /**
     * @brief Default constructor. Creates an empty VectorArray.
     */
    VectorArray() : _vec_data(nullptr), _num_vecs(0), _vec_dim(0) {}
    
    /**
     * @brief Construct a new Vector Pool object with a pre-defined size.
     * @param num_vecs The total number of vectors the pool will manage.
     * @param dim The dimension of each vector.
     */
    VectorArray(vecs_num_t num_vecs, vec_dim_t dim) : _num_vecs(num_vecs), _vec_dim(dim) {
        if (num_vecs > 0 && dim > 0) {
            _vec_data = new vec_ele_t[num_vecs * dim];
        } else {
            _vec_data = nullptr;
            throw std::invalid_argument("Error: VectorArray dimensions must be positive.");
        }
    }

    /**
     * @brief Construct a new Vector Pool object from a .fvecs or .ivecs file.
     * @param fvecs_file_path The path to the .fvecs or .ivecs file.
     */
    VectorArray(const std::string& fvecs_file_path) : _vec_data(nullptr), _num_vecs(0), _vec_dim(0) {
        from_vecs_file(fvecs_file_path);
    }

    /**
     * @brief Destroy the Vector Pool object.
     */
    ~VectorArray() {
        delete[] _vec_data;
    }

    __attribute__((always_inline))
    auto get(vec_id_t vid) -> vec_ele_t* {
        return _vec_data + static_cast<std::size_t>(vid) * _vec_dim;
    }

    __attribute__((always_inline))
    auto get_all() -> vec_ele_t* {
        return _vec_data;
    }

    __attribute__((always_inline))
    auto size() -> vecs_num_t {
        return _num_vecs;
    }

    __attribute__((always_inline))
    auto get_num_vecs() -> vecs_num_t {
        return _num_vecs;
    }

    __attribute__((always_inline))
    auto dim() -> vec_dim_t {
        return _vec_dim;
    }

    __attribute__((always_inline))
    auto get_vec_dim() -> vec_dim_t {
        return _vec_dim;
    }

    /**
     * @brief Determines vector count and dimension from file, allocates memory, and loads data in parallel.
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
            delete[] _vec_data;
            _vec_data = nullptr;
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
        if (record_size <= 0) {
            throw std::runtime_error("Error: Calculated record size is invalid.");
        }

        if (file_size % record_size != 0) {
            throw std::runtime_error("Error: File size indicates a malformed or incomplete file.");
        }
        
        vecs_num_t num_vecs_in_file = static_cast<vecs_num_t>(file_size / record_size);

        // --- Allocate memory ---
        delete[] _vec_data;
        _num_vecs = num_vecs_in_file;
        _vec_dim = static_cast<vec_dim_t>(first_dim);
        _vec_data = new vec_ele_t[static_cast<std::size_t>(_num_vecs) * _vec_dim];

        // --- Parallel read into the allocated memory ---
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
                for (vecs_num_t i = 0; i < _num_vecs; ++i) {
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
                    
                    vec_ele_t* vec_start = _vec_data + static_cast<std::size_t>(i) * _vec_dim;
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
            // If an error occurred, reset the pool to a clean state before throwing.
            delete[] _vec_data;
            _vec_data = nullptr;
            _num_vecs = 0;
            _vec_dim = 0;
            throw std::runtime_error(error_message);
        }
    }

private:
    vec_ele_t* _vec_data;
    vecs_num_t _num_vecs;
    vec_dim_t _vec_dim;
};

}   // namespace cpu
}   // namespace artea
