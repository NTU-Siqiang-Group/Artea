/*
 * @FilePath: /Artea/include/artea/cpu/containers/vector_array.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-01-26 16:14:08
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


    VectorArray(vec_dim_t dim) : _num_vecs(0), _vec_dim(dim) {}

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

    // ==========================================
    // Iterator Definitions
    // ==========================================

    /**
     * @brief A stride iterator that navigates the flat storage vector by vector.
     *
     * When dereferenced (*it), it returns a pointer (VecEleT*) to the start
     * of the current vector.
     */
    template <bool IsConst>
    class VecIterator {
    public:
        // Standard iterator traits for STL compatibility
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = VecEleT;
        using difference_type   = std::ptrdiff_t;
        // The 'reference' here is a pointer to the start of the vector data.
        using reference         = typename std::conditional<IsConst, const VecEleT*, VecEleT*>::type;
        using pointer           = typename std::conditional<IsConst, const VecEleT*, VecEleT*>::type;

        using internal_ptr_t    = typename std::conditional<IsConst, const VecEleT*, VecEleT*>::type;

        VecIterator() : _ptr(nullptr), _stride(0) {}
        VecIterator(internal_ptr_t ptr, vec_dim_t stride) : _ptr(ptr), _stride(stride) {}

        // Allow implicit conversion from iterator to const_iterator
        template <bool WasConst, typename = std::enable_if_t<IsConst && !WasConst>>
        VecIterator(const VecIterator<WasConst>& other) : _ptr(other._ptr), _stride(other._stride) {}

        // Dereference: Returns the pointer to the start of the current vector
        reference operator*() const { return _ptr; }

        // Random access: it[n] returns the pointer to the nth vector relative to current
        reference operator[](difference_type n) const { return _ptr + (n * _stride); }

        // --- Increment/Decrement Operations ---

        // Prefix ++
        VecIterator& operator++() {
            _ptr += _stride;
            return *this;
        }

        // Postfix ++
        VecIterator operator++(int) {
            VecIterator temp = *this;
            _ptr += _stride;
            return temp;
        }

        // Prefix --
        VecIterator& operator--() {
            _ptr -= _stride;
            return *this;
        }

        // Postfix --
        VecIterator operator--(int) {
            VecIterator temp = *this;
            _ptr -= _stride;
            return temp;
        }

        // --- Arithmetic Operations ---

        VecIterator& operator+=(difference_type n) { _ptr += n * _stride; return *this; }
        VecIterator& operator-=(difference_type n) { _ptr -= n * _stride; return *this; }

        VecIterator operator+(difference_type n) const { return VecIterator(_ptr + n * _stride, _stride); }
        VecIterator operator-(difference_type n) const { return VecIterator(_ptr - n * _stride, _stride); }

        friend VecIterator operator+(difference_type n, const VecIterator& it) { return it + n; }

        // Calculate distance (number of vectors) between iterators
        difference_type operator-(const VecIterator& other) const {
            if (_stride == 0) return 0;
            return (_ptr - other._ptr) / _stride;
        }

        // --- Comparison Operations ---

        bool operator==(const VecIterator& other) const { return _ptr == other._ptr; }
        bool operator!=(const VecIterator& other) const { return _ptr != other._ptr; }
        bool operator<(const VecIterator& other)  const { return _ptr < other._ptr; }
        bool operator>(const VecIterator& other)  const { return _ptr > other._ptr; }
        bool operator<=(const VecIterator& other) const { return _ptr <= other._ptr; }
        bool operator>=(const VecIterator& other) const { return _ptr >= other._ptr; }

    private:
        internal_ptr_t _ptr;
        vec_dim_t _stride;
        // Grant VectorArray access to private members for construction
        friend class VectorArray;
    };

    using iterator = VecIterator<false>;
    using const_iterator = VecIterator<true>;

    /**
     * @brief Returns an iterator to the first vector.
     */
    iterator begin() {
        return iterator(_storage.data(), _vec_dim);
    }

    /**
     * @brief Returns an iterator to the position past the last vector.
     */
    iterator end() {
        // Calculate the address immediately after the last element of the last vector
        return iterator(_storage.data() + static_cast<std::size_t>(_num_vecs) * _vec_dim, _vec_dim);
    }

    /**
     * @brief Returns a const iterator to the first vector.
     */
    const_iterator begin() const {
        return const_iterator(_storage.data(), _vec_dim);
    }

    /**
     * @brief Returns a const iterator to the position past the last vector.
     */
    const_iterator end() const {
        return const_iterator(_storage.data() + static_cast<std::size_t>(_num_vecs) * _vec_dim, _vec_dim);
    }

    const_iterator cbegin() const { return begin(); }
    const_iterator cend() const { return end(); }

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
    auto set(vec_id_t vid, const vec_ele_t* data) -> void {
        vec_ele_t* target = _storage.data() + static_cast<std::size_t>(vid) * _vec_dim;
        std::copy(data, data + _vec_dim, target);
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

    auto reserve(const vec_num_t new_num_vecs) -> void {
        _storage.reserve(static_cast<std::size_t>(new_num_vecs) * _vec_dim);
    }

    /**
     * @brief Appends a single vector to the end of the array.
     * @param vec_ptr Pointer to the vector data to append (must have at least _vec_dim elements).
     * @note This function efficiently appends data to the underlying storage using std::vector::insert.
     */
    auto append_vec(const vec_ele_t* vec_ptr) -> void {
        _storage.insert(_storage.end(), vec_ptr, vec_ptr + _vec_dim);
        _num_vecs++;
    }

    /**
     * @brief Creates a new VectorArray containing a subset of the current vectors.
     * @param start The starting index of the subset.
     * @param count The number of vectors to include in the subset.
     * @return A new VectorArray object containing the copied subset data.
     * @throw std::out_of_range If the requested range exceeds the current array bounds.
     */
    auto get_subset(vec_num_t start, vec_num_t count) const -> VectorArray {
        if (static_cast<std::size_t>(start) + count > _num_vecs) {
            throw std::out_of_range(fmt::format(
                "VectorArray::get_subset: Range out of bounds. Start: {}, Count: {}, Total: {}",
                start, count, _num_vecs));
        }
        // Create a new instance with the target size and same dimension
        VectorArray subset(count, _vec_dim);
        if (count > 0) {
            const vec_ele_t* src_ptr = this->get(start);
            vec_ele_t* dst_ptr = subset.get_all();
            std::size_t total_elements = static_cast<std::size_t>(count) * _vec_dim;
            // Standard copy from source to the new storage
            std::copy(src_ptr, src_ptr + total_elements, dst_ptr);
        }
        return subset;
    }

    auto get_subset(vec_num_t start, vec_num_t count, VectorArray subset) const -> void {
        if (static_cast<std::size_t>(start) + count > _num_vecs) {
            throw std::out_of_range(fmt::format(
                "VectorArray::get_subset: Range out of bounds. Start: {}, Count: {}, Total: {}",
                start, count, _num_vecs));
        }
        if (count > 0) {
            const vec_ele_t* src_ptr = this->get(start);
            vec_ele_t* dst_ptr = subset.get_all();
            if (subset.get_num_vecs() != count) {
                subset.resize(count);
            }
            std::size_t total_elements = static_cast<std::size_t>(count) * _vec_dim;
            // Standard copy from source to the new storage
            std::copy(src_ptr, src_ptr + total_elements, dst_ptr);
        }
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