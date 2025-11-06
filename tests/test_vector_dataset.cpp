/*
 * @FilePath: /yeweitang/Artea/tests/test_vector_dataset.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-05 12:50:26
 * @Date: 2025-10-23 15:50:42
 * @Description: Test the VectorDataset class for loading time and correctness.
 */

#include <iostream>
#include <string>
#include <chrono> 
#include <stdexcept>
#include <vector>
#include <fstream>
#include <utility>
#include <random>
#include <cmath>
#include <cassert>

#include <fmt/format.h>

#include <artea/cpu/vector_dataset.hpp>
#include <artea/types.hpp> 
#include <artea/logger.hpp>

namespace {
artea::ArteaLogger logger("testVectorDataset");
}   // anonymous namespace

// --- Start of Correctness Verification Code ---

/**
 * @brief A simple, single-threaded reference implementation to read a .vecs file.
 * @tparam T The element type of the vector (e.g., float, uint32_t).
 * @param file_path Path to the .fvecs or .ivecs file.
 * @return A pair containing a std::vector with all the flattened data and the dimension of vectors.
 */
template<typename T>
std::pair<std::vector<T>, int> load_vecs_file_simple(const std::string& file_path) {
    std::ifstream input(file_path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Reference loader could not open file: " + file_path);
    }

    // Read dimension from the first vector
    int dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(int));
    if (dim <= 0) {
        throw std::runtime_error("Reference loader read an invalid dimension: " + std::to_string(dim));
    }
    input.seekg(0, std::ios::beg); // Rewind to start

    const size_t record_size = sizeof(int) + static_cast<size_t>(dim) * sizeof(T);
    
    // Get file size to determine the number of vectors
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    input.seekg(0, std::ios::beg);

    if (file_size % record_size != 0) {
        throw std::runtime_error("Reference loader: File size indicates corruption.");
    }

    size_t num_vectors = file_size / record_size;
    std::vector<T> data;
    data.reserve(num_vectors * dim);
    
    int temp_dim = 0;
    std::vector<T> buffer(dim);

    // Read all vectors one by one
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&temp_dim), sizeof(int));
        if (temp_dim != dim) {
            throw std::runtime_error("Reference loader: Inconsistent dimension found.");
        }
        input.read(reinterpret_cast<char*>(buffer.data()), dim * sizeof(T));
        data.insert(data.end(), buffer.begin(), buffer.end());
    }

    return {data, dim};
}


/**
 * @brief Runs a correctness check by comparing artea's loader against a simple reference loader.
 * @tparam T The element type of the vector.
 * @tparam ArteaVecNumT The type for number of vectors in Artea's class.
 * @param file_type_name A descriptive name for the file being checked (e.g., "Base Vectors").
 * @param file_path The path to the .vecs file.
 * @param artea_array A pointer to the VectorArray loaded by the artea library.
 */
template<typename T, typename ArteaVecNumT>
void run_correctness_check(
    const std::string& file_type_name,
    const std::string& file_path,
    artea::cpu::VectorArray<ArteaVecNumT, T>* artea_array
) {
    std::cout << "\n--- Running correctness check for " << file_type_name << " ---" << std::endl;

    // 1. Load data using the simple reference implementation
    auto [ref_data, ref_dim] = load_vecs_file_simple<T>(file_path);
    size_t ref_num_vecs = ref_data.size() / ref_dim;
    
    logger.info(fmt::format("Reference loader: {} vectors, {} dims.", ref_num_vecs, ref_dim));
    
    // 2. Compare metadata (vector count and dimension)
    auto artea_num_vecs = artea_array->get_num_vecs();
    auto artea_dim = artea_array->get_vec_dim();
    logger.info(fmt::format("Artea loader: {} vectors, {} dims.", artea_num_vecs, artea_dim));
    
    if (artea_dim == ref_dim && artea_num_vecs == ref_num_vecs) {
        logger.success("Metadata check PASSED.");
    } else {
        logger.error("Metadata check FAILED.");
    }

    // 3. Randomly sample and compare vector data
    constexpr int NUM_CHECKS = 1000;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<ArteaVecNumT> dist(0, artea_num_vecs - 1);

    for (int i = 0; i < NUM_CHECKS; ++i) {
        ArteaVecNumT vec_id = dist(rng);
        T* artea_vec = artea_array->get(vec_id);
        T* ref_vec_start = &ref_data[static_cast<size_t>(vec_id) * artea_dim];

        for (artea::vec_dim_t j = 0; j < artea_dim; ++j) {
            if constexpr (std::is_floating_point_v<T>) {
                const T tolerance = 1e-6f;
                if (std::abs(artea_vec[j] - ref_vec_start[j]) >= tolerance) {
                    logger.error(fmt::format("Float data mismatch at vec_id {} dim {}: {} != {}", vec_id, j, artea_vec[j], ref_vec_start[j]));
                }
            } else {
                if (artea_vec[j] != ref_vec_start[j]) {
                    logger.error(fmt::format("Integer data mismatch at vec_id {} dim {}: {} != {}", vec_id, j, artea_vec[j], ref_vec_start[j]));
                }
            }
        }
    }
    logger.success(fmt::format("Random vector data check ({}) PASSED.", NUM_CHECKS));
}

// --- End of Correctness Verification Code ---


int main() {
    // NOTE: Hardcoded paths are used for this test.
    const std::string config_path = "/home/yeweitang/Artea/datasets.json";
    const std::string root_dir = "/home/yeweitang/ANNDatasets/sift-1m/";
    const std::string dataset_name = "sift-1m";

    std::cout << "Starting VectorDataset test for dataset: " << dataset_name << std::endl;
    std::cout << "Using config file: " << config_path << std::endl;

    try {
        // --- Load dataset using Artea and measure time ---
        auto start_time = std::chrono::high_resolution_clock::now();

        artea::cpu::VectorDataset<
            uint32_t, // vecs_num_t
            float,    // vec_ele_t for base and query
            uint32_t  // vec_id_t for ground truth
        > dataset(config_path, dataset_name);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;

        logger.success("Successfully loaded the dataset using Artea.");
        logger.info(fmt::format("Time taken to load: {} ms", elapsed_ms.count()));

        // --- Perform correctness check (on-par comparison) ---
        run_correctness_check<float>(
            "Base Vectors", 
            root_dir + "sift_base.fvecs", 
            dataset.get_base_vecs()
        );

        run_correctness_check<float>(
            "Query Vectors", 
            root_dir + "sift_query.fvecs", 
            dataset.get_query_vecs()
        );

        run_correctness_check<uint32_t>(
            "Ground Truth Vectors", 
            root_dir + "sift_groundtruth.ivecs", 
            dataset.get_gt_vecs()
        );

    } catch (const std::exception& e) { // Catch std::exception for broader coverage
        logger.error(fmt::format("\nAn error occurred during the test: {}", e.what()));
        return 1;
    } catch (...) {
        logger.error("\nAn unknown error occurred.");
        return 1;
    }
    
    logger.success("\nAll tests completed successfully!");
    return 0;
}