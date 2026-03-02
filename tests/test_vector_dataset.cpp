// Copyright 2026 Weitang Ye
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
 * @FilePath: /Artea/tests/test_vector_dataset.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: GoogleTest suite for VectorDataset correctness verification.
 */

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <random>
#include <filesystem>
#include <memory>
#include <type_traits>

#include <gtest/gtest.h>
#include <argparse/argparse.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>

using namespace artea;
using namespace artea::cpu;

// --- Type Definitions ---
using vec_num_t = uint32_t;
using vec_ele_t = float;
// Define Traits
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t, false>;
using vector_dataset_t = typename base_traits_t::vector_dataset_t;
using vector_array_t = typename base_traits_t::vector_array_t;

// --- Global Configuration ---
struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    uint32_t num_check_samples; // Number of random vectors to verify
} g_config;

// --- Helper Functions (Reference Implementation) ---

/**
 * @brief A simple, single-threaded reference implementation to read a .vecs file.
 *        Used as Ground Truth to verify Artea's parallel loader.
 *
 * @tparam T The element type of the vector (e.g., float, uint32_t).
 * @param file_path Path to the .fvecs or .ivecs file.
 * @return A pair containing:
 *         1. std::vector<T> with flattened data.
 *         2. int representing the dimension.
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
        throw std::runtime_error("Reference loader: File size indicates corruption or mismatch.");
    }

    size_t num_vectors = file_size / record_size;
    std::vector<T> data;
    // Pre-allocate memory
    data.reserve(num_vectors * dim);

    int temp_dim = 0;
    std::vector<T> buffer(dim);

    // Read all vectors one by one
    for (size_t i = 0; i < num_vectors; ++i) {
        input.read(reinterpret_cast<char*>(&temp_dim), sizeof(int));
        if (temp_dim != dim) {
            throw std::runtime_error("Reference loader: Inconsistent dimension found in file.");
        }
        input.read(reinterpret_cast<char*>(buffer.data()), dim * sizeof(T));
        data.insert(data.end(), buffer.begin(), buffer.end());
    }

    return {data, dim};
}

// --- Test Fixture ---

class VectorDatasetTest : public ::testing::Test {
protected:
    static std::unique_ptr<vector_dataset_t> dataset;
    static nlohmann::json json_config;

    /**
     * @brief Sets up the test suite by loading the dataset once.
     *        This mimics the logic in VectorDataset to resolve file paths for verification.
     */
    static void SetUpTestSuite() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }

        // 1. Load Dataset via Artea
        logger.info(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);

        // 2. Parse JSON manually to get raw file paths for the reference loader
        std::ifstream config_file(g_config.config_path);
        json_config = nlohmann::json::parse(config_file);
    }

    static void TearDownTestSuite() {
        dataset.reset();
    }

    /**
     * @brief Helper to reconstruct the full path for a specific file type (base/query/gt)
     */
    std::string get_file_path(const std::string& type_key) {
        auto dataset_cfg = json_config["datasets"][g_config.dataset_name];
        std::filesystem::path root_dir = json_config["root_dir"];
        std::filesystem::path dataset_dir = root_dir / dataset_cfg["dataset_dir"];
        std::filesystem::path file_path = dataset_dir / dataset_cfg[type_key];
        return file_path.string();
    }

    /**
     * @brief Generic verification logic.
     *
     * @tparam T Data type (float for base/query, usually int/uint32_t for GT).
     * @tparam ArteaArrayT The type of the Artea VectorArray.
     * @param artea_array Reference to the loaded Artea array.
     * @param json_key The key in the JSON config ("base_path", "query_path", etc.).
     * @param label A label for logging purposes.
     */
    template <typename T, typename ArteaArrayT>
    void verify_data(ArteaArrayT& artea_array, const std::string& json_key, const std::string& label) {
        std::string full_path = get_file_path(json_key);
        logger.info(fmt::format("Verifying {} against file: {}", label, full_path));

        // 1. Load Reference Data
        auto [ref_data, ref_dim] = load_vecs_file_simple<T>(full_path);
        size_t ref_num_vecs = ref_data.size() / ref_dim;

        // 2. Metadata Check
        EXPECT_EQ(artea_array.get_vec_dim(), ref_dim)
            << "Dimension mismatch for " << label;
        EXPECT_EQ(artea_array.get_num_vecs(), ref_num_vecs)
            << "Vector count mismatch for " << label;

        if (HasFatalFailure()) return; // Abort if metadata is wrong

        // 3. Random Sampling Check
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, ref_num_vecs - 1);

        logger.info(fmt::format("Checking {} random samples for {}...", g_config.num_check_samples, label));

        for (uint32_t i = 0; i < g_config.num_check_samples; ++i) {
            size_t vec_id = dist(rng);

            // Pointer to Artea data
            // Note: VectorArray::get returns a pointer to the start of the vector
            const T* artea_vec = reinterpret_cast<const T*>(artea_array.get(static_cast<vec_num_t>(vec_id)));

            // Pointer to Reference data
            const T* ref_vec = &ref_data[vec_id * ref_dim];

            for (int d = 0; d < ref_dim; ++d) {
                if constexpr (std::is_floating_point_v<T>) {
                    ASSERT_FLOAT_EQ(artea_vec[d], ref_vec[d])
                        << fmt::format("Mismatch in {} at ID {}, Dim {}", label, vec_id, d);
                } else {
                    ASSERT_EQ(artea_vec[d], ref_vec[d])
                        << fmt::format("Mismatch in {} at ID {}, Dim {}", label, vec_id, d);
                }
            }
        }
        logger.success(fmt::format("{} passed verification.", label));
    }
};

// Define static members
std::unique_ptr<vector_dataset_t> VectorDatasetTest::dataset = nullptr;
nlohmann::json VectorDatasetTest::json_config;

// --- Tests ---

TEST_F(VectorDatasetTest, VerifyBaseVectors) {
    ASSERT_TRUE(dataset != nullptr) << "Dataset failed to initialize.";
    // Base vectors are usually float
    verify_data<float>(dataset->get_base_vecs(), "base_path", "Base Vectors");
}

TEST_F(VectorDatasetTest, VerifyQueryVectors) {
    ASSERT_TRUE(dataset != nullptr) << "Dataset failed to initialize.";
    // Query vectors are usually float
    verify_data<float>(dataset->get_query_vecs(), "query_path", "Query Vectors");
}

TEST_F(VectorDatasetTest, VerifyGroundTruthVectors) {
    ASSERT_TRUE(dataset != nullptr) << "Dataset failed to initialize.";
    // Ground Truth vectors are typically integer (IDs)
    // Adjust type based on your dataset format (sift-1m GT is ivecs -> int/uint32_t)
    verify_data<uint32_t>(dataset->get_gt_vecs(), "gt_path", "Ground Truth Vectors");
}

// --- Main ---

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize Argument Parser
    argparse::ArgumentParser program("test_vector_dataset_gtest");

    program.add_argument("-c", "--config")
        .help("Path to the datasets.json configuration file")
        .default_value(std::string("./configs/datasets.json"));

    program.add_argument("-d", "--dataset")
        .help("Name of the dataset to verify (must exist in json)")
        .default_value(std::string("sift-1m"));

    program.add_argument("-s", "--samples")
        .help("Number of random samples to check for correctness")
        .scan<'u', uint32_t>()
        .default_value(uint32_t{1000});

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Populate Global Config
    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.num_check_samples = program.get<uint32_t>("--samples");

    logger.info("==========================================================");
    logger.info("      Starting VectorDataset Correctness Suite");
    logger.info("==========================================================");

    return RUN_ALL_TESTS();
}