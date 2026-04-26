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
#include <unordered_map>

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
using vec_dim_t = uint32_t;
using vec_ele_t = float;
// Define Traits
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t>;
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
        ARTEA_INFO(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
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
        ARTEA_INFO(fmt::format("Verifying {} against file: {}", label, full_path));

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

        ARTEA_INFO(fmt::format("Checking {} random samples for {}...", g_config.num_check_samples, label));

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
        ARTEA_SUCCESS(fmt::format("{} passed verification.", label));
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

TEST_F(VectorDatasetTest, VerifyGetSubset) {
    ASSERT_TRUE(dataset != nullptr) << "Dataset failed to initialize.";

    auto& base_vecs = dataset->get_base_vecs();
    vec_num_t total_vecs = base_vecs.get_num_vecs();
    vec_dim_t dim = base_vecs.get_vec_dim();

    // Test with random subset of vector IDs
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<vec_num_t> dist(0, total_vecs - 1);

    // Create a list of random vector IDs
    std::vector<vec_num_t> vec_ids;
    vec_num_t subset_size = std::min(static_cast<vec_num_t>(1000), total_vecs);
    for (vec_num_t i = 0; i < subset_size; ++i) {
        vec_ids.push_back(dist(rng));
    }

    ARTEA_INFO(fmt::format("Testing extract_subset with {} random vectors...", subset_size));

    // Get subset using the parallel implementation
    auto subset = base_vecs.extract_subset(vec_ids);

    // Verify metadata
    EXPECT_EQ(subset.get_num_vecs(), subset_size) << "Subset size mismatch";
    EXPECT_EQ(subset.get_vec_dim(), dim) << "Subset dimension mismatch";

    // Verify data correctness
    for (vec_num_t i = 0; i < subset_size; ++i) {
        const float* original_vec = base_vecs.get(vec_ids[i]);
        const float* subset_vec = subset.get(i);

        for (vec_dim_t d = 0; d < dim; ++d) {
            ASSERT_FLOAT_EQ(subset_vec[d], original_vec[d])
                << fmt::format("Mismatch in subset at index {}, dim {}", i, d);
        }
    }

    ARTEA_SUCCESS("extract_subset passed verification.");
}

TEST_F(VectorDatasetTest, VerifyShuffleInPlace) {
    ASSERT_TRUE(dataset != nullptr) << "Dataset failed to initialize.";

    auto& base_vecs = dataset->get_base_vecs();
    auto& gt_vecs = dataset->get_gt_vecs();

    vec_num_t total_vecs = base_vecs.get_num_vecs();
    vec_dim_t dim = base_vecs.get_vec_dim();
    vec_num_t num_gt = gt_vecs.get_num_vecs();
    vec_dim_t gt_dim = gt_vecs.get_vec_dim();

    ARTEA_INFO(fmt::format("Testing shuffle_in_place with {} base vectors...", total_vecs));

    // Sample a subset for verification to avoid O(n²) complexity
    constexpr vec_num_t sample_size = 1000;
    vec_num_t num_samples = std::min(sample_size, total_vecs);

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<vec_num_t> dist(0, total_vecs - 1);

    // Sample random indices
    std::vector<vec_num_t> sample_indices;
    sample_indices.reserve(num_samples);
    for (vec_num_t i = 0; i < num_samples; ++i) {
        sample_indices.push_back(dist(rng));
    }

    // Store original data for sampled vectors only
    std::unordered_map<vec_num_t, std::vector<float>> original_base_vecs;
    for (vec_num_t idx : sample_indices) {
        const float* vec = base_vecs.get(idx);
        original_base_vecs[idx].assign(vec, vec + dim);
    }

    // Store original ground truth
    std::vector<std::vector<uint32_t>> original_gt_vecs(num_gt);
    for (vec_num_t i = 0; i < num_gt; ++i) {
        const uint32_t* gt_row = gt_vecs.get(i);
        original_gt_vecs[i].assign(gt_row, gt_row + gt_dim);
    }

    // Perform shuffle with fixed seed for reproducibility
    uint32_t seed = 42;
    dataset->shuffle_in_place(seed);

    // Verify metadata unchanged
    EXPECT_EQ(base_vecs.get_num_vecs(), total_vecs) << "Base vectors count changed after shuffle";
    EXPECT_EQ(base_vecs.get_vec_dim(), dim) << "Base vectors dimension changed after shuffle";
    EXPECT_EQ(gt_vecs.get_num_vecs(), num_gt) << "Ground truth count changed after shuffle";
    EXPECT_EQ(gt_vecs.get_vec_dim(), gt_dim) << "Ground truth dimension changed after shuffle";

    // Verify sampled vectors still exist (just reordered)
    ARTEA_INFO(fmt::format("Verifying {} sampled vectors...", num_samples));
    for (const auto& [old_idx, old_vec] : original_base_vecs) {
        bool match_found = false;

        // Search for this vector in the shuffled dataset
        for (vec_num_t new_id = 0; new_id < total_vecs; ++new_id) {
            const float* shuffled_vec = base_vecs.get(new_id);

            bool matches = true;
            for (vec_dim_t d = 0; d < dim; ++d) {
                if (std::abs(shuffled_vec[d] - old_vec[d]) > 1e-6f) {
                    matches = false;
                    break;
                }
            }

            if (matches) {
                match_found = true;
                break;
            }
        }

        ASSERT_TRUE(match_found) << fmt::format("Original vector at index {} not found in shuffled data", old_idx);
    }

    // Verify ground truth IDs are correctly updated (sample a subset)
    ARTEA_INFO(fmt::format("Verifying ground truth for {} sampled queries...", std::min(100u, num_gt)));
    vec_num_t num_gt_samples = std::min(100u, num_gt);

    for (vec_num_t query_id = 0; query_id < num_gt_samples; ++query_id) {
        const uint32_t* new_gt_row = gt_vecs.get(query_id);
        const auto& old_gt_row = original_gt_vecs[query_id];

        // Only check first 10 ground truth entries per query
        vec_dim_t num_gt_checks = std::min(10u, gt_dim);
        for (vec_dim_t k = 0; k < num_gt_checks; ++k) {
            uint32_t old_base_id = old_gt_row[k];
            uint32_t new_base_id = new_gt_row[k];

            if (old_base_id >= total_vecs || new_base_id >= total_vecs) continue;

            // Verify the vectors are the same
            const float* old_vec_data = original_base_vecs.count(old_base_id)
                ? original_base_vecs[old_base_id].data()
                : nullptr;

            if (!old_vec_data) continue; // Skip if not in our sample

            const float* new_vec_data = base_vecs.get(new_base_id);

            bool vectors_match = true;
            for (vec_dim_t d = 0; d < dim; ++d) {
                if (std::abs(old_vec_data[d] - new_vec_data[d]) > 1e-6f) {
                    vectors_match = false;
                    break;
                }
            }

            ASSERT_TRUE(vectors_match)
                << fmt::format("Ground truth mismatch: query {}, k={}, old_id={}, new_id={}",
                              query_id, k, old_base_id, new_base_id);
        }
    }

    ARTEA_SUCCESS("shuffle_in_place passed verification.");
}

TEST_F(VectorDatasetTest, VerifyShuffleSeedReproducibility) {
    ASSERT_TRUE(dataset != nullptr) << "Dataset failed to initialize.";

    // Load two independent datasets
    vector_dataset_t dataset1(g_config.config_path, g_config.dataset_name);
    vector_dataset_t dataset2(g_config.config_path, g_config.dataset_name);

    auto& base_vecs1 = dataset1.get_base_vecs();
    auto& base_vecs2 = dataset2.get_base_vecs();
    auto& gt_vecs1 = dataset1.get_gt_vecs();
    auto& gt_vecs2 = dataset2.get_gt_vecs();

    vec_num_t total_vecs = base_vecs1.get_num_vecs();
    vec_dim_t dim = base_vecs1.get_vec_dim();
    vec_num_t num_gt = gt_vecs1.get_num_vecs();
    vec_dim_t gt_dim = gt_vecs1.get_vec_dim();

    ARTEA_INFO(fmt::format("Testing shuffle seed reproducibility with {} base vectors...", total_vecs));

    // Shuffle both datasets with the same seed
    uint32_t seed = 12345;
    dataset1.shuffle_in_place(seed);
    dataset2.shuffle_in_place(seed);

    // Verify metadata unchanged
    EXPECT_EQ(base_vecs1.get_num_vecs(), total_vecs) << "Dataset1 base vectors count changed";
    EXPECT_EQ(base_vecs2.get_num_vecs(), total_vecs) << "Dataset2 base vectors count changed";
    EXPECT_EQ(base_vecs1.get_vec_dim(), dim) << "Dataset1 dimension changed";
    EXPECT_EQ(base_vecs2.get_vec_dim(), dim) << "Dataset2 dimension changed";

    // Verify all base vectors match exactly
    ARTEA_INFO("Verifying all base vectors match...");
    for (vec_num_t i = 0; i < total_vecs; ++i) {
        const float* vec1 = base_vecs1.get(i);
        const float* vec2 = base_vecs2.get(i);

        for (vec_dim_t d = 0; d < dim; ++d) {
            ASSERT_FLOAT_EQ(vec1[d], vec2[d])
                << fmt::format("Base vector mismatch at index {}, dim {}", i, d);
        }
    }

    // Verify all ground truth vectors match exactly
    ARTEA_INFO("Verifying all ground truth vectors match...");
    for (vec_num_t i = 0; i < num_gt; ++i) {
        const uint32_t* gt1 = gt_vecs1.get(i);
        const uint32_t* gt2 = gt_vecs2.get(i);

        for (vec_dim_t k = 0; k < gt_dim; ++k) {
            ASSERT_EQ(gt1[k], gt2[k])
                << fmt::format("Ground truth mismatch at query {}, k={}", i, k);
        }
    }

    ARTEA_SUCCESS("Shuffle seed reproducibility test passed.");
}

// --- Main ---

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize Argument Parser
    argparse::ArgumentParser program("test_vector_dataset_gtest");

    program.add_argument("-c", "--config")
        .help("Path to the datasets.json configuration file")
        .default_value(artea::default_dataset_config_path());

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

    ARTEA_INFO("==========================================================");
    ARTEA_INFO("      Starting VectorDataset Correctness Suite");
    ARTEA_INFO("==========================================================");

    return RUN_ALL_TESTS();
}