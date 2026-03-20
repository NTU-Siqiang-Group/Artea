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

#include <gtest/gtest.h>
#include <argparse/argparse.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <artea/cpu/vertex_generator/random_vg.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <unordered_set>
#include <algorithm>
#include <filesystem>
#include <memory>

using namespace artea;
using namespace artea::cpu;

// Global configuration
struct TestConfig {
    std::string config_path;
    std::string dataset_name;
} g_config;

class RandomVGTest : public ::testing::Test {
protected:
    static std::unique_ptr<vector_dataset_t> dataset;
    static const vector_array_t* vecs_data;
    static uint32_t num_vecs;
    static uint32_t vec_dim;

    static void SetUpTestSuite() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }

        logger.info(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);

        vecs_data = &dataset->get_base_vecs();
        num_vecs = vecs_data->get_num_vecs();
        vec_dim = vecs_data->get_vec_dim();

        logger.info(fmt::format("Dataset loaded: {} vectors, {} dimensions", num_vecs, vec_dim));
    }

    static void TearDownTestSuite() {
        dataset.reset();
    }
};

std::unique_ptr<vector_dataset_t> RandomVGTest::dataset = nullptr;
const vector_array_t* RandomVGTest::vecs_data = nullptr;
uint32_t RandomVGTest::num_vecs = 0;
uint32_t RandomVGTest::vec_dim = 0;

// Test 1: Basic functionality - returns correct number of vertices
TEST_F(RandomVGTest, ReturnsCorrectSize) {
    random_vg_t random_vg;

    uint32_t result_size = std::min(10000u, num_vecs);
    auto result = random_vg.generate(*vecs_data, result_size);

    EXPECT_EQ(result.get_num_vecs(), result_size);
    EXPECT_EQ(result.vec_ids.size(), result_size);
    EXPECT_EQ(result.vecs_data.get_num_vecs(), result_size);
}

// Test 2: Sampling without replacement - all IDs are unique
TEST_F(RandomVGTest, SamplingWithoutReplacement) {
    random_vg_t random_vg;

    uint32_t result_size = std::min(50000u, num_vecs / 2);
    auto result = random_vg.generate(*vecs_data, result_size);

    // Check all IDs are unique
    std::unordered_set<vec_id_t> unique_ids(result.vec_ids.begin(), result.vec_ids.end());
    EXPECT_EQ(unique_ids.size(), result_size);
}


// Test 4: Vector data matches original vectors
TEST_F(RandomVGTest, VectorDataMatchesOriginal) {
    random_vg_t random_vg;

    uint32_t result_size = std::min(15000u, num_vecs / 4);
    auto result = random_vg.generate(*vecs_data, result_size);

    // Verify each vector matches the original
    for (uint32_t i = 0; i < result_size; ++i) {
        vec_id_t original_id = result.vec_ids[i];
        const float* original_vec = vecs_data->get(original_id);
        const float* result_vec = result.vecs_data.get(i);

        // Check all dimensions match
        for (uint32_t d = 0; d < vec_dim; ++d) {
            EXPECT_FLOAT_EQ(result_vec[d], original_vec[d])
                << "Mismatch at result index " << i
                << " (original ID " << original_id << "), dimension " << d;
        }
    }
}

// Test 5: All IDs are within valid range
TEST_F(RandomVGTest, IDsWithinValidRange) {
    random_vg_t random_vg;

    uint32_t result_size = std::min(30000u, num_vecs / 2);
    auto result = random_vg.generate(*vecs_data, result_size);

    for (const auto& id : result.vec_ids) {
        EXPECT_GE(id, 0);
        EXPECT_LT(id, num_vecs);
    }
}

// Test 6: Edge case - sample size equals total size
TEST_F(RandomVGTest, SampleSizeEqualsTotal) {
    random_vg_t random_vg;

    auto result = random_vg.generate(*vecs_data, num_vecs);

    EXPECT_EQ(result.get_num_vecs(), num_vecs);

    // All IDs should be present
    std::unordered_set<vec_id_t> unique_ids(result.vec_ids.begin(), result.vec_ids.end());
    EXPECT_EQ(unique_ids.size(), num_vecs);
}

// Test 7: Edge case - sample size exceeds total size (should clamp)
TEST_F(RandomVGTest, SampleSizeExceedsTotal) {
    random_vg_t random_vg;

    uint32_t oversized_request = num_vecs + 50000;
    auto result = random_vg.generate(*vecs_data, oversized_request);

    // Should return at most num_vecs
    EXPECT_LE(result.get_num_vecs(), num_vecs);
}

// Test 8: Edge case - zero sample size
TEST_F(RandomVGTest, ZeroSampleSize) {
    random_vg_t random_vg;

    auto result = random_vg.generate(*vecs_data, 0);

    EXPECT_EQ(result.get_num_vecs(), 0);
    EXPECT_EQ(result.vec_ids.size(), 0);
}

// Test 9: Randomness - multiple runs produce different results
TEST_F(RandomVGTest, ProducesRandomResults) {
    random_vg_t random_vg1;
    random_vg_t random_vg2;

    uint32_t result_size = std::min(10000u, num_vecs / 5);
    auto result1 = random_vg1.generate(*vecs_data, result_size);
    auto result2 = random_vg2.generate(*vecs_data, result_size);

    // Results should be different (with very high probability)
    bool are_different = false;
    for (size_t i = 0; i < result_size; ++i) {
        if (result1.vec_ids[i] != result2.vec_ids[i]) {
            are_different = true;
            break;
        }
    }

    EXPECT_TRUE(are_different) << "Two independent random samples should be different";
}

// Test 10: Small sample size
TEST_F(RandomVGTest, SmallSampleSize) {
    random_vg_t random_vg;

    uint32_t result_size = 100;
    auto result = random_vg.generate(*vecs_data, result_size);

    EXPECT_EQ(result.get_num_vecs(), result_size);

    // Verify uniqueness
    std::unordered_set<vec_id_t> unique_ids(result.vec_ids.begin(), result.vec_ids.end());
    EXPECT_EQ(unique_ids.size(), result_size);
}

// Test 11: Large sample size (90% of total)
TEST_F(RandomVGTest, LargeSampleSize) {
    random_vg_t random_vg;

    uint32_t result_size = static_cast<uint32_t>(num_vecs * 0.9);
    auto result = random_vg.generate(*vecs_data, result_size);

    EXPECT_EQ(result.get_num_vecs(), result_size);

    // Verify uniqueness
    std::unordered_set<vec_id_t> unique_ids(result.vec_ids.begin(), result.vec_ids.end());
    EXPECT_EQ(unique_ids.size(), result_size);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_random_vg");
    program.add_argument("-c", "--config")
        .help("Path to the dataset configuration JSON file")
        .default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset")
        .help("Name of the dataset to use")
        .default_value(std::string("sift-1m"));

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");

    return RUN_ALL_TESTS();
}
