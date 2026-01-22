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
 * @FilePath: /Artea/tests/test_bruteforce_router.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Test suite for Bruteforce Router using Google Test framework.
 */

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <memory> // For std::unique_ptr

#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>

using namespace artea;
using namespace artea::cpu;

// --- Type Definitions ---
using vec_num_t = uint32_t;
using vec_ele_t = float;
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using router_traits_t = RouterTraits<computer_traits_t, false>;
using vec_dim_t = typename base_traits_t::vec_dim_t;
using vector_dataset_t = typename base_traits_t::vector_dataset_t;
using dist_func_t = typename computer_traits_t::dist_func_t;
using artea_router_t = typename router_traits_t::bruteforce_router_t;
using recall_estimator_t = typename computer_traits_t::recall_estimator_t;

// Global configuration
struct TestConfig {
    std::string config_path;
    std::string dataset_name;
} g_test_config;

// --- Test Fixture ---
class BruteforceRouterTest : public ::testing::Test {
protected:
    static std::unique_ptr<vector_dataset_t> dataset;
    static std::unique_ptr<dist_func_t> dist_func;

    static void SetUpTestSuite() {
        if (!std::filesystem::exists(g_test_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_test_config.config_path);
        }

        logger.info(fmt::format("Loading Dataset: {} from {}", g_test_config.dataset_name, g_test_config.config_path));

        dataset = std::make_unique<vector_dataset_t>(g_test_config.config_path, g_test_config.dataset_name);
        auto dim = dataset->get_base_vecs().get_vec_dim();
        dist_func = std::make_unique<dist_func_t>(dim);

        logger.info(fmt::format("Dataset Loaded: {} base, {} queries, dim={}",
            dataset->get_base_vecs().get_num_vecs(),
            dataset->get_query_vecs().get_num_vecs(),
            dim));
    }

    static void TearDownTestSuite() {
        dist_func.reset();
        dataset.reset();
    }
};

// Define static members
std::unique_ptr<vector_dataset_t> BruteforceRouterTest::dataset = nullptr;
std::unique_ptr<dist_func_t> BruteforceRouterTest::dist_func = nullptr;

// --- Tests ---

/**
 * @brief Verify Artea's results against Ground Truth provided by the dataset.
 */
TEST_F(BruteforceRouterTest, RecallAccuracy) {
    ASSERT_TRUE(dataset != nullptr) << "Dataset failed to load.";

    const auto& base_vecs = dataset->get_base_vecs();
    const auto& query_vecs = dataset->get_query_vecs();
    const auto& gt_vecs = dataset->get_gt_vecs();

    // 1. Run Artea
    artea_router_t router(base_vecs, *dist_func);
    std::vector<vec_num_t> artea_labels = router.batch_query(query_vecs);

    // 2. Estimate Recall
    recall_estimator_t estimator(*dist_func);
    auto metrics = estimator.calculate_recall_at_1(
        artea_labels, gt_vecs, query_vecs, base_vecs
    );

    logger.info(fmt::format("Artea Soft Recall@1: {:.4f}", metrics.soft_recall));

    EXPECT_GE(metrics.soft_recall, 0.999f) << "Bruteforce router should have near-perfect recall.";
}

// --- Main ---
int main(int argc, char* argv[]) {
    // 1. Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // 2. Parse Custom Arguments
    argparse::ArgumentParser program("test_bruteforce_router_gtest");

    program.add_argument("-c", "--config").default_value(std::string("../datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_test_config.config_path = program.get<std::string>("--config");
    g_test_config.dataset_name = program.get<std::string>("--dataset");

    // 3. Run Tests
    return RUN_ALL_TESTS();
}