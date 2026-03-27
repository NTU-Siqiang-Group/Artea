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

// Copyright 2026 Weitang Ye
// Correctness test for Bruteforce Router using Google Test.

#include <iostream>
#include <vector>
#include <memory>
#include <filesystem>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    int num_samples;
    bool verbose;
} g_config;

// Singleton DataProvider to load dataset once
class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }
        ARTEA_INFO(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset_ = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dist_func_ = std::make_unique<dist_func_t>(dataset_->get_base_vecs().get_vec_dim());
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
};

class BruteforceCorrectnessTest : public ::testing::Test {};

TEST_F(BruteforceCorrectnessTest, VerifyRecallAccuracy) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt_vecs = dataset.get_gt_vecs();

    // 1. Initialize Artea Bruteforce Router with topk=1
    const uint32_t topk = 1;
    bruteforce_router_t router(base_vecs, dist_func, topk);

    // 2. Execute Batch Query
    // Returns knn_results_t flat array of num_queries * topk result entries
    knn_results_t predictions = router.batch_query(query_vecs);

    // 3. Verify against Ground Truth using calculate_recall_at_k with k=1
    recall_estimator_t estimator;
    auto recall = estimator.calculate_recall_at_k(
        predictions, gt_vecs, topk, query_vecs.get_num_vecs()
    );

    ARTEA_INFO(fmt::format("Artea Recall@1: {:.4f}", recall));

    // Bruteforce should theoretically be 100% (or extremely close due to float precision)
    EXPECT_GE(recall, 0.99f) << "Bruteforce router recall is lower than 0.99!";
}

TEST(BruteforceRouterTest, BatchTopKQuery) {
    auto& data = DataProvider::instance();
    auto& dataset = data.get_dataset();
    auto& dist_func = data.get_dist_func();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt_vecs = dataset.get_gt_vecs();

    // Test with different k values
    std::vector<uint32_t> k_values = {1, 5, 10, 20};

    for (uint32_t k : k_values) {
        ARTEA_INFO(fmt::format("Testing batch top-{} query", k));

        // Create router with specific topk value
        bruteforce_router_t router(base_vecs, dist_func, k);
        router.initialize();

        // Determine number of queries to test
        const uint32_t num_queries = std::min(
            static_cast<uint32_t>(query_vecs.get_num_vecs()),
            static_cast<uint32_t>(g_config.num_samples)
        );

        // Create subset of query vectors if needed
        auto query_subset = query_vecs.extract_subset(0, num_queries);
        auto gt_subset = gt_vecs.extract_subset(0, num_queries);

        // Use batch_query - returns knn_results_t flat array of num_queries * k entries
        auto batch_results = router.batch_query(query_subset);

        // Verify dimensions
        EXPECT_EQ(batch_results.size(), num_queries * k)
            << "Batch results should have num_queries * k entries";

        // Calculate Recall@K
        recall_estimator_t estimator;
        auto recall = estimator.calculate_recall_at_k(batch_results, gt_subset, k, num_queries);

        // Bruteforce should achieve perfect recall
        EXPECT_GE(recall, 0.99)
            << fmt::format("Bruteforce router batch Recall@{} is too low: {:.4f}", k, recall);

        ARTEA_INFO(fmt::format("Batch top-{} query test passed:", k));
        ARTEA_INFO(fmt::format("   -> Recall@{}: {:.2f}%", k, recall * 100.0));
    }

    ARTEA_INFO("Batch top-k query test passed");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    argparse::ArgumentParser program("test_bruteforce_router");

    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("-s", "--samples").default_value(100).scan<'i', int>().help("Number of samples (queries) to test [Ignored for full batch query]");
    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.num_samples = program.get<int>("--samples");
    g_config.verbose = program.get<bool>("--verbose");

    // Initialize data before running tests
    DataProvider::instance().init();

    return RUN_ALL_TESTS();
}