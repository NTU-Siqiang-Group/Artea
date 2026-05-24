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
#include <set>
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
        dispatcher_ = std::make_unique<simd_dispatcher_t>(dataset_->get_base_vecs().get_vec_dim());
        // Populate the per-base ||p||^2 cache so the FastL2 equivalence
        // test can call BruteforceRouter::query_fast / batch_query_fast.
        dataset_->enable_fast_L2();
    }

    vector_dataset_t&   get_dataset()    { return *dataset_; }
    simd_dispatcher_t&  get_dispatcher() { return *dispatcher_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t>  dataset_;
    std::unique_ptr<simd_dispatcher_t> dispatcher_;
};

class BruteforceCorrectnessTest : public ::testing::Test {};

TEST_F(BruteforceCorrectnessTest, VerifyRecallAccuracy) {
    auto& provider   = DataProvider::instance();
    auto& dataset    = provider.get_dataset();
    auto& dispatcher = provider.get_dispatcher();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt_vecs = dataset.get_gt_vecs();

    // Subset queries to g_config.num_samples to keep the test bounded
    // (full query set on large datasets is far more than we need to
    // demonstrate ~1.0 recall on bruteforce).
    const uint32_t num_queries = std::min(
        static_cast<uint32_t>(query_vecs.get_num_vecs()),
        static_cast<uint32_t>(g_config.num_samples));
    auto query_subset = query_vecs.extract_subset(0, num_queries);
    auto gt_subset    = gt_vecs.extract_subset(0, num_queries);

    ARTEA_INFO(fmt::format(
        "VerifyRecallAccuracy: dataset={}, base={}, queries={}/{}  (subset), topk=1",
        g_config.dataset_name, base_vecs.get_num_vecs(),
        num_queries, query_vecs.get_num_vecs()));

    dispatcher.dispatch([&](const auto& dist_func) {
        using DistFunc = std::decay_t<decltype(dist_func)>;
        // 1. Initialize Artea Bruteforce Router with topk=1
        const uint32_t topk = 1;
        bruteforce_router_t<DistFunc> router(base_vecs, dist_func, topk);

        // 2. Execute Batch Query
        knn_results_t predictions = router.batch_query(query_subset);

        // 3. Verify against Ground Truth using calculate_recall_at_k with k=1
        recall_estimator_t estimator;
        auto recall = estimator.calculate_recall_at_k(
            predictions, gt_subset, topk, num_queries);

        ARTEA_INFO(fmt::format("Artea Recall@1: {:.4f}", recall));

        // Bruteforce should theoretically be 100% (or extremely close due to float precision)
        EXPECT_GE(recall, 0.99f) << "Bruteforce router recall is lower than 0.99!";
    });
}

TEST(BruteforceRouterTest, BatchTopKQuery) {
    auto& data       = DataProvider::instance();
    auto& dataset    = data.get_dataset();
    auto& dispatcher = data.get_dispatcher();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt_vecs = dataset.get_gt_vecs();

    // Test with different k values
    std::vector<uint32_t> k_values = {1, 5, 10, 20};

    dispatcher.dispatch([&](const auto& dist_func) {
        using DistFunc = std::decay_t<decltype(dist_func)>;
        for (uint32_t k : k_values) {
            ARTEA_INFO(fmt::format("Testing batch top-{} query", k));

            // Create router with specific topk value
            bruteforce_router_t<DistFunc> router(base_vecs, dist_func, k);
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
    });

    ARTEA_INFO("Batch top-k query test passed");
}

/**
 * @brief Equivalence test: bruteforce L2 (router.batch_query) and
 *        bruteforce FastL2 (router.batch_query_fast) must return the
 *        same set of top-K vids per query. FastL2 is a ranking-equivalent
 *        proxy — same argmin/argsort up to FP tie-break — so the
 *        UNORDERED vid sets must match exactly.
 */
TEST(BruteforceFastL2EquivalenceTest, TopKMatchesL2) {
    auto& provider   = DataProvider::instance();
    auto& dataset    = provider.get_dataset();
    auto& dispatcher = provider.get_dispatcher();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt_vecs = dataset.get_gt_vecs();
    const auto& base_norms = dataset.get_base_norms();

    ASSERT_FALSE(base_norms.empty())
        << "base_norms is empty — DataProvider::init() forgot enable_fast_L2()";

    const uint32_t num_queries = std::min(
        static_cast<uint32_t>(query_vecs.get_num_vecs()),
        static_cast<uint32_t>(g_config.num_samples)
    );
    auto query_subset = query_vecs.extract_subset(0, num_queries);
    auto gt_subset    = gt_vecs.extract_subset(0, num_queries);

    dispatcher.dispatch([&](const auto& dist_func) {
        using DistFunc = std::decay_t<decltype(dist_func)>;
        for (uint32_t k : {1u, 10u, 100u}) {
            if (k > base_vecs.get_num_vecs()) continue;
            ARTEA_INFO(fmt::format("FastL2 equivalence: k={}, num_queries={}", k, num_queries));

            bruteforce_router_t<DistFunc> router(base_vecs, dist_func, k);
            router.initialize();

            auto preds_l2   = router.batch_query(query_subset);
            auto preds_fast = router.batch_query_fast(query_subset, base_norms);

            ASSERT_EQ(preds_l2.size(),   num_queries * k);
            ASSERT_EQ(preds_fast.size(), num_queries * k);

            uint32_t mismatches = 0;
            for (uint32_t q = 0; q < num_queries; ++q) {
                std::set<vec_id_t> set_l2, set_fast;
                for (uint32_t r = 0; r < k; ++r) {
                    set_l2.insert(preds_l2[q * k + r].get_vid());
                    set_fast.insert(preds_fast[q * k + r].get_vid());
                }
                if (set_l2 != set_fast) {
                    ++mismatches;
                    if (mismatches <= 3) {  // Bound the dump.
                        ARTEA_WARN(fmt::format(
                            "k={} query={}: L2 set != FastL2 set", k, q));
                    }
                }
            }

            EXPECT_EQ(mismatches, 0u)
                << fmt::format(
                    "FastL2 disagrees with L2 on {} / {} queries at k={}. "
                    "Check fast_euclidean math or VectorDataset::_compute_base_norms.",
                    mismatches, num_queries, k);

            // Backstop: recall must not regress.
            recall_estimator_t est;
            auto r_l2   = est.calculate_recall_at_k(preds_l2,   gt_subset, k, num_queries);
            auto r_fast = est.calculate_recall_at_k(preds_fast, gt_subset, k, num_queries);
            EXPECT_GE(r_fast, r_l2 - 1e-6f)
                << fmt::format("Recall@{} regressed: L2={:.6f} fast={:.6f}", k, r_l2, r_fast);

            ARTEA_INFO(fmt::format(
                "  k={:>3}  recall L2={:.4f} fast={:.4f}", k, r_l2, r_fast));
        }
    });
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    argparse::ArgumentParser program("test_bruteforce_router");

    program.add_argument("-c", "--config").default_value(artea::default_dataset_config_path());
    program.add_argument("-d", "--dataset").default_value(std::string("crawl"));
    program.add_argument("-s", "--samples").default_value(200).scan<'i', int>().help("Number of queries to test (subset of query set)");
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