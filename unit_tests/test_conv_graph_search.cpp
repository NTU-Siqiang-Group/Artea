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
 * @FilePath: /Artea/unit_tests/test_conv_graph_search.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Search tests for DescentGraphRouter (compact_mode & dynamic_mode)
 *               on a convergent graph. Reports dataset info, recall, and throughput.
 *               Covers: SearchModeBatchQuery, SearchModeParallelSingleQuery,
 *                       ConstructModeBatchQuery.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <filesystem>
#include <chrono>
#include <atomic>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <gtest/gtest.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

// ============================================================
// Config & results
// ============================================================

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    uint32_t topk               = 20;
    uint32_t queue_size         = 80;
    uint32_t extracted_nbr_size = 16;
    uint32_t warmup_runs        = 3;
    uint32_t test_runs          = 5;
    bool verbose                = false;
} g_config;

struct SuiteResults {
    std::string dataset_name;
    uint32_t    num_base    = 0;
    uint32_t    num_queries = 0;
    uint32_t    vec_dim     = 0;
    float       search_batch_recall   = 0.f;
    double      search_batch_qps      = 0.0;
    float       search_parallel_recall = 0.f;
    double      search_parallel_qps   = 0.0;
    float       construct_batch_recall = 0.f;
    double      construct_batch_qps   = 0.0;
} g_results;

// ============================================================
// DataProvider: loads dataset, builds graph, computes GT once
// ============================================================

class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }
        ARTEA_INFO(fmt::format("Loading dataset '{}' from {}", g_config.dataset_name, g_config.config_path));
        dataset_   = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dist_func_ = std::make_unique<dist_func_t>(dataset_->get_base_vecs().get_vec_dim());

        const auto& base_vecs = dataset_->get_base_vecs();
        g_results.dataset_name = g_config.dataset_name;
        g_results.num_base     = base_vecs.get_num_vecs();
        g_results.num_queries  = dataset_->get_query_vecs().get_num_vecs();
        g_results.vec_dim      = base_vecs.get_vec_dim();

        // Build convergent graph
        ARTEA_INFO("Building convergent graph...");
        layer_config_t layer_cfg(16, 24);
        conv_graph::pruning_config_t pruning_cfg(1.0f, 0.0f);
        conv_graph::propagate_config_t propagate_cfg(4, 14, 0.6f);
        descent_graph_ = std::make_unique<conv_graph::index_t>(
            conv_graph::factory_t::construct_graph(base_vecs, layer_cfg, pruning_cfg, propagate_cfg)
        );

        // Convert to search graph
        compact_descent_graph_ = std::make_unique<compact_descent_graph_t>(
            descent_graph_compactor_t::from_descent_graph(*descent_graph_, g_config.extracted_nbr_size)
        );

        ARTEA_INFO("DataProvider ready.");
    }

    vector_dataset_t&    get_dataset()          { return *dataset_; }
    dist_func_t&         get_dist_func()         { return *dist_func_; }
    conv_graph::index_t&        get_descent_graph()         { return *descent_graph_; }
    compact_descent_graph_t& get_compact_descent_graph() { return *compact_descent_graph_; }
    const idlist_array_t& get_gt()              { return dataset_->get_gt_vecs(); }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t>    dataset_;
    std::unique_ptr<dist_func_t>         dist_func_;
    std::unique_ptr<conv_graph::index_t>        descent_graph_;
    std::unique_ptr<compact_descent_graph_t> compact_descent_graph_;
};

// ============================================================
// Test fixture
// ============================================================

class ConvGraphSearchTest : public ::testing::Test {};

// ============================================================
// compact_mode: BatchQuery
// ============================================================

TEST_F(ConvGraphSearchTest, SearchModeBatchQuery) {
    auto& p = DataProvider::instance();
    const auto& base_vecs  = p.get_dataset().get_base_vecs();
    const auto& query_vecs = p.get_dataset().get_query_vecs();

    descent_graph_router_t<graph_mode_t::compact_mode> router(
        base_vecs, p.get_dist_func(), p.get_compact_descent_graph(),
        g_config.topk, g_config.queue_size
    );
    router.initialize();

    // Warmup runs
    for (uint32_t w = 0; w < g_config.warmup_runs; ++w) {
        [[maybe_unused]] auto _ = router.batch_query(query_vecs);
    }

    // Test runs
    double total_us = 0.0;
    float total_recall = 0.0f;
    knn_results_t last_results;
    recall_estimator_t re;
    for (uint32_t r = 0; r < g_config.test_runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        last_results = router.batch_query(query_vecs);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        total_recall += re.calculate_recall_at_k(last_results, p.get_gt(), g_config.topk, g_results.num_queries);
    }
    double us = total_us / g_config.test_runs;

    ASSERT_EQ(last_results.size(), static_cast<size_t>(g_results.num_queries * g_config.topk));

    g_results.search_batch_recall = total_recall / g_config.test_runs;
    g_results.search_batch_qps    = g_results.num_queries * 1e6 / us;

    EXPECT_GT(g_results.search_batch_recall, 0.0f);
}

// ============================================================
// compact_mode: ParallelSingleQuery
// ============================================================

TEST_F(ConvGraphSearchTest, SearchModeParallelSingleQuery) {
    auto& p = DataProvider::instance();
    const auto& base_vecs  = p.get_dataset().get_base_vecs();
    const auto& query_vecs = p.get_dataset().get_query_vecs();

    descent_graph_router_t<graph_mode_t::compact_mode> router(
        base_vecs, p.get_dist_func(), p.get_compact_descent_graph(),
        g_config.topk, g_config.queue_size
    );
    router.initialize();

    // Warmup runs
    for (uint32_t w = 0; w < g_config.warmup_runs; ++w) {
        knn_results_t warmup_results(g_results.num_queries * g_config.topk);
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, g_results.num_queries),
            [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t i = r.begin(); i != r.end(); ++i) {
                    knn_results_t res = router.query(query_vecs.get(i));
                    std::copy(res.begin(), res.end(), warmup_results.begin() + i * g_config.topk);
                }
            }
        );
    }

    // Test runs
    double total_us = 0.0;
    std::atomic<int> error_count{0};
    knn_results_t all_results(g_results.num_queries * g_config.topk);

    for (uint32_t run = 0; run < g_config.test_runs; ++run) {
        error_count.store(0, std::memory_order_relaxed);

        auto t0 = std::chrono::high_resolution_clock::now();
        tbb::parallel_for(
            tbb::blocked_range<uint32_t>(0, g_results.num_queries),
            [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t i = r.begin(); i != r.end(); ++i) {
                    knn_results_t res = router.query(query_vecs.get(i));
                    if (res.size() != g_config.topk) {
                        error_count.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    for (const auto& e : res) {
                        if (!e.is_invalid() && e.get_base_id() >= g_results.num_base)
                            error_count.fetch_add(1, std::memory_order_relaxed);
                    }
                    std::copy(res.begin(), res.end(), all_results.begin() + i * g_config.topk);
                }
            }
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    }
    double us = total_us / g_config.test_runs;

    EXPECT_EQ(error_count.load(), 0) << "Parallel single queries produced invalid results";

    recall_estimator_t re;
    g_results.search_parallel_recall = re.calculate_recall_at_k(all_results, p.get_gt(), g_config.topk, g_results.num_queries);
    g_results.search_parallel_qps    = g_results.num_queries * 1e6 / us;
}

// ============================================================
// dynamic_mode: BatchQuery
// ============================================================

TEST_F(ConvGraphSearchTest, ConstructModeBatchQuery) {
    auto& p = DataProvider::instance();
    const auto& base_vecs  = p.get_dataset().get_base_vecs();
    const auto& query_vecs = p.get_dataset().get_query_vecs();

    descent_graph_router_t<graph_mode_t::dynamic_mode> router(
        base_vecs, p.get_dist_func(),
        g_config.topk, g_config.queue_size
    );
    router.initialize();

    // Warmup runs
    for (uint32_t w = 0; w < g_config.warmup_runs; ++w) {
        [[maybe_unused]] auto _ = router.batch_query(query_vecs, p.get_descent_graph());
    }

    // Test runs
    double total_us = 0.0;
    float total_recall = 0.0f;
    knn_results_t last_results;
    recall_estimator_t re;
    for (uint32_t r = 0; r < g_config.test_runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        last_results = router.batch_query(query_vecs, p.get_descent_graph());
        auto t1 = std::chrono::high_resolution_clock::now();
        total_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        total_recall += re.calculate_recall_at_k(last_results, p.get_gt(), g_config.topk, g_results.num_queries);
    }
    double us = total_us / g_config.test_runs;

    ASSERT_EQ(last_results.size(), static_cast<size_t>(g_results.num_queries * g_config.topk));

    g_results.construct_batch_recall = total_recall / g_config.test_runs;
    g_results.construct_batch_qps    = g_results.num_queries * 1e6 / us;

    EXPECT_GT(g_results.construct_batch_recall, 0.0f);
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_conv_graph_search");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-size").default_value(80u).scan<'u', uint32_t>();
    program.add_argument("--extracted-nbr-size").default_value(16u).scan<'u', uint32_t>();
    program.add_argument("--warmup-runs").default_value(5u).scan<'u', uint32_t>();
    program.add_argument("--test-runs").default_value(10u).scan<'u', uint32_t>();
    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try { program.parse_args(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n" << program; return 1; }

    g_config.config_path        = program.get<std::string>("--config");
    g_config.dataset_name       = program.get<std::string>("--dataset");
    g_config.topk               = program.get<uint32_t>("--topk");
    g_config.queue_size         = program.get<uint32_t>("--candidate-queue-size");
    g_config.extracted_nbr_size = program.get<uint32_t>("--extracted-nbr-size");
    g_config.warmup_runs        = program.get<uint32_t>("--warmup-runs");
    g_config.test_runs          = program.get<uint32_t>("--test-runs");
    g_config.verbose            = program.get<bool>("--verbose");

    DataProvider::instance().init();

    int ret = RUN_ALL_TESTS();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  CONV GRAPH SEARCH TEST SUMMARY\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << fmt::format("  Dataset:          {}\n", g_results.dataset_name);
    std::cout << fmt::format("  Base vectors:     {}\n", g_results.num_base);
    std::cout << fmt::format("  Query vectors:    {}\n", g_results.num_queries);
    std::cout << fmt::format("  Dimension:        {}\n", g_results.vec_dim);
    std::cout << fmt::format("  Top-k:                 {}\n", g_config.topk);
    std::cout << fmt::format("  Candidate queue size:  {}\n", g_config.queue_size);
    std::cout << fmt::format("  Extracted nbr size:    {}\n", g_config.extracted_nbr_size);
    std::cout << fmt::format("  Warmup runs:           {}\n", g_config.warmup_runs);
    std::cout << fmt::format("  Test runs:             {}\n", g_config.test_runs);
    std::cout << "\n";
    std::cout << fmt::format("  {:40s}  {:>10s}  {:>12s}  {:>10s}\n", "Test", "Recall@k", "QPS", "Runs");
    std::cout << std::string(80, '-') << "\n";
    std::string runs_str = fmt::format("{}w+{}r", g_config.warmup_runs, g_config.test_runs);
    std::cout << fmt::format("  {:40s}  {:>10.4f}  {:>12.1f}  {:>10s}\n",
        "compact_mode  / BatchQuery",       g_results.search_batch_recall,    g_results.search_batch_qps, runs_str);
    std::cout << fmt::format("  {:40s}  {:>10.4f}  {:>12.1f}  {:>10s}\n",
        "compact_mode  / ParallelSingleQuery", g_results.search_parallel_recall, g_results.search_parallel_qps, runs_str);
    std::cout << fmt::format("  {:40s}  {:>10.4f}  {:>12.1f}  {:>10s}\n",
        "dynamic_mode / BatchQuery",     g_results.construct_batch_recall,  g_results.construct_batch_qps, runs_str);
    std::cout << std::string(70, '=') << "\n";

    return ret;
}
