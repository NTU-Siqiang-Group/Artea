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
 * @FilePath: /Artea/unit_tests/test_artea_graph_search.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Search tests for HierarchicalGraphRouter (search_mode & construct_mode)
 *               on an Artea hierarchical graph. Reports dataset info, recall, and throughput.
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
    uint32_t topk                  = 20;
    uint32_t queue_size            = 90;
    uint32_t bl_extracted_nbr_size = 64;
    uint32_t ul_extracted_nbr_size = 32;
    bool verbose                   = false;
} g_config;

struct SuiteResults {
    std::string dataset_name;
    uint32_t    num_base    = 0;
    uint32_t    num_queries = 0;
    uint32_t    vec_dim     = 0;
    float       search_batch_recall    = 0.f;
    double      search_batch_qps       = 0.0;
    float       search_parallel_recall = 0.f;
    double      search_parallel_qps    = 0.0;
    float       construct_batch_recall = 0.f;
    double      construct_batch_qps    = 0.0;
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

        // Build hierarchical graph
        ARTEA_INFO("Building Artea hierarchical graph...");
        layer_config_t bottom_cfg(96, 144);
        layer_config_t upper_cfg(96, 144);
        artea_graph::pruning_config_t bottom_pruning(1.10f, 0.10f);
        artea_graph::pruning_config_t upper_pruning(1.10f, 0.10f);
        artea_graph::propagate_config_t propagate_cfg(5, 12, 0.34f);

        ARTEA_INFO("Probing min_radius...");
        distance_prober_t prober(*dist_func_);
        auto probe = prober.probe(base_vecs, 0.001f, 0.95f, 0.05f);
        greedy_vertices_builder_config_t vb_cfg(probe.radius, 1.44f, 0.999f, 0.95f, 0.2f, 2048);

        hgraph_ = std::make_unique<artea_graph::index_t>(
            artea_graph::factory_t::construct_graph(
                base_vecs, bottom_cfg, upper_cfg, bottom_pruning, upper_pruning, propagate_cfg, vb_cfg
            )
        );

        // Convert to hierarchical search graph
        hsearch_graph_ = std::make_unique<hierarchical_search_graph_t>(
            search_graph_converter_t::from_hierarchical_graph(
                *hgraph_, g_config.bl_extracted_nbr_size, g_config.ul_extracted_nbr_size)
        );

        ARTEA_INFO("DataProvider ready.");
    }

    vector_dataset_t&            get_dataset()        { return *dataset_; }
    dist_func_t&                 get_dist_func()       { return *dist_func_; }
    artea_graph::index_t&        get_hgraph()          { return *hgraph_; }
    hierarchical_search_graph_t& get_hsearch_graph()  { return *hsearch_graph_; }
    const idlist_array_t&        get_gt()              { return dataset_->get_gt_vecs(); }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t>            dataset_;
    std::unique_ptr<dist_func_t>                 dist_func_;
    std::unique_ptr<artea_graph::index_t>        hgraph_;
    std::unique_ptr<hierarchical_search_graph_t> hsearch_graph_;
};

// ============================================================
// Test fixture
// ============================================================

class ArteaGraphSearchTest : public ::testing::Test {};

// ============================================================
// search_mode: BatchQuery
// ============================================================

TEST_F(ArteaGraphSearchTest, SearchModeBatchQuery) {
    auto& p = DataProvider::instance();
    const auto& base_vecs  = p.get_dataset().get_base_vecs();
    const auto& query_vecs = p.get_dataset().get_query_vecs();

    hierarchical_graph_router_t<graph_mode_t::search_mode> router(
        base_vecs, p.get_dist_func(), p.get_hsearch_graph(),
        g_config.topk, g_config.queue_size
    );
    router.initialize();

    auto t0 = std::chrono::high_resolution_clock::now();
    knn_results_t results = router.batch_query(query_vecs);
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    ASSERT_EQ(results.size(), static_cast<size_t>(g_results.num_queries * g_config.topk));

    recall_estimator_t re;
    g_results.search_batch_recall = re.calculate_recall_at_k(results, p.get_gt(), g_config.topk, g_results.num_queries);
    g_results.search_batch_qps    = g_results.num_queries * 1e6 / us;

    EXPECT_GT(g_results.search_batch_recall, 0.0f);
}

// ============================================================
// search_mode: ParallelSingleQuery
// ============================================================

TEST_F(ArteaGraphSearchTest, SearchModeParallelSingleQuery) {
    auto& p = DataProvider::instance();
    const auto& base_vecs  = p.get_dataset().get_base_vecs();
    const auto& query_vecs = p.get_dataset().get_query_vecs();

    hierarchical_graph_router_t<graph_mode_t::search_mode> router(
        base_vecs, p.get_dist_func(), p.get_hsearch_graph(),
        g_config.topk, g_config.queue_size
    );
    router.initialize();

    std::atomic<int> error_count{0};
    knn_results_t all_results(g_results.num_queries * g_config.topk);

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
                    if (!e.is_invalid() && e.get_id() >= g_results.num_base)
                        error_count.fetch_add(1, std::memory_order_relaxed);
                }
                std::copy(res.begin(), res.end(), all_results.begin() + i * g_config.topk);
            }
        }
    );
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    EXPECT_EQ(error_count.load(), 0) << "Parallel single queries produced invalid results";

    recall_estimator_t re;
    g_results.search_parallel_recall = re.calculate_recall_at_k(all_results, p.get_gt(), g_config.topk, g_results.num_queries);
    g_results.search_parallel_qps    = g_results.num_queries * 1e6 / us;
}

// ============================================================
// construct_mode: BatchQuery
// ============================================================

TEST_F(ArteaGraphSearchTest, ConstructModeBatchQuery) {
    auto& p = DataProvider::instance();
    const auto& base_vecs  = p.get_dataset().get_base_vecs();
    const auto& query_vecs = p.get_dataset().get_query_vecs();

    hierarchical_graph_router_t<graph_mode_t::construct_mode> router(
        base_vecs, p.get_dist_func(),
        g_config.topk, g_config.queue_size
    );
    router.initialize();

    auto t0 = std::chrono::high_resolution_clock::now();
    knn_results_t results = router.batch_query(query_vecs, p.get_hgraph());
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    ASSERT_EQ(results.size(), static_cast<size_t>(g_results.num_queries * g_config.topk));

    recall_estimator_t re;
    g_results.construct_batch_recall = re.calculate_recall_at_k(results, p.get_gt(), g_config.topk, g_results.num_queries);
    g_results.construct_batch_qps    = g_results.num_queries * 1e6 / us;

    EXPECT_GT(g_results.construct_batch_recall, 0.0f);
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_artea_graph_search");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-size").default_value(90u).scan<'u', uint32_t>();
    program.add_argument("--bl-extracted-nbr-size").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--ul-extracted-nbr-size").default_value(32u).scan<'u', uint32_t>();
    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try { program.parse_args(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n" << program; return 1; }

    g_config.config_path           = program.get<std::string>("--config");
    g_config.dataset_name          = program.get<std::string>("--dataset");
    g_config.topk                  = program.get<uint32_t>("--topk");
    g_config.queue_size            = program.get<uint32_t>("--candidate-queue-size");
    g_config.bl_extracted_nbr_size = program.get<uint32_t>("--bl-extracted-nbr-size");
    g_config.ul_extracted_nbr_size = program.get<uint32_t>("--ul-extracted-nbr-size");
    g_config.verbose               = program.get<bool>("--verbose");

    DataProvider::instance().init();

    int ret = RUN_ALL_TESTS();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  ARTEA GRAPH SEARCH TEST SUMMARY\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << fmt::format("  Dataset:          {}\n", g_results.dataset_name);
    std::cout << fmt::format("  Base vectors:     {}\n", g_results.num_base);
    std::cout << fmt::format("  Query vectors:    {}\n", g_results.num_queries);
    std::cout << fmt::format("  Dimension:        {}\n", g_results.vec_dim);
    std::cout << fmt::format("  Top-k:                    {}\n", g_config.topk);
    std::cout << fmt::format("  Candidate queue size:     {}\n", g_config.queue_size);
    std::cout << fmt::format("  BL extracted nbr size:    {}\n", g_config.bl_extracted_nbr_size);
    std::cout << fmt::format("  UL extracted nbr size:    {}\n", g_config.ul_extracted_nbr_size);
    std::cout << "\n";
    std::cout << fmt::format("  {:40s}  {:>10s}  {:>12s}\n", "Test", "Recall@k", "QPS");
    std::cout << std::string(70, '-') << "\n";
    std::cout << fmt::format("  {:40s}  {:>10.4f}  {:>12.1f}\n",
        "search_mode  / BatchQuery",          g_results.search_batch_recall,    g_results.search_batch_qps);
    std::cout << fmt::format("  {:40s}  {:>10.4f}  {:>12.1f}\n",
        "search_mode  / ParallelSingleQuery", g_results.search_parallel_recall, g_results.search_parallel_qps);
    std::cout << fmt::format("  {:40s}  {:>10.4f}  {:>12.1f}\n",
        "construct_mode / BatchQuery",        g_results.construct_batch_recall,  g_results.construct_batch_qps);
    std::cout << std::string(70, '=') << "\n";

    return ret;
}
