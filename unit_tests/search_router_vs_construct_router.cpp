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
 * @FilePath: /Artea/unit_tests/search_router_vs_construct_router.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Compares SingleLayerRouter on a dynamic vs compact RefiningGraph
 *               of a ConvGraph. Measures construct-mode query time, search-graph
 *               conversion time, and search-mode query time.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
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
    uint32_t max_nbr_size       = 64;
    float scale_coeffs          = 1.0f;
    float shifted_coeffs        = 0.0f;
    uint32_t num_build_loops    = 4;
    uint32_t num_triu_iters     = 14;
    float prefill_ratio         = 0.6f;
    uint32_t num_routing_loops  = 1;
    uint32_t extracted_nbr_size = 64;
    uint32_t topk               = 20;
    uint32_t queue_size         = 80;
    bool verbose                = false;
} g_config;

struct SuiteResults {
    std::string dataset_name;
    uint32_t    num_base        = 0;
    uint32_t    num_queries     = 0;
    uint32_t    vec_dim         = 0;
    double      build_time_s    = 0.0;
    double      conversion_time_ms = 0.0;
    float       construct_recall   = 0.f;
    double      construct_qps      = 0.0;
    float       search_recall      = 0.f;
    double      search_qps         = 0.0;
} g_results;

// ============================================================
// DataProvider: builds graph and converts once
// ============================================================

class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }
        ARTEA_INFO(fmt::format("Loading dataset '{}' from {}",
            g_config.dataset_name, g_config.config_path));
        dataset_    = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dispatcher_ = std::make_unique<simd_dispatcher_t>(dataset_->get_base_vecs().get_vec_dim());

        const auto& base_vecs = dataset_->get_base_vecs();
        g_results.dataset_name = g_config.dataset_name;
        g_results.num_base     = base_vecs.get_num_vecs();
        g_results.num_queries  = dataset_->get_query_vecs().get_num_vecs();
        g_results.vec_dim      = base_vecs.get_vec_dim();

        // Build convergent graph
        layer_config_t layer_cfg(g_config.max_nbr_size);
        conv_graph::pruning_config_t pruning_cfg(g_config.scale_coeffs, g_config.shifted_coeffs);
        conv_graph::propagate_config_t propagate_cfg(
            g_config.num_build_loops, g_config.num_triu_iters, g_config.prefill_ratio,
            g_config.num_routing_loops);

        ARTEA_INFO("Building convergent graph...");
        auto t0 = std::chrono::high_resolution_clock::now();
        graph_index_ = std::make_unique<conv_graph::index_t>(dispatcher_->dispatch(
            [&](const auto& dist_func) {
                return std::move(
                    conv_graph::factory_t::construct_graph(
                        base_vecs, layer_cfg, pruning_cfg, propagate_cfg, dist_func
                    ).graph
                );
            }
        ));
        auto t1 = std::chrono::high_resolution_clock::now();
        g_results.build_time_s =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        ARTEA_INFO(fmt::format("Graph built: {} vertices in {:.2f} s",
            graph_index_->get_num_vertices(), g_results.build_time_s));

        // Convert to flat search graph
        ARTEA_INFO("Converting to flat search graph...");
        auto tc0 = std::chrono::high_resolution_clock::now();
        compact_refining_graph_ = std::make_unique<compact::refining_graph_t>(
            refining_graph_compactor_t::compact_graph(*graph_index_, g_config.extracted_nbr_size)
        );
        auto tc1 = std::chrono::high_resolution_clock::now();
        g_results.conversion_time_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(tc1 - tc0).count() / 1e3;
        ARTEA_INFO(fmt::format("Conversion done in {:.2f} ms", g_results.conversion_time_ms));

        ARTEA_INFO("DataProvider ready.");
    }

    vector_dataset_t&    get_dataset()           { return *dataset_; }
    simd_dispatcher_t&   get_dispatcher()         { return *dispatcher_; }
    conv_graph::index_t&        get_graph_index()          { return *graph_index_; }
    compact::refining_graph_t& get_compact_refining_graph()  { return *compact_refining_graph_; }
    const idlist_array_t& get_gt()               { return dataset_->get_gt_vecs(); }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t>    dataset_;
    std::unique_ptr<simd_dispatcher_t>   dispatcher_;
    std::unique_ptr<conv_graph::index_t>        graph_index_;
    std::unique_ptr<compact::refining_graph_t> compact_refining_graph_;
};

// ============================================================
// Test fixture
// ============================================================

class RouterComparisonTest : public ::testing::Test {};

// ============================================================
// Test 1: dynamic_mode router on flat graph
// ============================================================

TEST_F(RouterComparisonTest, ConstructModeRouter) {
    auto& p = DataProvider::instance();
    const auto& base_vecs  = p.get_dataset().get_base_vecs();
    const auto& query_vecs = p.get_dataset().get_query_vecs();

    ARTEA_WITH_DIM(p.get_dispatcher(), DistFunc, dist_func) {
        single_layer_router_t<DistFunc> router(
            base_vecs, dist_func,
            g_config.topk, g_config.queue_size
        );
        router.initialize();

        auto t0 = std::chrono::high_resolution_clock::now();
        knn_results_t results = router.batch_query(
            query_vecs, p.get_graph_index().get_refining_graph());
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        ASSERT_EQ(results.size(), static_cast<size_t>(g_results.num_queries * g_config.topk));

        recall_estimator_t re;
        g_results.construct_recall = re.calculate_recall_at_k(
            results, p.get_gt(), g_config.topk, g_results.num_queries);
        g_results.construct_qps = g_results.num_queries * 1e6 / us;

        EXPECT_GT(g_results.construct_recall, 0.0f);
    } ARTEA_END_DIM(p.get_dispatcher());
}

// ============================================================
// Test 2: compact_mode router on flat search graph
// ============================================================

TEST_F(RouterComparisonTest, SearchModeRouter) {
    auto& p = DataProvider::instance();
    const auto& base_vecs  = p.get_dataset().get_base_vecs();
    const auto& query_vecs = p.get_dataset().get_query_vecs();

    ARTEA_WITH_DIM(p.get_dispatcher(), DistFunc, dist_func) {
        single_layer_router_t<DistFunc> router(
            base_vecs, dist_func,
            g_config.topk, g_config.queue_size
        );
        router.initialize();

        auto t0 = std::chrono::high_resolution_clock::now();
        knn_results_t results = router.batch_query(
            query_vecs, p.get_compact_refining_graph());
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        ASSERT_EQ(results.size(), static_cast<size_t>(g_results.num_queries * g_config.topk));

        recall_estimator_t re;
        g_results.search_recall = re.calculate_recall_at_k(
            results, p.get_gt(), g_config.topk, g_results.num_queries);
        g_results.search_qps = g_results.num_queries * 1e6 / us;

        EXPECT_GT(g_results.search_recall, 0.0f);
    } ARTEA_END_DIM(p.get_dispatcher());
}

// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("search_router_vs_construct_router");
    program.add_argument("-c", "--config").default_value(artea::default_dataset_config_path());
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--max-nbr-size").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--scale-coeffs").default_value(1.0f).scan<'g', float>();
    program.add_argument("--shifted-coeffs").default_value(0.0f).scan<'g', float>();
    program.add_argument("--num-build-loops").default_value(5u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters").default_value(12u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio").default_value(0.6f).scan<'g', float>();
    program.add_argument("--num-routing-loops").default_value(1u).scan<'u', uint32_t>();
    program.add_argument("--extracted-nbr-size").scan<'u', uint32_t>()
        .help("Extracted neighbor size for search graph (defaults to max-nbr-size)");
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-size").default_value(80u).scan<'u', uint32_t>();
    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try { program.parse_args(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n" << program; return 1; }

    g_config.config_path     = program.get<std::string>("--config");
    g_config.dataset_name    = program.get<std::string>("--dataset");
    g_config.max_nbr_size    = program.get<uint32_t>("--max-nbr-size");
    g_config.scale_coeffs    = program.get<float>("--scale-coeffs");
    g_config.shifted_coeffs  = program.get<float>("--shifted-coeffs");
    g_config.num_build_loops    = program.get<uint32_t>("--num-build-loops");
    g_config.num_triu_iters     = program.get<uint32_t>("--num-triu-iters");
    g_config.prefill_ratio      = program.get<float>("--prefill-ratio");
    g_config.num_routing_loops  = program.get<uint32_t>("--num-routing-loops");
    g_config.extracted_nbr_size = program.is_used("--extracted-nbr-size")
        ? program.get<uint32_t>("--extracted-nbr-size")
        : g_config.max_nbr_size;
    g_config.topk            = program.get<uint32_t>("--topk");
    g_config.queue_size      = program.get<uint32_t>("--candidate-queue-size");
    g_config.verbose         = program.get<bool>("--verbose");

    std::cout << "\n=== Test Configuration ===\n";
    std::cout << fmt::format("  Dataset:              {}\n", g_config.dataset_name);
    std::cout << fmt::format("  Max nbr size:         {}\n", g_config.max_nbr_size);
    std::cout << fmt::format("  Scale coeffs:         {}\n", g_config.scale_coeffs);
    std::cout << fmt::format("  Shifted coeffs:       {}\n", g_config.shifted_coeffs);
    std::cout << fmt::format("  Build loops:          {}\n", g_config.num_build_loops);
    std::cout << fmt::format("  Triangle updater iters: {}\n", g_config.num_triu_iters);
    std::cout << fmt::format("  Prefill ratio:        {}\n", g_config.prefill_ratio);
    std::cout << fmt::format("  Routing loops:        {}\n", g_config.num_routing_loops);
    std::cout << fmt::format("  Extracted nbr size:   {}\n", g_config.extracted_nbr_size);
    std::cout << fmt::format("  Top-k:                {}\n", g_config.topk);
    std::cout << fmt::format("  Candidate queue size: {}\n", g_config.queue_size);
    std::cout << "==========================\n\n";

    DataProvider::instance().init();

    int ret = RUN_ALL_TESTS();

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  CONSTRUCT-MODE vs SEARCH-MODE ROUTER COMPARISON\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << fmt::format("  Dataset:          {}\n", g_results.dataset_name);
    std::cout << fmt::format("  Base vectors:     {}\n", g_results.num_base);
    std::cout << fmt::format("  Query vectors:    {}\n", g_results.num_queries);
    std::cout << fmt::format("  Dimension:        {}\n", g_results.vec_dim);
    std::cout << fmt::format("  Top-k:            {}\n", g_config.topk);
    std::cout << fmt::format("  Queue size:       {}\n", g_config.queue_size);
    std::cout << "\n";
    std::cout << fmt::format("  Graph build time:       {:.2f} s\n", g_results.build_time_s);
    std::cout << fmt::format("  Search graph conv time: {:.2f} ms\n", g_results.conversion_time_ms);
    std::cout << "\n";
    std::cout << fmt::format("  {:45s}  {:>10s}  {:>12s}\n", "Mode", "Recall@k", "QPS");
    std::cout << std::string(70, '-') << "\n";
    std::cout << fmt::format("  {:45s}  {:>10.4f}  {:>12.1f}\n",
        "dynamic_mode (RefiningGraph, no conversion)",
        g_results.construct_recall, g_results.construct_qps);
    std::cout << fmt::format("  {:45s}  {:>10.4f}  {:>12.1f}\n",
        "compact_mode (CompactRefiningGraph, +conv time)",
        g_results.search_recall, g_results.search_qps);
    std::cout << std::string(70, '=') << "\n";

    return ret;
}
