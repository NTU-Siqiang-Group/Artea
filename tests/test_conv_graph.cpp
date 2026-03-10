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

#include <iostream>
#include <vector>
#include <memory>
#include <filesystem>
#include <chrono>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;
using namespace artea::cpu::default_context;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    uint32_t max_nbr_size;
    uint32_t reserved_nbr_size;
    uint32_t extracted_nbr_size;
    float scale_coeffs;
    float shifted_coeffs;
    uint32_t num_outer_iters;
    uint32_t num_inner_iters;
    uint32_t topk;
    uint32_t candidate_queue_size;
    bool verbose;
} g_config;

struct TestResults {
    double build_time_s = 0.0;
    double conversion_time_ms = 0.0;
    double query_time_ms = 0.0;
    double avg_query_time_us = 0.0;
    double throughput_qps = 0.0;
    float recall = 0.0f;
    uint32_t num_vertices = 0;
    uint32_t num_queries = 0;
} g_test_results;

class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }
        logger.info(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset_ = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dist_func_ = std::make_unique<dist_func_t>(dataset_->get_base_vecs().get_vec_dim());

        const auto& base_vecs = dataset_->get_base_vecs();
        if (g_config.verbose) {
            logger.info(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
            logger.info(fmt::format("Max nbr size: {}", g_config.max_nbr_size));
            logger.info(fmt::format("Reserved nbr size: {}", g_config.reserved_nbr_size));
            logger.info(fmt::format("Extracted nbr size: {}", g_config.extracted_nbr_size));
            logger.info(fmt::format("Scale coeffs: {}", g_config.scale_coeffs));
            logger.info(fmt::format("Shifted coeffs: {}", g_config.shifted_coeffs));
            logger.info(fmt::format("Num outer iters: {}", g_config.num_outer_iters));
            logger.info(fmt::format("Num inner iters: {}", g_config.num_inner_iters));
            logger.info(fmt::format("Top-k: {}", g_config.topk));
            logger.info(fmt::format("Candidate queue size: {}", g_config.candidate_queue_size));
        }

        // Build convergent graph
        logger.info("Building convergent graph...");
        auto start_time = std::chrono::high_resolution_clock::now();

        conv_graph_factory_t factory;
        flat_graph_ = std::make_unique<flat_graph_t>(factory.construct_graph(
            base_vecs,
            g_config.max_nbr_size,
            g_config.reserved_nbr_size,
            g_config.scale_coeffs,
            g_config.shifted_coeffs,
            g_config.num_outer_iters,
            g_config.num_inner_iters
        ));

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        g_test_results.build_time_s = duration.count() / 1000000.0;
        g_test_results.num_vertices = flat_graph_->get_num_vertices();

        logger.info(fmt::format("Graph built with {} vertices", g_test_results.num_vertices));
        logger.info(fmt::format("Build time: {:.2f} s", g_test_results.build_time_s));

        // Convert to flat search graph
        logger.info("Converting to flat search graph...");
        start_time = std::chrono::high_resolution_clock::now();

        flat_search_graph_ = std::make_unique<flat_search_graph_t>(
            flat_search_graph_factory_t::from_flat_graph(*flat_graph_, g_config.extracted_nbr_size)
        );

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        g_test_results.conversion_time_ms = duration.count() / 1000.0;

        logger.info(fmt::format("Conversion time: {:.2f} ms", g_test_results.conversion_time_ms));

        // Get ground truth from dataset
        const auto& query_vecs = dataset_->get_query_vecs();
        g_test_results.num_queries = query_vecs.get_num_vecs();

        logger.info(fmt::format("Loaded ground truth for {} queries", g_test_results.num_queries));
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }
    flat_search_graph_t& get_flat_search_graph() { return *flat_search_graph_; }
    const idlist_array_t& get_groundtruth() { return dataset_->get_gt_vecs(); }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<flat_graph_t> flat_graph_;
    std::unique_ptr<flat_search_graph_t> flat_search_graph_;
};

class ConvGraphTest : public ::testing::Test {};

TEST_F(ConvGraphTest, QueryRecall) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();
    auto& flat_search_graph = provider.get_flat_search_graph();
    const auto& groundtruth = provider.get_groundtruth();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();

    logger.info("Testing query recall...");

    // Create router
    monolayer_graph_router_t router(
        base_vecs,
        dist_func,
        flat_search_graph,
        g_config.topk,
        g_config.candidate_queue_size
    );
    router.initialize();

    // Query all vectors
    logger.info(fmt::format("Querying {} vectors...", query_vecs.get_num_vecs()));
    auto start_time = std::chrono::high_resolution_clock::now();

    idlist_array_t results = router.batch_query(query_vecs);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    g_test_results.query_time_ms = duration.count() / 1000.0;
    g_test_results.avg_query_time_us = static_cast<double>(duration.count()) / query_vecs.get_num_vecs();
    g_test_results.throughput_qps = query_vecs.get_num_vecs() * 1000000.0 / duration.count();

    logger.info(fmt::format("Query time: {:.2f} ms", g_test_results.query_time_ms));
    logger.info(fmt::format("Avg query time: {:.2f} us", g_test_results.avg_query_time_us));
    logger.info(fmt::format("Throughput: {:.2f} QPS", g_test_results.throughput_qps));

    // Compute recall
    recall_estimator_t recall_estimator(dist_func);
    auto recall_metrics = recall_estimator.calculate_recall_at_k(
        results,
        groundtruth,
        query_vecs,
        base_vecs
    );
    g_test_results.recall = recall_metrics.soft_recall;

    logger.info(fmt::format("Recall@{}: {:.4f} (soft: {:.4f})",
        g_config.topk, recall_metrics.strict_recall, recall_metrics.soft_recall));

    // Expect reasonable recall (at least 50%)
    EXPECT_GE(g_test_results.recall, 0.5f) << "Recall should be at least 50%";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_conv_graph");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--max-nbr-size").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--reserved-nbr-size").default_value(96u).scan<'u', uint32_t>();
    program.add_argument("--extracted-nbr-size").default_value(32u).scan<'u', uint32_t>();
    program.add_argument("--scale-coeffs").default_value(1.0f).scan<'g', float>();
    program.add_argument("--shifted-coeffs").default_value(0.0f).scan<'g', float>();
    program.add_argument("--num-outer-iters").default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--num-inner-iters").default_value(14u).scan<'u', uint32_t>();
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-size").default_value(100u).scan<'u', uint32_t>();
    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.max_nbr_size = program.get<uint32_t>("--max-nbr-size");
    g_config.reserved_nbr_size = program.get<uint32_t>("--reserved-nbr-size");
    g_config.extracted_nbr_size = program.get<uint32_t>("--extracted-nbr-size");
    g_config.scale_coeffs = program.get<float>("--scale-coeffs");
    g_config.shifted_coeffs = program.get<float>("--shifted-coeffs");
    g_config.num_outer_iters = program.get<uint32_t>("--num-outer-iters");
    g_config.num_inner_iters = program.get<uint32_t>("--num-inner-iters");
    g_config.topk = program.get<uint32_t>("--topk");
    g_config.candidate_queue_size = program.get<uint32_t>("--candidate-queue-size");
    g_config.verbose = program.get<bool>("--verbose");

    // Print test configuration
    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Config path: " << g_config.config_path << std::endl;
    std::cout << "Max nbr size: " << g_config.max_nbr_size << std::endl;
    std::cout << "Reserved nbr size: " << g_config.reserved_nbr_size << std::endl;
    std::cout << "Extracted nbr size: " << g_config.extracted_nbr_size << std::endl;
    std::cout << "Scale coeffs: " << g_config.scale_coeffs << std::endl;
    std::cout << "Shifted coeffs: " << g_config.shifted_coeffs << std::endl;
    std::cout << "Num outer iters: " << g_config.num_outer_iters << std::endl;
    std::cout << "Num inner iters: " << g_config.num_inner_iters << std::endl;
    std::cout << "Top-k: " << g_config.topk << std::endl;
    std::cout << "Candidate queue size: " << g_config.candidate_queue_size << std::endl;
    std::cout << "Verbose: " << (g_config.verbose ? "true" : "false") << std::endl;
    std::cout << "==========================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    // Print summary table
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                        TEST RESULTS SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "\n--- Graph Construction ---" << std::endl;
    std::cout << fmt::format("  Build Time:             {:.2f} s", g_test_results.build_time_s) << std::endl;
    std::cout << fmt::format("  Num Vertices:           {}", g_test_results.num_vertices) << std::endl;
    std::cout << fmt::format("  Conversion Time:        {:.2f} ms", g_test_results.conversion_time_ms) << std::endl;
    std::cout << "\n--- Query Performance ---" << std::endl;
    std::cout << fmt::format("  Num Queries:            {}", g_test_results.num_queries) << std::endl;
    std::cout << fmt::format("  Total Query Time:       {:.2f} ms", g_test_results.query_time_ms) << std::endl;
    std::cout << fmt::format("  Avg Query Time:         {:.2f} us", g_test_results.avg_query_time_us) << std::endl;
    std::cout << fmt::format("  Throughput:             {:.2f} QPS", g_test_results.throughput_qps) << std::endl;
    std::cout << fmt::format("  Recall@{}:              {:.4f}", g_config.topk, g_test_results.recall) << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return result;
}
