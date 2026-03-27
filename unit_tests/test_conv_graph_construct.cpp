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
#include <sstream>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    layer_config_t layer_config{16, 32};
    conv_graph::pruning_config_t pruning_config{1.0f, 0.0f};
    conv_graph::propagate_config_t propagate_config{4, 14};
    uint32_t extracted_nbr_size;
    uint32_t topk;
    uint32_t queue_start;
    uint32_t queue_end;
    uint32_t queue_step;
    bool verbose;
} g_config;

struct QueryResult {
    uint32_t candidate_queue_size;
    double query_time_ms;
    double avg_query_time_us;
    double throughput_qps;
    float recall;
};

struct TestResults {
    double build_time_s = 0.0;
    double conversion_time_ms = 0.0;
    uint32_t num_vertices = 0;
    uint32_t num_queries = 0;
    std::vector<QueryResult> query_results;
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
        ARTEA_INFO(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset_ = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dist_func_ = std::make_unique<dist_func_t>(dataset_->get_base_vecs().get_vec_dim());

        const auto& base_vecs = dataset_->get_base_vecs();
        if (g_config.verbose) {
            ARTEA_INFO(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
            ARTEA_INFO(fmt::format("Max nbr size: {}", g_config.layer_config.max_nbr_size()));
            ARTEA_INFO(fmt::format("Reserved nbr size: {}", g_config.layer_config.reserved_nbr_size()));
            ARTEA_INFO(fmt::format("Extracted nbr size: {}", g_config.extracted_nbr_size));
            ARTEA_INFO(fmt::format("Scale coeffs: {}", g_config.pruning_config.scale_coeffs()));
            ARTEA_INFO(fmt::format("Shifted coeffs: {}", g_config.pruning_config.shifted_coeffs()));
            ARTEA_INFO(fmt::format("Build loops: {}", g_config.propagate_config.num_build_loops()));
            ARTEA_INFO(fmt::format("Triangle updater iterations: {}", g_config.propagate_config.num_triu_iters()));
            ARTEA_INFO(fmt::format("Top-k: {}", g_config.topk));
            ARTEA_INFO(fmt::format("Candidate queue size range: {} to {} step {}",
                g_config.queue_start, g_config.queue_end, g_config.queue_step));
        }

        // Build convergent graph
        ARTEA_INFO("Building convergent graph...");
        auto start_time = std::chrono::high_resolution_clock::now();

        flat_graph_ = std::make_unique<flat_graph_t>(conv_graph_factory_t::construct_graph(
            base_vecs,
            g_config.layer_config,
            g_config.pruning_config,
            g_config.propagate_config
        ));

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        g_test_results.build_time_s = duration.count() / 1000000.0;
        g_test_results.num_vertices = flat_graph_->get_num_vertices();

        ARTEA_INFO(fmt::format("Graph built with {} vertices", g_test_results.num_vertices));
        ARTEA_INFO(fmt::format("Build time: {:.2f} s", g_test_results.build_time_s));

        // Convert to flat search graph
        ARTEA_INFO("Converting to flat search graph...");
        start_time = std::chrono::high_resolution_clock::now();

        flat_search_graph_ = std::make_unique<flat_search_graph_t>(
            search_graph_converter_t::from_flat_graph(*flat_graph_, g_config.extracted_nbr_size)
        );

        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        g_test_results.conversion_time_ms = duration.count() / 1000.0;

        ARTEA_INFO(fmt::format("Conversion time: {:.2f} ms", g_test_results.conversion_time_ms));

        // Get ground truth from dataset
        const auto& query_vecs = dataset_->get_query_vecs();
        g_test_results.num_queries = query_vecs.get_num_vecs();

        ARTEA_INFO(fmt::format("Loaded ground truth for {} queries", g_test_results.num_queries));
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

    ARTEA_INFO("Testing query recall...");

    // Run grid search over candidate queue sizes
    ARTEA_INFO(fmt::format("\nRunning Grid Search: candidate queue size {} to {}, step {}",
        g_config.queue_start, g_config.queue_end, g_config.queue_step));
    ARTEA_INFO("");

    recall_estimator_t recall_estimator;

    for (uint32_t queue_size = g_config.queue_start; queue_size <= g_config.queue_end; queue_size += g_config.queue_step) {
        // Create router with current queue size
        monolayer_graph_router_t<graph_mode_t::search_mode> router(
            base_vecs,
            dist_func,
            flat_search_graph,
            g_config.topk,
            queue_size
        );
        router.initialize();

        // Query all vectors
        auto start_time = std::chrono::high_resolution_clock::now();
        knn_results_t results = router.batch_query(query_vecs);
        auto end_time = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // Compute metrics
        QueryResult result;
        result.candidate_queue_size = queue_size;
        result.query_time_ms = duration.count() / 1000.0;
        result.avg_query_time_us = static_cast<double>(duration.count()) / query_vecs.get_num_vecs();
        result.throughput_qps = query_vecs.get_num_vecs() * 1000000.0 / duration.count();
        result.recall = recall_estimator.calculate_recall_at_k(results, groundtruth, g_config.topk, query_vecs.get_num_vecs());

        g_test_results.query_results.push_back(result);

        ARTEA_INFO(fmt::format("CandidateQueue={:3}: Recall@{}={:.4f}, QPS={:8.2f}, AvgTime={:.2f}ms",
            queue_size, g_config.topk, result.recall, result.throughput_qps, result.query_time_ms));
    }

    // Expect reasonable recall for at least one configuration
    bool has_good_recall = false;
    for (const auto& result : g_test_results.query_results) {
        if (result.recall >= 0.5f) {
            has_good_recall = true;
            break;
        }
    }
    EXPECT_TRUE(has_good_recall) << "At least one configuration should achieve recall >= 50%";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_conv_graph");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--max-nbr-size").default_value(32u).scan<'u', uint32_t>();
    program.add_argument("--extracted-nbr-size").default_value(32u).scan<'u', uint32_t>();
    program.add_argument("--scale-coeffs").default_value(1.0f).scan<'g', float>();
    program.add_argument("--shifted-coeffs").default_value(0.0f).scan<'g', float>();
    program.add_argument("--num-build-loops").default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters").default_value(14u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio").default_value(0.6f).scan<'g', float>();
    program.add_argument("--num-routing-loops").default_value(1u).scan<'u', uint32_t>();
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-config")
        .default_value(std::string("40,100,20"))
        .help("Candidate queue size grid search: start,end,step (default: 40,100,20)");
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

    uint32_t max_nbr_size = program.get<uint32_t>("--max-nbr-size");
    uint32_t reserved_nbr_size = static_cast<uint32_t>(max_nbr_size * 1.5);

    g_config.layer_config = layer_config_t(max_nbr_size, reserved_nbr_size);
    g_config.pruning_config = conv_graph::pruning_config_t(
        program.get<float>("--scale-coeffs"),
        program.get<float>("--shifted-coeffs")
    );
    g_config.propagate_config = conv_graph::propagate_config_t(
        program.get<uint32_t>("--num-build-loops"),
        program.get<uint32_t>("--num-triu-iters"),
        program.get<float>("--prefill-ratio"),
        program.get<uint32_t>("--num-routing-loops")
    );
    g_config.extracted_nbr_size = program.get<uint32_t>("--extracted-nbr-size");
    g_config.topk = program.get<uint32_t>("--topk");

    // Parse candidate queue config
    std::string queue_config_str = program.get<std::string>("--candidate-queue-config");
    {
        std::istringstream ss(queue_config_str);
        std::string token;
        std::vector<uint32_t> values;
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stoul(token));
        }
        if (values.size() != 3) {
            fprintf(stderr, "Error: --candidate-queue-config must have 3 values: start,end,step\n");
            return 1;
        }
        g_config.queue_start = values[0];
        g_config.queue_end = values[1];
        g_config.queue_step = values[2];
    }

    g_config.verbose = program.get<bool>("--verbose");

    // Print test configuration
    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Config path: " << g_config.config_path << std::endl;
    std::cout << "Max nbr size: " << g_config.layer_config.max_nbr_size() << std::endl;
    std::cout << "Extracted nbr size: " << g_config.extracted_nbr_size << std::endl;
    std::cout << "Scale coeffs: " << g_config.pruning_config.scale_coeffs() << std::endl;
    std::cout << "Shifted coeffs: " << g_config.pruning_config.shifted_coeffs() << std::endl;
    std::cout << "Build loops: " << g_config.propagate_config.num_build_loops() << std::endl;
    std::cout << "Triangle updater iterations: " << g_config.propagate_config.num_triu_iters() << std::endl;
    std::cout << "Top-k: " << g_config.topk << std::endl;
    std::cout << "Candidate queue config: " << g_config.queue_start << "," << g_config.queue_end << "," << g_config.queue_step << std::endl;
    std::cout << "Verbose: " << (g_config.verbose ? "true" : "false") << std::endl;
    std::cout << "==========================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    // Print summary table
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                        TEST RESULTS SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "\n--- Configuration ---" << std::endl;
    std::cout << fmt::format("  Dataset:                {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Max Nbr Size:           {}", g_config.layer_config.max_nbr_size()) << std::endl;
    std::cout << fmt::format("  Extracted Nbr Size:     {}", g_config.extracted_nbr_size) << std::endl;
    std::cout << fmt::format("  Scale Coeffs:           {}", g_config.pruning_config.scale_coeffs()) << std::endl;
    std::cout << fmt::format("  Shifted Coeffs:         {}", g_config.pruning_config.shifted_coeffs()) << std::endl;
    std::cout << fmt::format("  Build Loops:            {}", g_config.propagate_config.num_build_loops()) << std::endl;
    std::cout << fmt::format("  Triangle Updater Iters: {}", g_config.propagate_config.num_triu_iters()) << std::endl;
    std::cout << fmt::format("  Top-k:                  {}", g_config.topk) << std::endl;
    std::cout << "\n--- Graph Construction ---" << std::endl;
    std::cout << fmt::format("  Build Time:             {:.2f} s", g_test_results.build_time_s) << std::endl;
    std::cout << fmt::format("  Num Vertices:           {}", g_test_results.num_vertices) << std::endl;
    std::cout << fmt::format("  Conversion Time:        {:.2f} ms", g_test_results.conversion_time_ms) << std::endl;
    std::cout << "\n--- Query Performance ---" << std::endl;
    std::cout << fmt::format("  Num Queries:            {}", g_test_results.num_queries) << std::endl;
    std::cout << fmt::format("  Top-k:                  {}", g_config.topk) << std::endl;
    std::cout << "\n--- Grid Search Results ---" << std::endl;
    std::cout << fmt::format("{:<15} {:<12} {:<12} {:<12}",
        "CandidateQueue", "Recall@k", "QPS", "AvgTime(ms)") << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (const auto& result : g_test_results.query_results) {
        std::cout << fmt::format("{:<15} {:<12.4f} {:<12.2f} {:<12.2f}",
            result.candidate_queue_size,
            result.recall,
            result.throughput_qps,
            result.query_time_ms) << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;

    return result;
}
