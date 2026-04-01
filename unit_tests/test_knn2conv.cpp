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

    // KNN graph build params
    layer_config_t knn_layer_config{64, 96};
    static constexpr float knn_scale_coeffs = 1.0f;
    static constexpr float knn_shifted_coeffs = 0.0f;
    knn_graph::pruning_config_t knn_pruning_config{knn_scale_coeffs, knn_shifted_coeffs};
    knn_graph::propagate_config_t knn_propagate_config{4, 14};

    // Conv graph refinement params
    conv_graph::pruning_config_t conv_pruning_config{1.0f, 0.0f};

    uint32_t extracted_nbr_size;
    uint32_t topk;
    uint32_t queue_start;
    uint32_t queue_end;
    uint32_t queue_step;
    bool verbose;
} g_config;

struct QueryResult {
    uint32_t candidate_queue_size;
    double avg_query_time_us;
    double throughput_qps;
    float recall;
};

struct TestResults {
    double knn_build_time_s = 0.0;
    double conv_build_time_s = 0.0;
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
        g_test_results.num_vertices = static_cast<uint32_t>(base_vecs.get_num_vecs());
        g_test_results.num_queries = static_cast<uint32_t>(dataset_->get_query_vecs().get_num_vecs());

        // Part 1: Build KNN graph
        ARTEA_INFO("Part 1: Building KNN graph...");
        auto t0 = std::chrono::high_resolution_clock::now();

        knn_graph::index_t knn_index = knn_graph::factory_t::construct_graph(
            base_vecs,
            g_config.knn_layer_config,
            g_config.knn_pruning_config,
            g_config.knn_propagate_config
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        g_test_results.knn_build_time_s =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        ARTEA_INFO(fmt::format("KNN graph built in {:.2f} s", g_test_results.knn_build_time_s));

        // Part 2: Convert knn_graph -> conv_graph via move
        ARTEA_INFO("Part 2: Converting KNN graph to conv_graph (move + triangle/reverse pruning)...");
        t0 = std::chrono::high_resolution_clock::now();

        conv_graph_ = std::make_unique<conv_graph::index_t>(conv_graph::factory_t::construct_graph(
            std::move(knn_index),
            g_config.conv_pruning_config
        ));

        t1 = std::chrono::high_resolution_clock::now();
        g_test_results.conv_build_time_s =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        ARTEA_INFO(fmt::format("Conv graph refined in {:.2f} s", g_test_results.conv_build_time_s));

        // Convert to flat search graph
        ARTEA_INFO("Converting to flat search graph...");
        t0 = std::chrono::high_resolution_clock::now();
        flat_search_graph_ = std::make_unique<flat_search_graph_t>(
            search_graph_converter_t::from_flat_graph(*conv_graph_, g_config.extracted_nbr_size)
        );
        t1 = std::chrono::high_resolution_clock::now();
        g_test_results.conversion_time_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        ARTEA_INFO(fmt::format("Conversion time: {:.2f} ms", g_test_results.conversion_time_ms));
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }
    flat_search_graph_t& get_flat_search_graph() { return *flat_search_graph_; }
    const idlist_array_t& get_groundtruth() { return dataset_->get_gt_vecs(); }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<conv_graph::index_t> conv_graph_;
    std::unique_ptr<flat_search_graph_t> flat_search_graph_;
};

class Knn2ConvTest : public ::testing::Test {};

TEST_F(Knn2ConvTest, QueryRecall) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();
    auto& flat_search_graph = provider.get_flat_search_graph();
    const auto& groundtruth = provider.get_groundtruth();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();

    recall_estimator_t recall_estimator;

    ARTEA_INFO(fmt::format("\nRunning Grid Search: candidate queue size {} to {}, step {}",
        g_config.queue_start, g_config.queue_end, g_config.queue_step));

    for (uint32_t queue_size = g_config.queue_start; queue_size <= g_config.queue_end; queue_size += g_config.queue_step) {
        monolayer_graph_router_t<graph_mode_t::search_mode> router(base_vecs, dist_func, flat_search_graph, g_config.topk, queue_size);
        router.initialize();

        auto t0 = std::chrono::high_resolution_clock::now();
        knn_results_t results = router.batch_query(query_vecs);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        QueryResult result;
        result.candidate_queue_size = queue_size;
        result.avg_query_time_us = static_cast<double>(us) / query_vecs.get_num_vecs();
        result.throughput_qps = query_vecs.get_num_vecs() * 1e6 / us;
        result.recall = recall_estimator.calculate_recall_at_k(results, groundtruth, g_config.topk, query_vecs.get_num_vecs());
        g_test_results.query_results.push_back(result);

        ARTEA_INFO(fmt::format("CandidateQueue={:3}: Recall@{}={:.4f}, QPS={:8.2f}, AvgTime={:.2f}us",
            queue_size, g_config.topk, result.recall, result.throughput_qps, result.avg_query_time_us));
    }

    bool has_good_recall = false;
    for (const auto& r : g_test_results.query_results) {
        if (r.recall >= 0.5f) { has_good_recall = true; break; }
    }
    EXPECT_TRUE(has_good_recall) << "At least one configuration should achieve recall >= 50%";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_knn2conv");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));

    // KNN graph params
    program.add_argument("--knn-max-nbr-size").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--knn-num-build-loops").default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--knn-num-triu-iters").default_value(14u).scan<'u', uint32_t>();
    program.add_argument("--knn-prefill-ratio").default_value(0.5f).scan<'g', float>();
    program.add_argument("--knn-num-routing-loops").default_value(1u).scan<'u', uint32_t>();

    // Conv graph refinement params
    program.add_argument("--conv-scale-coeffs").default_value(1.0f).scan<'g', float>();
    program.add_argument("--conv-shifted-coeffs").default_value(0.0f).scan<'g', float>();

    // Search params
    program.add_argument("--extracted-nbr-size").default_value(32u).scan<'u', uint32_t>();
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-config")
        .default_value(std::string("40,200,20"))
        .help("start,end,step");
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

    // KNN graph config
    uint32_t knn_max_nbr_size = program.get<uint32_t>("--knn-max-nbr-size");
    g_config.knn_layer_config = layer_config_t(knn_max_nbr_size, static_cast<uint32_t>(knn_max_nbr_size * 1.5));
    g_config.knn_propagate_config = knn_graph::propagate_config_t(
        program.get<uint32_t>("--knn-num-build-loops"),
        program.get<uint32_t>("--knn-num-triu-iters"),
        program.get<float>("--knn-prefill-ratio"),
        program.get<uint32_t>("--knn-num-routing-loops")
    );

    // Conv graph refinement config
    g_config.conv_pruning_config = conv_graph::pruning_config_t(
        program.get<float>("--conv-scale-coeffs"),
        program.get<float>("--conv-shifted-coeffs")
    );

    // Search config
    g_config.extracted_nbr_size = program.get<uint32_t>("--extracted-nbr-size");
    g_config.topk = program.get<uint32_t>("--topk");
    g_config.verbose = program.get<bool>("--verbose");

    std::string queue_config_str = program.get<std::string>("--candidate-queue-config");
    {
        std::istringstream ss(queue_config_str);
        std::string token;
        std::vector<uint32_t> values;
        while (std::getline(ss, token, ',')) values.push_back(std::stoul(token));
        if (values.size() != 3) {
            fprintf(stderr, "Error: --candidate-queue-config must have 3 values: start,end,step\n");
            return 1;
        }
        g_config.queue_start = values[0];
        g_config.queue_end = values[1];
        g_config.queue_step = values[2];
    }

    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << fmt::format("  Dataset:                  {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Config path:              {}", g_config.config_path) << std::endl;
    std::cout << "  --- KNN Graph ---" << std::endl;
    std::cout << fmt::format("  KNN max nbr size:         {}", g_config.knn_layer_config.max_nbr_size()) << std::endl;
    std::cout << fmt::format("  KNN build loops:          {}", g_config.knn_propagate_config.num_build_loops()) << std::endl;
    std::cout << fmt::format("  KNN triangle updater its: {}", g_config.knn_propagate_config.num_triu_iters()) << std::endl;
    std::cout << fmt::format("  KNN prefill ratio:        {}", g_config.knn_propagate_config.prefill_ratio()) << std::endl;
    std::cout << fmt::format("  KNN routing loops:        {}", g_config.knn_propagate_config.num_routing_loops()) << std::endl;
    std::cout << "  --- Conv Refinement ---" << std::endl;
    std::cout << fmt::format("  Conv max nbr size:        {} (inherited from KNN)", g_config.knn_layer_config.max_nbr_size()) << std::endl;
    std::cout << fmt::format("  Conv scale coeffs:        {}", g_config.conv_pruning_config.scale_coeffs()) << std::endl;
    std::cout << fmt::format("  Conv shifted coeffs:      {}", g_config.conv_pruning_config.shifted_coeffs()) << std::endl;
    std::cout << "  --- Search ---" << std::endl;
    std::cout << fmt::format("  Extracted nbr size:       {}", g_config.extracted_nbr_size) << std::endl;
    std::cout << fmt::format("  Top-k:                    {}", g_config.topk) << std::endl;
    std::cout << fmt::format("  Queue config:             {},{},{}", g_config.queue_start, g_config.queue_end, g_config.queue_step) << std::endl;
    std::cout << fmt::format("  Verbose:                  {}", g_config.verbose ? "true" : "false") << std::endl;
    std::cout << "=====================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    TEST RESULTS SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << fmt::format("  Dataset:            {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Num Vertices:       {}", g_test_results.num_vertices) << std::endl;
    std::cout << fmt::format("  Num Queries:        {}", g_test_results.num_queries) << std::endl;
    std::cout << fmt::format("  KNN Build Time:     {:.2f} s", g_test_results.knn_build_time_s) << std::endl;
    std::cout << fmt::format("  Conv Refine Time:   {:.2f} s", g_test_results.conv_build_time_s) << std::endl;
    std::cout << fmt::format("  Total Build Time:   {:.2f} s", g_test_results.knn_build_time_s + g_test_results.conv_build_time_s) << std::endl;
    std::cout << fmt::format("  Conversion Time:    {:.2f} ms", g_test_results.conversion_time_ms) << std::endl;
    std::cout << "\n--- Grid Search Results ---" << std::endl;
    std::cout << fmt::format("{:<15} {:<12} {:<12} {:<12}",
        "CandidateQueue", "Recall@k", "QPS", "AvgTime(us)") << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    for (const auto& r : g_test_results.query_results) {
        std::cout << fmt::format("{:<15} {:<12.4f} {:<12.2f} {:<12.2f}",
            r.candidate_queue_size, r.recall, r.throughput_qps, r.avg_query_time_us) << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;

    return result;
}
