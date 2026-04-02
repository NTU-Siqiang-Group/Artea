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
    static constexpr float knn_scale_coeffs = 1.0f;
    static constexpr float knn_shifted_coeffs = 0.0f;
    knn_graph::pruning_config_t pruning_config{knn_scale_coeffs, knn_shifted_coeffs};
    knn_graph::propagate_config_t propagate_config{4, 14};
    uint32_t extracted_nbr_size;
    uint32_t topk;
    uint32_t queue_start;
    uint32_t queue_end;
    uint32_t queue_step;
    uint32_t warmup_runs;
    uint32_t test_runs;
    bool verbose;
} g_config;

struct QueryResult {
    uint32_t candidate_queue_size;
    double avg_query_time_us;
    double throughput_qps;
    float recall;
};

struct TestResults {
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

        if (g_config.verbose) {
            ARTEA_INFO(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
            ARTEA_INFO(fmt::format("Max nbr size: {}", g_config.layer_config.max_nbr_size()));
            ARTEA_INFO(fmt::format("Extracted nbr size: {}", g_config.extracted_nbr_size));
            ARTEA_INFO(fmt::format("Build loops: {}", g_config.propagate_config.num_build_loops()));
            ARTEA_INFO(fmt::format("Triangle updater iterations: {}", g_config.propagate_config.num_triu_iters()));
            ARTEA_INFO(fmt::format("Top-k: {}", g_config.topk));
        }

        // Build KNN graph using dataset version (per-iter profiling logged inside)
        ARTEA_INFO("Building KNN graph (dataset mode, per-iter profiling)...");
        knn_graph::factory_t::profile_graph_quality(
            *dataset_,
            g_config.layer_config,
            g_config.pruning_config,
            g_config.propagate_config
        );
        flat_graph_ = std::make_unique<knn_graph::index_t>(knn_graph::factory_t::construct_graph(
            dataset_->get_base_vecs(),
            g_config.layer_config,
            g_config.pruning_config,
            g_config.propagate_config
        ));

        // Convert to flat search graph
        ARTEA_INFO("Converting to flat search graph...");
        auto t0 = std::chrono::high_resolution_clock::now();
        flat_search_graph_ = std::make_unique<flat_search_graph_t>(
            search_graph_converter_t::from_flat_graph(*flat_graph_, g_config.extracted_nbr_size)
        );
        auto t1 = std::chrono::high_resolution_clock::now();
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
    std::unique_ptr<knn_graph::index_t> flat_graph_;
    std::unique_ptr<flat_search_graph_t> flat_search_graph_;
};

class KnnGraphQualityTest : public ::testing::Test {};

TEST_F(KnnGraphQualityTest, QueryRecall) {
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

        // Warmup runs
        for (uint32_t w = 0; w < g_config.warmup_runs; ++w) {
            [[maybe_unused]] auto _ = router.batch_query(query_vecs);
        }

        // Test runs with averaging
        double total_time_us = 0.0;
        float total_recall = 0.0f;
        for (uint32_t r = 0; r < g_config.test_runs; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            knn_results_t results = router.batch_query(query_vecs);
            auto t1 = std::chrono::high_resolution_clock::now();
            total_time_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            total_recall += recall_estimator.calculate_recall_at_k(results, groundtruth, g_config.topk, query_vecs.get_num_vecs());
        }
        double avg_time_us = total_time_us / g_config.test_runs;

        QueryResult result;
        result.candidate_queue_size = queue_size;
        result.avg_query_time_us = avg_time_us / query_vecs.get_num_vecs();
        result.throughput_qps = query_vecs.get_num_vecs() * 1e6 / avg_time_us;
        result.recall = total_recall / g_config.test_runs;
        g_test_results.query_results.push_back(result);

        ARTEA_INFO(fmt::format("CandidateQueue={:3}: Recall@{}={:.4f}, QPS={:8.2f}, AvgTime={:.2f}us ({}w+{}r)",
            queue_size, g_config.topk, result.recall, result.throughput_qps, result.avg_query_time_us,
            g_config.warmup_runs, g_config.test_runs));
    }

    bool has_good_recall = false;
    for (const auto& r : g_test_results.query_results) {
        if (r.recall >= 0.5f) { has_good_recall = true; break; }
    }
    EXPECT_TRUE(has_good_recall) << "At least one configuration should achieve recall >= 50%";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("profile_knn_graph_quality");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--max-nbr-size").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--extracted-nbr-size").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--num-build-loops").default_value(5u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters").default_value(12u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio").default_value(0.5f).scan<'g', float>();
    program.add_argument("--num-routing-loops").default_value(1u).scan<'u', uint32_t>();
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-config")
        .default_value(std::string("40,200,20"))
        .help("start,end,step");
    program.add_argument("--warmup-runs").default_value(5u).scan<'u', uint32_t>();
    program.add_argument("--test-runs").default_value(10u).scan<'u', uint32_t>();
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
    g_config.layer_config = layer_config_t(max_nbr_size, static_cast<uint32_t>(max_nbr_size * 1.5));
    g_config.propagate_config = knn_graph::propagate_config_t(
        program.get<uint32_t>("--num-build-loops"),
        program.get<uint32_t>("--num-triu-iters"),
        program.get<float>("--prefill-ratio"),
        program.get<uint32_t>("--num-routing-loops")
    );
    g_config.extracted_nbr_size = program.get<uint32_t>("--extracted-nbr-size");
    g_config.topk = program.get<uint32_t>("--topk");
    g_config.warmup_runs = program.get<uint32_t>("--warmup-runs");
    g_config.test_runs = program.get<uint32_t>("--test-runs");
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
    std::cout << fmt::format("  Dataset:              {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Config path:          {}", g_config.config_path) << std::endl;
    std::cout << fmt::format("  Max nbr size:         {}", g_config.layer_config.max_nbr_size()) << std::endl;
    std::cout << fmt::format("  Extracted nbr size:   {}", g_config.extracted_nbr_size) << std::endl;
    std::cout << fmt::format("  Build loops:          {}", g_config.propagate_config.num_build_loops()) << std::endl;
    std::cout << fmt::format("  Triangle updater its: {}", g_config.propagate_config.num_triu_iters()) << std::endl;
    std::cout << fmt::format("  Prefill ratio:        {}", g_config.propagate_config.prefill_ratio()) << std::endl;
    std::cout << fmt::format("  Routing loops:        {}", g_config.propagate_config.num_routing_loops()) << std::endl;
    std::cout << fmt::format("  Top-k:                {}", g_config.topk) << std::endl;
    std::cout << fmt::format("  Queue config:         {},{},{}", g_config.queue_start, g_config.queue_end, g_config.queue_step) << std::endl;
    std::cout << fmt::format("  Warmup runs:          {}", g_config.warmup_runs) << std::endl;
    std::cout << fmt::format("  Test runs:            {}", g_config.test_runs) << std::endl;
    std::cout << fmt::format("  Verbose:              {}", g_config.verbose ? "true" : "false") << std::endl;
    std::cout << "=====================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    TEST RESULTS SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << fmt::format("  Dataset:            {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Num Vertices:       {}", g_test_results.num_vertices) << std::endl;
    std::cout << fmt::format("  Num Queries:        {}", g_test_results.num_queries) << std::endl;
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
