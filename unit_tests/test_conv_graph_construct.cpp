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
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>

using namespace artea;
using namespace artea::cpu;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    std::string metric;
    // layer_config_t is metric/dim-independent; the pruning / propagate configs
    // are now <Metric, Dim> aliases, so we keep their raw params here and build the
    // objects inside the dispatched body.
    layer_config_t layer_config{16};
    float    scale_coeffs       = 1.0f;
    float    shifted_coeffs     = 0.0f;
    uint32_t num_build_loops    = 4;
    uint32_t num_triu_iters     = 14;
    float    prefill_ratio      = 0.6f;
    uint32_t num_routing_loops  = 1;
    uint32_t routing_topk       = 64;
    uint32_t routing_queue_size = 96;
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

        const auto& base_vecs = dataset_->get_base_vecs();

        // Metric from the --metric input, padded dim from the loaded dataset;
        // the dataset and the resulting compact::refining_graph_t are
        // metric/dim-independent, so only the build runs behind <Metric, Dim>.
        dataset_info_ = DatasetInfra{parse_metric(g_config.metric), base_vecs.get_vec_dim()};

        if (g_config.verbose) {
            ARTEA_INFO(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
            ARTEA_INFO(fmt::format("Max nbr size: {}", g_config.layer_config.max_nbr_size()));
            ARTEA_INFO(fmt::format("Extracted nbr size: {}", g_config.extracted_nbr_size));
            ARTEA_INFO(fmt::format("Scale coeffs: {}", g_config.scale_coeffs));
            ARTEA_INFO(fmt::format("Shifted coeffs: {}", g_config.shifted_coeffs));
            ARTEA_INFO(fmt::format("Build loops: {}", g_config.num_build_loops));
            ARTEA_INFO(fmt::format("Triangle updater iterations: {}", g_config.num_triu_iters));
            ARTEA_INFO(fmt::format("Top-k: {}", g_config.topk));
            ARTEA_INFO(fmt::format("Candidate queue size range: {} to {} step {}",
                g_config.queue_start, g_config.queue_end, g_config.queue_step));
        }

        infra_dispatch(dataset_info_, ARTEA_METRIC_LAMBDA(void) {
            conv_graph::pruning_config_t<Metric, Dim> pruning_config(
                g_config.scale_coeffs, g_config.shifted_coeffs);
            conv_graph::propagate_config_t<Metric, Dim> propagate_config(
                g_config.num_build_loops,
                g_config.num_triu_iters,
                g_config.prefill_ratio,
                g_config.num_routing_loops,
                g_config.routing_topk,
                g_config.routing_queue_size);

            // Build convergent graph
            ARTEA_INFO("Building convergent graph...");
            auto start_time = std::chrono::high_resolution_clock::now();

            conv_graph::index_t<Metric, Dim> graph_index = std::move(
                conv_graph::factory_t<Metric, Dim>::construct_graph(
                    base_vecs,
                    g_config.layer_config,
                    pruning_config,
                    propagate_config
                ).graph
            );

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            g_test_results.build_time_s = duration.count() / 1000000.0;
            g_test_results.num_vertices = graph_index.get_num_vertices();

            ARTEA_INFO(fmt::format("Graph built with {} vertices", g_test_results.num_vertices));
            ARTEA_INFO(fmt::format("Build time: {:.2f} s", g_test_results.build_time_s));

            // Convert to flat search graph (metric/dim-independent result)
            ARTEA_INFO("Converting to flat search graph...");
            start_time = std::chrono::high_resolution_clock::now();

            compact_refining_graph_ = std::make_unique<compact::refining_graph_t>(
                refining_graph_compactor_t::compact_graph(graph_index, g_config.extracted_nbr_size)
            );

            end_time = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            g_test_results.conversion_time_ms = duration.count() / 1000.0;

            ARTEA_INFO(fmt::format("Conversion time: {:.2f} ms", g_test_results.conversion_time_ms));
        });

        // Get ground truth from dataset
        const auto& query_vecs = dataset_->get_query_vecs();
        g_test_results.num_queries = query_vecs.get_num_vecs();

        ARTEA_INFO(fmt::format("Loaded ground truth for {} queries", g_test_results.num_queries));
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    DatasetInfra get_dataset_info() const { return dataset_info_; }
    compact::refining_graph_t& get_compact_refining_graph() { return *compact_refining_graph_; }
    const idlist_array_t& get_groundtruth() { return dataset_->get_gt_vecs(); }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<compact::refining_graph_t> compact_refining_graph_;
    DatasetInfra dataset_info_{};
};

class ConvGraphTest : public ::testing::Test {};

TEST_F(ConvGraphTest, QueryRecall) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& compact_refining_graph = provider.get_compact_refining_graph();
    const auto& groundtruth = provider.get_groundtruth();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();

    ARTEA_INFO("Testing query recall...");

    // Run grid search over candidate queue sizes
    ARTEA_INFO(fmt::format("\nRunning Grid Search: candidate queue size {} to {}, step {}",
        g_config.queue_start, g_config.queue_end, g_config.queue_step));
    ARTEA_INFO("");

    infra_dispatch(provider.get_dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        // Stateless functor: the dimension is a compile-time trait now.
        dist_func_t<Metric, Dim> dist_func;
        recall_estimator_t<Metric, Dim> recall_estimator;

        for (uint32_t queue_size = g_config.queue_start; queue_size <= g_config.queue_end; queue_size += g_config.queue_step) {
            // Create single-layer router with current queue size
            single_layer_router_t<Metric, Dim> router(
                base_vecs,
                dist_func,
                g_config.topk,
                queue_size
            );
            router.initialize();

            // Warmup runs
            for (uint32_t w = 0; w < g_config.warmup_runs; ++w) {
                [[maybe_unused]] auto _ = router.batch_query(query_vecs, compact_refining_graph);
            }

            // Test runs with averaging
            double total_time_us = 0.0;
            float total_recall = 0.0f;
            for (uint32_t r = 0; r < g_config.test_runs; ++r) {
                auto start_time = std::chrono::high_resolution_clock::now();
                knn_results_t<Metric, Dim> results = router.batch_query(query_vecs, compact_refining_graph);
                auto end_time = std::chrono::high_resolution_clock::now();
                total_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
                total_recall += recall_estimator.calculate_recall_at_k(results, groundtruth, g_config.topk, query_vecs.get_num_vecs());
            }
            double avg_time_us = total_time_us / g_config.test_runs;

            // Compute metrics
            QueryResult result;
            result.candidate_queue_size = queue_size;
            result.query_time_ms = avg_time_us / 1000.0;
            result.avg_query_time_us = avg_time_us / query_vecs.get_num_vecs();
            result.throughput_qps = query_vecs.get_num_vecs() * 1000000.0 / avg_time_us;
            result.recall = total_recall / g_config.test_runs;

            g_test_results.query_results.push_back(result);

            ARTEA_INFO(fmt::format("CandidateQueue={:3}: Recall@{}={:.4f}, QPS={:8.2f}, AvgTime={:.2f}ms ({}w+{}r)",
                queue_size, g_config.topk, result.recall, result.throughput_qps, result.query_time_ms,
                g_config.warmup_runs, g_config.test_runs));
        }
    });

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
    program.add_argument("-c", "--config").default_value(artea::default_dataset_config_path());
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--metric").default_value(std::string("euclidean")).help("Distance metric: 'euclidean', 'inner_product', or 'cosine'");
    program.add_argument("--max-nbr-size").default_value(96u).scan<'u', uint32_t>();
    program.add_argument("--extracted-nbr-size").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--scale-coeffs").default_value(1.1f).scan<'g', float>();
    program.add_argument("--shifted-coeffs").default_value(3.0f).scan<'g', float>();
    program.add_argument("--num-build-loops").default_value(5u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters").default_value(12u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio").default_value(0.6f).scan<'g', float>();
    program.add_argument("--num-routing-loops").default_value(1u).scan<'u', uint32_t>();
    program.add_argument("--routing-topk").default_value(64u).scan<'u', uint32_t>()
        .help("Routing updater top-k (default: 96)");
    program.add_argument("--routing-queue-size").default_value(96u).scan<'u', uint32_t>()
        .help("Routing updater candidate queue size (default: 128)");
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-config")
        .default_value(std::string("40,200,20"))
        .help("Candidate queue size grid search: start,end,step (default: 40,200,20)");
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
    g_config.metric = program.get<std::string>("--metric");

    uint32_t max_nbr_size = program.get<uint32_t>("--max-nbr-size");

    g_config.layer_config = layer_config_t(max_nbr_size);
    g_config.scale_coeffs       = program.get<float>("--scale-coeffs");
    g_config.shifted_coeffs     = program.get<float>("--shifted-coeffs");
    g_config.num_build_loops    = program.get<uint32_t>("--num-build-loops");
    g_config.num_triu_iters     = program.get<uint32_t>("--num-triu-iters");
    g_config.prefill_ratio      = program.get<float>("--prefill-ratio");
    g_config.num_routing_loops  = program.get<uint32_t>("--num-routing-loops");
    g_config.routing_topk       = program.get<uint32_t>("--routing-topk");
    g_config.routing_queue_size = program.get<uint32_t>("--routing-queue-size");
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

    g_config.warmup_runs = program.get<uint32_t>("--warmup-runs");
    g_config.test_runs = program.get<uint32_t>("--test-runs");
    g_config.verbose = program.get<bool>("--verbose");

    // Print test configuration
    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Config path: " << g_config.config_path << std::endl;
    std::cout << "Max nbr size: " << g_config.layer_config.max_nbr_size() << std::endl;
    std::cout << "Extracted nbr size: " << g_config.extracted_nbr_size << std::endl;
    std::cout << "Scale coeffs: " << g_config.scale_coeffs << std::endl;
    std::cout << "Shifted coeffs: " << g_config.shifted_coeffs << std::endl;
    std::cout << "Build loops: " << g_config.num_build_loops << std::endl;
    std::cout << "Triangle updater iterations: " << g_config.num_triu_iters << std::endl;
    std::cout << "Top-k: " << g_config.topk << std::endl;
    std::cout << "Candidate queue config: " << g_config.queue_start << "," << g_config.queue_end << "," << g_config.queue_step << std::endl;
    std::cout << "Warmup runs: " << g_config.warmup_runs << std::endl;
    std::cout << "Test runs: " << g_config.test_runs << std::endl;
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
    std::cout << fmt::format("  Scale Coeffs:           {}", g_config.scale_coeffs) << std::endl;
    std::cout << fmt::format("  Shifted Coeffs:         {}", g_config.shifted_coeffs) << std::endl;
    std::cout << fmt::format("  Build Loops:            {}", g_config.num_build_loops) << std::endl;
    std::cout << fmt::format("  Triangle Updater Iters: {}", g_config.num_triu_iters) << std::endl;
    std::cout << fmt::format("  Top-k:                  {}", g_config.topk) << std::endl;
    std::cout << fmt::format("  Warmup runs:            {}", g_config.warmup_runs) << std::endl;
    std::cout << fmt::format("  Test runs:              {}", g_config.test_runs) << std::endl;
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
