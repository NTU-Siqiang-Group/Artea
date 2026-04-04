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

    // Bottom layer config
    uint32_t bl_max_nbr_size;
    float bl_scale_coeffs;
    float bl_shifted_coeffs;

    // Upper layer config
    uint32_t ul_max_nbr_size;
    float ul_scale_coeffs;
    float ul_shifted_coeffs;

    // Propagate config (shared)
    uint32_t num_build_loops;
    uint32_t num_triu_iters;
    float prefill_ratio;
    uint32_t num_routing_loops;
    uint32_t routing_topk;
    uint32_t routing_queue_size;

    // R-net config
    float mis_radix;
    uint32_t mis_max_power;
    float rnet_beta;

    // Query config
    uint32_t topk;
    uint32_t queue_start;
    uint32_t queue_end;
    uint32_t queue_step;
    uint32_t bl_extracted_nbr_size;
    uint32_t ul_extracted_nbr_size;

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

struct LayerMetrics {
    uint32_t layer_id;
    uint32_t num_vertices;
    uint32_t num_edges;
    float layer_ratio;
};

struct TestResults {
    double total_construction_time_ms = 0.0;
    uint32_t num_layers = 0;
    uint32_t total_base_vecs = 0;
    uint32_t num_queries = 0;
    std::vector<LayerMetrics> layer_metrics;
    std::vector<QueryResult> query_results;
};

TestResults g_results;

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

        // Always shuffle dataset
        ARTEA_INFO("Shuffling dataset...");
        dataset_->shuffle_in_place();

        const auto& base_vecs = dataset_->get_base_vecs();
        g_results.total_base_vecs = base_vecs.get_num_vecs();

        // Create configs
        layer_config_t bottom_layer_config(
            g_config.bl_max_nbr_size,
            static_cast<uint32_t>(g_config.bl_max_nbr_size * 1.5));
        layer_config_t upper_layer_config(
            g_config.ul_max_nbr_size,
            static_cast<uint32_t>(g_config.ul_max_nbr_size * 1.5));

        artea_graph::pruning_config_t bottom_pruning_config(
            g_config.bl_scale_coeffs, g_config.bl_shifted_coeffs);
        artea_graph::pruning_config_t upper_pruning_config(
            g_config.ul_scale_coeffs, g_config.ul_shifted_coeffs);

        artea_graph::propagate_config_t propagate_config(
            g_config.num_build_loops, g_config.num_triu_iters,
            g_config.prefill_ratio, g_config.num_routing_loops,
            g_config.routing_topk, g_config.routing_queue_size);

        artea_graph::rnet_config_t rnet_config(
            g_config.mis_radix, g_config.mis_max_power, g_config.rnet_beta);

        // Construct hierarchical graph
        ARTEA_INFO("Constructing hierarchical Artea graph...");
        auto construction_start = std::chrono::high_resolution_clock::now();

        auto graph = artea_graph::factory_t::mis_construct_graph(
            base_vecs,
            bottom_layer_config, upper_layer_config,
            bottom_pruning_config, upper_pruning_config,
            propagate_config, rnet_config
        );
        hierarchical_graph_ = std::make_unique<artea_graph::index_t>(std::move(graph));

        auto construction_end = std::chrono::high_resolution_clock::now();
        g_results.total_construction_time_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(construction_end - construction_start).count() / 1000.0;
        g_results.num_layers = hierarchical_graph_->get_num_layers();

        ARTEA_INFO(fmt::format("Graph construction completed: {} layers in {:.2f} ms",
            g_results.num_layers, g_results.total_construction_time_ms));

        // Collect layer metrics
        for (uint32_t lid = 0; lid < g_results.num_layers; ++lid) {
            const auto& layer_vecs = hierarchical_graph_->get_hier_vecs_manager().get_layer_vecs(lid);
            const auto& layer_graph = hierarchical_graph_->get_layer_graph(lid);

            LayerMetrics metrics;
            metrics.layer_id = lid;
            metrics.num_vertices = layer_vecs.get_num_vecs();

            uint32_t total_edges = 0;
            const auto& nbrs_arr = layer_graph.get_nbrs_arr();
            for (const auto& nbrs : nbrs_arr) { total_edges += nbrs.size(); }
            metrics.num_edges = total_edges;
            metrics.layer_ratio = 100.0f * metrics.num_vertices / g_results.total_base_vecs;

            g_results.layer_metrics.push_back(metrics);
        }
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    artea_graph::index_t& get_hierarchical_graph() { return *hierarchical_graph_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<artea_graph::index_t> hierarchical_graph_;
};

class ArteaGraphConstructTest : public ::testing::Test {};

TEST_F(ArteaGraphConstructTest, VerifyGraphConstruction) {
    auto& hg = DataProvider::instance().get_hierarchical_graph();

    EXPECT_GE(g_results.num_layers, 1) << "Should have at least 1 layer";

    for (uint32_t lid = 0; lid < g_results.num_layers; ++lid) {
        const auto& m = g_results.layer_metrics[lid];
        EXPECT_GT(m.num_vertices, 0) << fmt::format("Layer {} should have vertices", lid);
        EXPECT_GT(m.num_edges, 0) << fmt::format("Layer {} should have edges", lid);

        const auto& layer_graph = hg.get_layer_graph(lid);
        EXPECT_EQ(layer_graph.get_num_vertices(), m.num_vertices);
    }

    for (uint32_t lid = 1; lid < g_results.num_layers; ++lid) {
        EXPECT_LT(g_results.layer_metrics[lid].num_vertices,
                  g_results.layer_metrics[lid - 1].num_vertices);
    }
}

TEST_F(ArteaGraphConstructTest, QueryRecall) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& hierarchical_graph = provider.get_hierarchical_graph();
    const auto& groundtruth = dataset.get_gt_vecs();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    g_results.num_queries = query_vecs.get_num_vecs();

    // Convert to hierarchical search graph (copy inter-layer links, no move)
    auto hierarchical_search_graph = search_graph_converter_t::from_hierarchical_graph(
        hierarchical_graph,
        g_config.bl_extracted_nbr_size,
        g_config.ul_extracted_nbr_size);

    recall_estimator_t recall_estimator;
    dist_func_t dist_func(base_vecs.get_vec_dim());

    for (uint32_t queue_size = g_config.queue_start; queue_size <= g_config.queue_end; queue_size += g_config.queue_step) {
        hierarchical_graph_router_t<graph_mode_t::search_mode> router(
            base_vecs, dist_func, hierarchical_search_graph, g_config.topk, queue_size);
        router.initialize();

        for (uint32_t w = 0; w < g_config.warmup_runs; ++w) {
            [[maybe_unused]] auto _ = router.batch_query(query_vecs);
        }

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
        result.query_time_ms = avg_time_us / 1000.0;
        result.avg_query_time_us = avg_time_us / query_vecs.get_num_vecs();
        result.throughput_qps = query_vecs.get_num_vecs() * 1e6 / avg_time_us;
        result.recall = total_recall / g_config.test_runs;

        g_results.query_results.push_back(result);

        ARTEA_INFO(fmt::format("CandidateQueue={:3}: Recall@{}={:.4f}, QPS={:8.2f}, AvgTime={:.2f}ms ({}w+{}r)",
            queue_size, g_config.topk, result.recall, result.throughput_qps, result.query_time_ms,
            g_config.warmup_runs, g_config.test_runs));
    }

    bool has_good_recall = false;
    for (const auto& r : g_results.query_results) {
        if (r.recall >= 0.5f) { has_good_recall = true; break; }
    }
    EXPECT_TRUE(has_good_recall) << "At least one configuration should achieve recall >= 50%";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_artea_graph");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));

    // Bottom layer config
    program.add_argument("--bl-max-nbr-size").default_value(96u).scan<'u', uint32_t>();
    program.add_argument("--bl-scale-coeffs").default_value(1.10f).scan<'g', float>();
    program.add_argument("--bl-shifted-coeffs").default_value(0.10f).scan<'g', float>();

    // Upper layer config
    program.add_argument("--ul-max-nbr-size").default_value(96u).scan<'u', uint32_t>();
    program.add_argument("--ul-scale-coeffs").default_value(1.10f).scan<'g', float>();
    program.add_argument("--ul-shifted-coeffs").default_value(0.10f).scan<'g', float>();

    // Propagate config (shared)
    program.add_argument("--num-build-loops").default_value(5u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters").default_value(12u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio").default_value(0.34f).scan<'g', float>();
    program.add_argument("--num-routing-loops").default_value(1u).scan<'u', uint32_t>();
    program.add_argument("--routing-topk").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--routing-queue-size").default_value(96u).scan<'u', uint32_t>();

    // R-net config
    program.add_argument("--mis-radix").default_value(1.2f).scan<'g', float>();
    program.add_argument("--mis-max-power").default_value(12u).scan<'u', uint32_t>();
    program.add_argument("--beta").default_value(1.44f).scan<'g', float>();

    // Router parameters
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-config")
        .default_value(std::string("40,200,20"))
        .help("Candidate queue size grid search: start,end,step (default: 40,200,20)");
    program.add_argument("--bl-extracted-nbr-size").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--ul-extracted-nbr-size").default_value(32u).scan<'u', uint32_t>();

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

    g_config.bl_max_nbr_size = program.get<uint32_t>("--bl-max-nbr-size");
    g_config.bl_scale_coeffs = program.get<float>("--bl-scale-coeffs");
    g_config.bl_shifted_coeffs = program.get<float>("--bl-shifted-coeffs");

    g_config.ul_max_nbr_size = program.get<uint32_t>("--ul-max-nbr-size");
    g_config.ul_scale_coeffs = program.get<float>("--ul-scale-coeffs");
    g_config.ul_shifted_coeffs = program.get<float>("--ul-shifted-coeffs");

    g_config.num_build_loops = program.get<uint32_t>("--num-build-loops");
    g_config.num_triu_iters = program.get<uint32_t>("--num-triu-iters");
    g_config.prefill_ratio = program.get<float>("--prefill-ratio");
    g_config.num_routing_loops = program.get<uint32_t>("--num-routing-loops");
    g_config.routing_topk = program.get<uint32_t>("--routing-topk");
    g_config.routing_queue_size = program.get<uint32_t>("--routing-queue-size");

    g_config.mis_radix = program.get<float>("--mis-radix");
    g_config.mis_max_power = program.get<uint32_t>("--mis-max-power");
    g_config.rnet_beta = program.get<float>("--beta");

    g_config.topk = program.get<uint32_t>("--topk");

    std::string queue_config_str = program.get<std::string>("--candidate-queue-config");
    {
        std::istringstream ss(queue_config_str);
        std::string token;
        std::vector<uint32_t> values;
        while (std::getline(ss, token, ',')) { values.push_back(std::stoul(token)); }
        if (values.size() != 3) {
            fprintf(stderr, "Error: --candidate-queue-config must have 3 values: start,end,step\n");
            return 1;
        }
        g_config.queue_start = values[0];
        g_config.queue_end = values[1];
        g_config.queue_step = values[2];
    }

    g_config.bl_extracted_nbr_size = program.get<uint32_t>("--bl-extracted-nbr-size");
    g_config.ul_extracted_nbr_size = program.get<uint32_t>("--ul-extracted-nbr-size");
    g_config.warmup_runs = program.get<uint32_t>("--warmup-runs");
    g_config.test_runs = program.get<uint32_t>("--test-runs");
    g_config.verbose = program.get<bool>("--verbose");

    // Print configuration
    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "\n--- Layer Configuration ---" << std::endl;
    std::cout << "Bottom layer max nbr size: " << g_config.bl_max_nbr_size << std::endl;
    std::cout << "Upper layer max nbr size: " << g_config.ul_max_nbr_size << std::endl;
    std::cout << "\n--- Pruning Configuration ---" << std::endl;
    std::cout << "Bottom: scale=" << g_config.bl_scale_coeffs << ", shifted=" << g_config.bl_shifted_coeffs << std::endl;
    std::cout << "Upper:  scale=" << g_config.ul_scale_coeffs << ", shifted=" << g_config.ul_shifted_coeffs << std::endl;
    std::cout << "\n--- Propagate Configuration ---" << std::endl;
    std::cout << "Build loops: " << g_config.num_build_loops << std::endl;
    std::cout << "Triangle updater iterations: " << g_config.num_triu_iters << std::endl;
    std::cout << "Prefill ratio: " << g_config.prefill_ratio << std::endl;
    std::cout << "Routing loops: " << g_config.num_routing_loops << std::endl;
    std::cout << "Routing top-k: " << g_config.routing_topk << std::endl;
    std::cout << "Routing queue size: " << g_config.routing_queue_size << std::endl;
    std::cout << "\n--- R-Net Configuration ---" << std::endl;
    std::cout << "MIS radix: " << g_config.mis_radix << std::endl;
    std::cout << "MIS max power: " << g_config.mis_max_power << std::endl;
    std::cout << "R-net beta: " << g_config.rnet_beta << std::endl;
    std::cout << "\n--- Router Configuration ---" << std::endl;
    std::cout << "Top-k: " << g_config.topk << std::endl;
    std::cout << "Candidate queue config: " << g_config.queue_start << "," << g_config.queue_end << "," << g_config.queue_step << std::endl;
    std::cout << "BL extracted nbr size: " << g_config.bl_extracted_nbr_size << std::endl;
    std::cout << "UL extracted nbr size: " << g_config.ul_extracted_nbr_size << std::endl;
    std::cout << "Warmup runs: " << g_config.warmup_runs << std::endl;
    std::cout << "Test runs: " << g_config.test_runs << std::endl;
    std::cout << "==========================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    // Print summary
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "                        ARTEA HIERARCHICAL GRAPH TEST SUMMARY" << std::endl;
    std::cout << std::string(100, '=') << std::endl;

    std::cout << fmt::format("  Graph Construction:     {:.2f} ms", g_results.total_construction_time_ms) << std::endl;
    std::cout << fmt::format("  Total Layers:           {}", g_results.num_layers) << std::endl;
    std::cout << fmt::format("  Base Vectors:           {}", g_results.total_base_vecs) << std::endl;

    if (!g_results.layer_metrics.empty()) {
        std::cout << "\n--- Layer-by-Layer Analysis ---" << std::endl;
        std::cout << fmt::format("{:<8} {:<15} {:<15} {:<12}",
            "Layer", "Vertices", "Edges", "Ratio (%)") << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        for (const auto& m : g_results.layer_metrics) {
            std::cout << fmt::format("{:<8} {:<15} {:<15} {:<12.2f}",
                m.layer_id, m.num_vertices, m.num_edges, m.layer_ratio) << std::endl;
        }
    }

    if (!g_results.query_results.empty()) {
        std::cout << "\n--- Grid Search Results ---" << std::endl;
        std::cout << fmt::format("{:<15} {:<12} {:<12} {:<12}",
            "CandidateQueue", "Recall@k", "QPS", "AvgTime(ms)") << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        for (const auto& r : g_results.query_results) {
            std::cout << fmt::format("{:<15} {:<12.4f} {:<12.2f} {:<12.2f}",
                r.candidate_queue_size, r.recall, r.throughput_qps, r.query_time_ms) << std::endl;
        }
    }

    std::cout << std::string(100, '=') << std::endl;

    return result;
}
