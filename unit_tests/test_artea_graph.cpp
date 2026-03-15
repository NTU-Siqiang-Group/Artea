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

struct VerticesBuilderConfigParams {
    float min_radius;
    float beta_sq;
    float coverage_ratio;
    float confidence;
    float max_result_ratio;
    uint32_t sampling_batch_size;
    bool is_shuffle;
};

struct LayerConfigParams {
    uint32_t max_nbr_size;
    uint32_t reserved_nbr_size;
};

struct EdgesBuilderConfigParams {
    float scale_coeffs;
    float shifted_coeffs;
    uint32_t num_outer_iters;
    uint32_t num_inner_iters;
};

struct TestConfig {
    std::string config_path;
    std::string dataset_name;

    VerticesBuilderConfigParams vertices_config;
    LayerConfigParams bottom_layer_config;
    LayerConfigParams upper_layer_config;
    EdgesBuilderConfigParams bottom_edges_config;
    EdgesBuilderConfigParams upper_edges_config;

    uint32_t topk;
    uint32_t ul_candidate_queue_size;
    uint32_t bl_candidate_queue_size;

    bool verbose;
} g_config;

struct LayerMetrics {
    uint32_t layer_id;
    uint32_t num_vertices;
    uint32_t num_edges;
    float layer_ratio;
    double construction_time_ms;
};

struct TestResults {
    double vertices_construction_time_ms = 0.0;
    double edges_construction_time_ms = 0.0;
    double total_construction_time_ms = 0.0;
    double query_time_ms = 0.0;
    double avg_query_time_us = 0.0;
    double throughput_qps = 0.0;
    float recall = 0.0f;
    uint32_t num_layers = 0;
    uint32_t total_base_vecs = 0;
    uint32_t num_queries = 0;
    std::vector<LayerMetrics> layer_metrics;
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
        logger.info(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset_ = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dist_func_ = std::make_unique<dist_func_t>(dataset_->get_base_vecs().get_vec_dim());

        const auto& base_vecs = dataset_->get_base_vecs();
        g_results.total_base_vecs = base_vecs.get_num_vecs();

        if (g_config.verbose) {
            logger.info(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
        }
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }
    hierarchical_graph_t& get_hierarchical_graph() { return *hierarchical_graph_; }
    hierarchical_vecs_manager_t& get_hier_vecs_manager() { return *hier_vecs_manager_; }

    void set_hierarchical_graph(
        std::unique_ptr<hierarchical_vecs_manager_t> vecs_manager,
        std::unique_ptr<hierarchical_graph_t> graph
    ) {
        hier_vecs_manager_ = std::move(vecs_manager);
        hierarchical_graph_ = std::move(graph);
    }

    bool has_hierarchical_graph() const {
        return hierarchical_graph_ != nullptr;
    }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<hierarchical_vecs_manager_t> hier_vecs_manager_;
    std::unique_ptr<hierarchical_graph_t> hierarchical_graph_;
};

class ArteaGraphConstructTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto& provider = DataProvider::instance();

        // Only construct the graph once
        if (provider.has_hierarchical_graph()) {
            return;
        }

        auto& dataset = provider.get_dataset();
        auto& dist_func = provider.get_dist_func();
        const auto& base_vecs = dataset.get_base_vecs();

        logger.info("Starting Artea graph construction test with conv_graph_descent policy");

        // Create layer configs
        layer_config_t bottom_layer_config(
            g_config.bottom_layer_config.max_nbr_size,
            g_config.bottom_layer_config.reserved_nbr_size
        );
        layer_config_t upper_layer_config(
            g_config.upper_layer_config.max_nbr_size,
            g_config.upper_layer_config.reserved_nbr_size
        );

        // Create edges builder configs
        edges_builder_config_t bottom_edges_config(
            g_config.bottom_edges_config.scale_coeffs,
            g_config.bottom_edges_config.shifted_coeffs,
            g_config.bottom_edges_config.num_outer_iters,
            g_config.bottom_edges_config.num_inner_iters
        );
        edges_builder_config_t upper_edges_config(
            g_config.upper_edges_config.scale_coeffs,
            g_config.upper_edges_config.shifted_coeffs,
            g_config.upper_edges_config.num_outer_iters,
            g_config.upper_edges_config.num_inner_iters
        );

        // Create vertices builder config
        greedy_vertices_builder_config_t vertices_builder_config(
            g_config.vertices_config.min_radius,
            g_config.vertices_config.beta_sq,
            g_config.vertices_config.coverage_ratio,
            g_config.vertices_config.confidence,
            g_config.vertices_config.max_result_ratio,
            g_config.vertices_config.sampling_batch_size,
            g_config.vertices_config.is_shuffle
        );

        // Create hierarchical vecs manager
        auto hier_vecs_manager = std::make_unique<hierarchical_vecs_manager_t>(base_vecs);

        // Create hierarchical graph
        auto hierarchical_graph = std::make_unique<hierarchical_graph_t>(
            *hier_vecs_manager,
            bottom_layer_config,
            upper_layer_config,
            bottom_edges_config,
            upper_edges_config,
            vertices_builder_config
        );

        // hier_vecs_manager is already referenced by hierarchical_graph, no need to transfer ownership

        // Step 1: Construct vertices
        logger.info("Step 1: Constructing hierarchical vertices...");
        auto vertices_start = std::chrono::high_resolution_clock::now();

        hierarchical_vertices_builder_t::template construct<VGPolicyT::rnet_selection>(
            dist_func,
            *hierarchical_graph,
            vertices_builder_config
        );

        auto vertices_end = std::chrono::high_resolution_clock::now();
        auto vertices_duration = std::chrono::duration_cast<std::chrono::microseconds>(vertices_end - vertices_start);
        g_results.vertices_construction_time_ms = vertices_duration.count() / 1000.0;
        g_results.num_layers = hierarchical_graph->get_num_layers();

        logger.info(fmt::format("Vertices construction completed: {} layers in {:.2f} ms",
            g_results.num_layers, g_results.vertices_construction_time_ms));

        // Output layer vertices information
        logger.info("Layer vertices information:");
        for (uint32_t layer_id = 0; layer_id < g_results.num_layers; ++layer_id) {
            const auto& layer_vecs = hierarchical_graph->get_hier_vecs_manager().get_layer_vecs(layer_id);
            uint32_t num_vertices = layer_vecs.get_num_vecs();
            float layer_ratio = 100.0f * num_vertices / g_results.total_base_vecs;
            logger.info(fmt::format("  Layer {}: {} vertices ({:.2f}%)",
                layer_id, num_vertices, layer_ratio));
        }

        // Step 2: Construct edges
        logger.info("Step 2: Constructing hierarchical edges...");
        auto edges_start = std::chrono::high_resolution_clock::now();

        hierarchical_edges_builder_t::template construct<EGPolicyT::conv_graph_descent>(
            dist_func,
            *hierarchical_graph
        );

        auto edges_end = std::chrono::high_resolution_clock::now();
        auto edges_duration = std::chrono::duration_cast<std::chrono::microseconds>(edges_end - edges_start);
        g_results.edges_construction_time_ms = edges_duration.count() / 1000.0;

        logger.info(fmt::format("Edges construction completed in {:.2f} ms",
            g_results.edges_construction_time_ms));

        g_results.total_construction_time_ms = g_results.vertices_construction_time_ms +
                                                g_results.edges_construction_time_ms;

        // Collect layer metrics
        for (uint32_t layer_id = 0; layer_id < g_results.num_layers; ++layer_id) {
            const auto& layer_vecs = hierarchical_graph->get_hier_vecs_manager().get_layer_vecs(layer_id);
            const auto& layer_graph = hierarchical_graph->get_layer_graph(layer_id);

            LayerMetrics metrics;
            metrics.layer_id = layer_id;
            metrics.num_vertices = layer_vecs.get_num_vecs();

            // Calculate total edges by summing neighbor counts
            uint32_t total_edges = 0;
            const auto& nbrs_arr = layer_graph.get_nbrs_arr();
            for (const auto& nbrs : nbrs_arr) {
                total_edges += nbrs.size();
            }
            metrics.num_edges = total_edges;

            metrics.layer_ratio = 100.0f * metrics.num_vertices / g_results.total_base_vecs;
            metrics.construction_time_ms = 0.0;  // Individual layer timing not tracked

            g_results.layer_metrics.push_back(metrics);

            if (g_config.verbose) {
                logger.info(fmt::format("Layer {}: {} vertices, {} edges ({:.2f}%)",
                    layer_id, metrics.num_vertices, metrics.num_edges, metrics.layer_ratio));
            }
        }

        provider.set_hierarchical_graph(std::move(hier_vecs_manager), std::move(hierarchical_graph));
    }
};

TEST_F(ArteaGraphConstructTest, VerifyGraphConstruction) {
    auto& provider = DataProvider::instance();
    auto& hierarchical_graph = provider.get_hierarchical_graph();

    logger.info("Verifying graph construction...");

    // Test 1: Should have at least 1 layer
    EXPECT_GE(g_results.num_layers, 1) << "Should have at least 1 layer (bottom layer)";

    // Test 2: Each layer should have valid vertices and edges
    for (uint32_t layer_id = 0; layer_id < g_results.num_layers; ++layer_id) {
        const auto& metrics = g_results.layer_metrics[layer_id];

        EXPECT_GT(metrics.num_vertices, 0)
            << fmt::format("Layer {} should have vertices", layer_id);

        EXPECT_GT(metrics.num_edges, 0)
            << fmt::format("Layer {} should have edges", layer_id);

        // Verify graph structure
        const auto& layer_graph = hierarchical_graph.get_layer_graph(layer_id);
        EXPECT_EQ(layer_graph.get_num_vertices(), metrics.num_vertices)
            << fmt::format("Layer {} graph vertex count mismatch", layer_id);
    }

    // Test 3: Upper layers should have fewer vertices than lower layers
    for (uint32_t layer_id = 1; layer_id < g_results.num_layers; ++layer_id) {
        EXPECT_LT(g_results.layer_metrics[layer_id].num_vertices,
                  g_results.layer_metrics[layer_id - 1].num_vertices)
            << fmt::format("Layer {} should have fewer vertices than layer {}",
                layer_id, layer_id - 1);
    }

    // Test 4: Construction time should be reasonable
    EXPECT_GT(g_results.total_construction_time_ms, 0.0)
        << "Total construction time should be positive";

    logger.info("Graph construction verification passed");
}

TEST_F(ArteaGraphConstructTest, VerifyGraphConnectivity) {
    auto& provider = DataProvider::instance();
    auto& hierarchical_graph = provider.get_hierarchical_graph();

    logger.info("Verifying graph connectivity...");

    // Test each layer's graph connectivity
    for (uint32_t layer_id = 0; layer_id < g_results.num_layers; ++layer_id) {
        const auto& layer_graph = hierarchical_graph.get_layer_graph(layer_id);
        const auto& metrics = g_results.layer_metrics[layer_id];

        // Check that each vertex has at least one neighbor (except for very small graphs)
        if (metrics.num_vertices > 1) {
            uint32_t isolated_vertices = 0;
            const auto& nbrs_arr = layer_graph.get_nbrs_arr();
            for (uint32_t v = 0; v < metrics.num_vertices; ++v) {
                const auto& neighbors = nbrs_arr[v];
                if (neighbors.empty()) {
                    isolated_vertices++;
                    if (g_config.verbose && isolated_vertices <= 5) {
                        logger.warn(fmt::format("Layer {} vertex {} has no neighbors", layer_id, v));
                    }
                }
            }

            // Allow a small number of isolated vertices in upper layers
            float isolation_ratio = static_cast<float>(isolated_vertices) / metrics.num_vertices;

            logger.info(fmt::format("Layer {} connectivity: {}/{} isolated vertices ({:.2f}%)",
                layer_id, isolated_vertices, metrics.num_vertices, isolation_ratio * 100.0f));

            EXPECT_LT(isolation_ratio, 0.1f)
                << fmt::format("Layer {} has too many isolated vertices: {}/{} ({:.2f}%)",
                    layer_id, isolated_vertices, metrics.num_vertices, isolation_ratio * 100.0f);
        }
    }

    logger.info("Graph connectivity verification passed");
}

TEST_F(ArteaGraphConstructTest, QueryRecall) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();
    auto& hierarchical_graph = provider.get_hierarchical_graph();
    const auto& groundtruth = dataset.get_gt_vecs();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    g_results.num_queries = query_vecs.get_num_vecs();

    logger.info("Testing hierarchical graph router query recall...");

    // Convert hierarchical_graph to hierarchical_search_graph
    logger.info("Converting to hierarchical search graph...");
    auto start_time = std::chrono::high_resolution_clock::now();

    auto hierarchical_search_graph = search_graph_converter_t::from_hierarchical_graph(
        hierarchical_graph,
        g_config.bottom_layer_config.max_nbr_size,
        g_config.upper_layer_config.max_nbr_size
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double conversion_time_ms = duration.count() / 1000.0;
    logger.info(fmt::format("Conversion time: {:.2f} ms", conversion_time_ms));

    // Create hierarchical router
    hierarchical_graph_router_t router(
        base_vecs,
        dist_func,
        hierarchical_search_graph,
        g_config.topk,
        g_config.ul_candidate_queue_size,
        g_config.bl_candidate_queue_size
    );
    router.initialize();

    // Query all vectors
    logger.info(fmt::format("Querying {} vectors...", query_vecs.get_num_vecs()));
    start_time = std::chrono::high_resolution_clock::now();

    idlist_array_t results = router.batch_query(query_vecs);

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    g_results.query_time_ms = duration.count() / 1000.0;
    g_results.avg_query_time_us = static_cast<double>(duration.count()) / query_vecs.get_num_vecs();
    g_results.throughput_qps = query_vecs.get_num_vecs() * 1000000.0 / duration.count();

    logger.info(fmt::format("Query time: {:.2f} ms", g_results.query_time_ms));
    logger.info(fmt::format("Avg query time: {:.2f} us", g_results.avg_query_time_us));
    logger.info(fmt::format("Throughput: {:.2f} QPS", g_results.throughput_qps));

    // Compute recall
    recall_estimator_t recall_estimator(dist_func);
    auto recall_metrics = recall_estimator.calculate_recall_at_k(
        results,
        groundtruth,
        query_vecs,
        base_vecs
    );
    g_results.recall = recall_metrics.soft_recall;

    logger.info(fmt::format("Recall@{}: {:.4f} (soft: {:.4f})",
        g_config.topk, recall_metrics.strict_recall, recall_metrics.soft_recall));

    // Expect reasonable recall (at least 50%)
    EXPECT_GE(g_results.recall, 0.5f) << "Recall should be at least 50%";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_artea_graph");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));

    // Vertices builder parameters
    program.add_argument("--vb-min-radius").default_value(34875.0f).scan<'g', float>();
    program.add_argument("--vb-beta-sq").default_value(2.56f).scan<'g', float>();
    program.add_argument("--vb-coverage-ratio").default_value(0.96f).scan<'g', float>();
    program.add_argument("--vb-confidence").default_value(0.99f).scan<'g', float>();
    program.add_argument("--vb-max-result-ratio").default_value(0.2f).scan<'g', float>();
    program.add_argument("--vb-sampling-batch-size").default_value(2048u).scan<'u', uint32_t>();
    program.add_argument("--vb-shuffle").default_value(false).implicit_value(true);

    // Bottom layer config
    program.add_argument("--bl-max-nbr-size").default_value(32u).scan<'u', uint32_t>();
    program.add_argument("--bl-reserved-nbr-size").default_value(48u).scan<'u', uint32_t>();

    // Upper layer config
    program.add_argument("--ul-max-nbr-size").default_value(24u).scan<'u', uint32_t>();
    program.add_argument("--ul-reserved-nbr-size").default_value(40u).scan<'u', uint32_t>();

    // Bottom edges builder config
    program.add_argument("--bl-scale-coeffs").default_value(1.0f).scan<'g', float>();
    program.add_argument("--bl-shifted-coeffs").default_value(0.0f).scan<'g', float>();
    program.add_argument("--bl-num-outer-iters").default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--bl-num-inner-iters").default_value(14u).scan<'u', uint32_t>();

    // Upper edges builder config
    program.add_argument("--ul-scale-coeffs").default_value(1.0f).scan<'g', float>();
    program.add_argument("--ul-shifted-coeffs").default_value(0.0f).scan<'g', float>();
    program.add_argument("--ul-num-outer-iters").default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--ul-num-inner-iters").default_value(14u).scan<'u', uint32_t>();

    // Router parameters
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--ul-candidate-queue-size").default_value(8u).scan<'u', uint32_t>();
    program.add_argument("--bl-candidate-queue-size").default_value(50u).scan<'u', uint32_t>();

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

    g_config.vertices_config.min_radius = program.get<float>("--vb-min-radius");
    g_config.vertices_config.beta_sq = program.get<float>("--vb-beta-sq");
    g_config.vertices_config.coverage_ratio = program.get<float>("--vb-coverage-ratio");
    g_config.vertices_config.confidence = program.get<float>("--vb-confidence");
    g_config.vertices_config.max_result_ratio = program.get<float>("--vb-max-result-ratio");
    g_config.vertices_config.sampling_batch_size = program.get<uint32_t>("--vb-sampling-batch-size");
    g_config.vertices_config.is_shuffle = program.get<bool>("--vb-shuffle");

    g_config.bottom_layer_config.max_nbr_size = program.get<uint32_t>("--bl-max-nbr-size");
    g_config.bottom_layer_config.reserved_nbr_size = program.get<uint32_t>("--bl-reserved-nbr-size");

    g_config.upper_layer_config.max_nbr_size = program.get<uint32_t>("--ul-max-nbr-size");
    g_config.upper_layer_config.reserved_nbr_size = program.get<uint32_t>("--ul-reserved-nbr-size");

    g_config.bottom_edges_config.scale_coeffs = program.get<float>("--bl-scale-coeffs");
    g_config.bottom_edges_config.shifted_coeffs = program.get<float>("--bl-shifted-coeffs");
    g_config.bottom_edges_config.num_outer_iters = program.get<uint32_t>("--bl-num-outer-iters");
    g_config.bottom_edges_config.num_inner_iters = program.get<uint32_t>("--bl-num-inner-iters");

    g_config.upper_edges_config.scale_coeffs = program.get<float>("--ul-scale-coeffs");
    g_config.upper_edges_config.shifted_coeffs = program.get<float>("--ul-shifted-coeffs");
    g_config.upper_edges_config.num_outer_iters = program.get<uint32_t>("--ul-num-outer-iters");
    g_config.upper_edges_config.num_inner_iters = program.get<uint32_t>("--ul-num-inner-iters");

    g_config.topk = program.get<uint32_t>("--topk");
    g_config.ul_candidate_queue_size = program.get<uint32_t>("--ul-candidate-queue-size");
    g_config.bl_candidate_queue_size = program.get<uint32_t>("--bl-candidate-queue-size");

    g_config.verbose = program.get<bool>("--verbose");

    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Config path: " << g_config.config_path << std::endl;
    std::cout << "\n--- Vertices Builder Parameters ---" << std::endl;
    std::cout << "Min radius: " << g_config.vertices_config.min_radius << std::endl;
    std::cout << "Beta squared: " << g_config.vertices_config.beta_sq << std::endl;
    std::cout << "Coverage ratio: " << g_config.vertices_config.coverage_ratio << std::endl;
    std::cout << "Confidence: " << g_config.vertices_config.confidence << std::endl;
    std::cout << "Max result ratio: " << g_config.vertices_config.max_result_ratio << std::endl;
    std::cout << "Sampling batch size: " << g_config.vertices_config.sampling_batch_size << std::endl;
    std::cout << "Shuffle: " << (g_config.vertices_config.is_shuffle ? "true" : "false") << std::endl;
    std::cout << "\n--- Layer Configuration ---" << std::endl;
    std::cout << "Bottom layer max nbr size: " << g_config.bottom_layer_config.max_nbr_size << std::endl;
    std::cout << "Bottom layer reserved nbr size: " << g_config.bottom_layer_config.reserved_nbr_size << std::endl;
    std::cout << "Upper layer max nbr size: " << g_config.upper_layer_config.max_nbr_size << std::endl;
    std::cout << "Upper layer reserved nbr size: " << g_config.upper_layer_config.reserved_nbr_size << std::endl;
    std::cout << "\n--- Edges Builder Configuration ---" << std::endl;
    std::cout << "Bottom edges scale coeffs: " << g_config.bottom_edges_config.scale_coeffs << std::endl;
    std::cout << "Bottom edges shifted coeffs: " << g_config.bottom_edges_config.shifted_coeffs << std::endl;
    std::cout << "Bottom edges num outer iters: " << g_config.bottom_edges_config.num_outer_iters << std::endl;
    std::cout << "Bottom edges num inner iters: " << g_config.bottom_edges_config.num_inner_iters << std::endl;
    std::cout << "Upper edges scale coeffs: " << g_config.upper_edges_config.scale_coeffs << std::endl;
    std::cout << "Upper edges shifted coeffs: " << g_config.upper_edges_config.shifted_coeffs << std::endl;
    std::cout << "Upper edges num outer iters: " << g_config.upper_edges_config.num_outer_iters << std::endl;
    std::cout << "Upper edges num inner iters: " << g_config.upper_edges_config.num_inner_iters << std::endl;
    std::cout << "\n--- Router Configuration ---" << std::endl;
    std::cout << "Top-k: " << g_config.topk << std::endl;
    std::cout << "Upper layer candidate queue size: " << g_config.ul_candidate_queue_size << std::endl;
    std::cout << "Bottom layer candidate queue size: " << g_config.bl_candidate_queue_size << std::endl;
    std::cout << "Verbose: " << (g_config.verbose ? "true" : "false") << std::endl;
    std::cout << "==========================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    // Print summary
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "                        ARTEA HIERARCHICAL GRAPH TEST SUMMARY" << std::endl;
    std::cout << std::string(100, '=') << std::endl;

    std::cout << "\n=== Construction Timing ===" << std::endl;
    std::cout << fmt::format("  Vertices Construction:  {:.2f} ms", g_results.vertices_construction_time_ms) << std::endl;
    std::cout << fmt::format("  Edges Construction:     {:.2f} ms", g_results.edges_construction_time_ms) << std::endl;
    std::cout << fmt::format("  Total Construction:     {:.2f} ms", g_results.total_construction_time_ms) << std::endl;

    std::cout << "\n=== Graph Structure ===" << std::endl;
    std::cout << fmt::format("  Total Layers:           {}", g_results.num_layers) << std::endl;
    std::cout << fmt::format("  Base Vectors:           {}", g_results.total_base_vecs) << std::endl;

    if (!g_results.layer_metrics.empty()) {
        std::cout << "\n--- Layer-by-Layer Analysis ---" << std::endl;
        std::cout << fmt::format("{:<8} {:<15} {:<15} {:<12}",
            "Layer", "Vertices", "Edges", "Ratio (%)") << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (const auto& metrics : g_results.layer_metrics) {
            std::cout << fmt::format("{:<8} {:<15} {:<15} {:<12.2f}",
                metrics.layer_id,
                metrics.num_vertices,
                metrics.num_edges,
                metrics.layer_ratio) << std::endl;
        }
    }

    std::cout << "\n=== Query Performance ===" << std::endl;
    std::cout << fmt::format("  Num Queries:            {}", g_results.num_queries) << std::endl;
    std::cout << fmt::format("  Total Query Time:       {:.2f} ms", g_results.query_time_ms) << std::endl;
    std::cout << fmt::format("  Avg Query Time:         {:.2f} us", g_results.avg_query_time_us) << std::endl;
    std::cout << fmt::format("  Throughput:             {:.2f} QPS", g_results.throughput_qps) << std::endl;
    std::cout << fmt::format("  Recall@{}:              {:.4f}", g_config.topk, g_results.recall) << std::endl;

    std::cout << std::string(100, '=') << std::endl;

    return result;
}