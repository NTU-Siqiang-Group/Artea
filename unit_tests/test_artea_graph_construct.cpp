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

struct VerticesBuilderConfigParams {
    float beta;
    float coverage_ratio;
    float confidence;
    float max_result_ratio;
    uint32_t sampling_batch_size;
};

struct LayerConfigParams {
    uint32_t max_nbr_size;
    uint32_t reserved_nbr_size;
};

struct PruningConfigParams {
    float scale_coeffs;
    float shifted_coeffs;
};

struct PropagateConfigParams {
    uint32_t num_build_loops;
    uint32_t num_triu_iters;
    float prefill_ratio;
    uint32_t num_routing_loops;
};

struct TestConfig {
    std::string config_path;
    std::string dataset_name;

    VerticesBuilderConfigParams vertices_config;
    LayerConfigParams bottom_layer_config;
    LayerConfigParams upper_layer_config;
    PruningConfigParams bottom_pruning_config;
    PruningConfigParams upper_pruning_config;
    PropagateConfigParams propagate_config;

    uint32_t topk;
    uint32_t queue_start;
    uint32_t queue_end;
    uint32_t queue_step;
    uint32_t bl_extracted_nbr_size;
    uint32_t ul_extracted_nbr_size;

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
    double construction_time_ms;
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

        dist_func_ = std::make_unique<dist_func_t>(dataset_->get_base_vecs().get_vec_dim());

        const auto& base_vecs = dataset_->get_base_vecs();
        g_results.total_base_vecs = base_vecs.get_num_vecs();

        // Probe min_radius from dataset
        ARTEA_INFO("Probing min_radius from dataset...");
        radius_prober_t prober(*dist_func_);
        auto probe_result = prober.probe(base_vecs, 0.001f, 0.95f, 0.05f);
        _min_radius = probe_result.radius;
        ARTEA_INFO(fmt::format("Probed min_radius: {:.6f}", _min_radius));

        if (g_config.verbose) {
            ARTEA_INFO(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
        }
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }
    float get_min_radius() const { return _min_radius; }
    artea_graph_index_t& get_hierarchical_graph() { return *hierarchical_graph_; }
    hierarchical_vecs_manager_t& get_hier_vecs_manager() { return hierarchical_graph_->get_hier_vecs_manager(); }

    void set_hierarchical_graph(
        std::unique_ptr<artea_graph_index_t> graph
    ) {
        hierarchical_graph_ = std::move(graph);
    }

    bool has_hierarchical_graph() const {
        return hierarchical_graph_ != nullptr;
    }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<artea_graph_index_t> hierarchical_graph_;
    float _min_radius = 0.0f;
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
        const auto& base_vecs = dataset.get_base_vecs();

        ARTEA_INFO("Starting Artea graph construction test with conv_graph_descent policy");

        // Create layer configs
        layer_config_t bottom_layer_config(
            g_config.bottom_layer_config.max_nbr_size,
            g_config.bottom_layer_config.reserved_nbr_size
        );
        layer_config_t upper_layer_config(
            g_config.upper_layer_config.max_nbr_size,
            g_config.upper_layer_config.reserved_nbr_size
        );

        // Create pruning configs (per layer)
        artea_graph::pruning_config_t bottom_pruning_config(
            g_config.bottom_pruning_config.scale_coeffs,
            g_config.bottom_pruning_config.shifted_coeffs
        );
        artea_graph::pruning_config_t upper_pruning_config(
            g_config.upper_pruning_config.scale_coeffs,
            g_config.upper_pruning_config.shifted_coeffs
        );

        // Create propagate config (shared between layers)
        artea_graph::propagate_config_t propagate_config(
            g_config.propagate_config.num_build_loops,
            g_config.propagate_config.num_triu_iters,
            g_config.propagate_config.prefill_ratio,
            g_config.propagate_config.num_routing_loops
        );

        // Create vertices builder config
        greedy_vertices_builder_config_t vertices_builder_config(
            provider.get_min_radius(),
            g_config.vertices_config.beta,
            g_config.vertices_config.coverage_ratio,
            g_config.vertices_config.confidence,
            g_config.vertices_config.max_result_ratio,
            g_config.vertices_config.sampling_batch_size
        );

        // Construct hierarchical graph via factory
        ARTEA_INFO("Constructing hierarchical Artea graph...");
        auto construction_start = std::chrono::high_resolution_clock::now();

        auto graph = artea_graph_factory_t::construct_graph(
            base_vecs,
            bottom_layer_config,
            upper_layer_config,
            bottom_pruning_config,
            upper_pruning_config,
            propagate_config,
            vertices_builder_config
        );
        auto hierarchical_graph = std::make_unique<artea_graph_index_t>(std::move(graph));

        auto construction_end = std::chrono::high_resolution_clock::now();
        auto construction_duration = std::chrono::duration_cast<std::chrono::microseconds>(construction_end - construction_start);
        g_results.total_construction_time_ms = construction_duration.count() / 1000.0;
        g_results.num_layers = hierarchical_graph->get_num_layers();

        ARTEA_INFO(fmt::format("Graph construction completed: {} layers in {:.2f} ms",
            g_results.num_layers, g_results.total_construction_time_ms));

        // Output layer vertices information
        ARTEA_INFO("Layer vertices information:");
        for (uint32_t layer_id = 0; layer_id < g_results.num_layers; ++layer_id) {
            const auto& layer_vecs = hierarchical_graph->get_hier_vecs_manager().get_layer_vecs(layer_id);
            uint32_t num_vertices = layer_vecs.get_num_vecs();
            float layer_ratio = 100.0f * num_vertices / g_results.total_base_vecs;
            ARTEA_INFO(fmt::format("  Layer {}: {} vertices ({:.2f}%)",
                layer_id, num_vertices, layer_ratio));
        }

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
                ARTEA_INFO(fmt::format("Layer {}: {} vertices, {} edges ({:.2f}%)",
                    layer_id, metrics.num_vertices, metrics.num_edges, metrics.layer_ratio));
            }
        }

        provider.set_hierarchical_graph(std::move(hierarchical_graph));
    }
};

TEST_F(ArteaGraphConstructTest, VerifyGraphConstruction) {
    auto& provider = DataProvider::instance();
    auto& hierarchical_graph = provider.get_hierarchical_graph();

    ARTEA_INFO("Verifying graph construction...");

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

    ARTEA_INFO("Graph construction verification passed");
}

TEST_F(ArteaGraphConstructTest, VerifyGraphConnectivity) {
    auto& provider = DataProvider::instance();
    auto& hierarchical_graph = provider.get_hierarchical_graph();

    ARTEA_INFO("Verifying graph connectivity...");

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
                        ARTEA_WARN(fmt::format("Layer {} vertex {} has no neighbors", layer_id, v));
                    }
                }
            }

            // Allow a small number of isolated vertices in upper layers
            float isolation_ratio = static_cast<float>(isolated_vertices) / metrics.num_vertices;

            ARTEA_INFO(fmt::format("Layer {} connectivity: {}/{} isolated vertices ({:.2f}%)",
                layer_id, isolated_vertices, metrics.num_vertices, isolation_ratio * 100.0f));

            EXPECT_LT(isolation_ratio, 0.1f)
                << fmt::format("Layer {} has too many isolated vertices: {}/{} ({:.2f}%)",
                    layer_id, isolated_vertices, metrics.num_vertices, isolation_ratio * 100.0f);
        }
    }

    ARTEA_INFO("Graph connectivity verification passed");
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

    ARTEA_INFO("Testing hierarchical graph router query recall...");

    // Convert hierarchical_graph to hierarchical_search_graph
    ARTEA_INFO("Converting to hierarchical search graph...");
    auto start_time = std::chrono::high_resolution_clock::now();

    auto hierarchical_search_graph = search_graph_converter_t::from_hierarchical_graph(
        hierarchical_graph,
        g_config.bl_extracted_nbr_size,
        g_config.ul_extracted_nbr_size
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    double conversion_time_ms = duration.count() / 1000.0;
    ARTEA_INFO(fmt::format("Conversion time: {:.2f} ms", conversion_time_ms));

    // Run grid search over candidate queue sizes
    ARTEA_INFO(fmt::format("\nRunning Grid Search: candidate queue size {} to {}, step {}",
        g_config.queue_start, g_config.queue_end, g_config.queue_step));
    ARTEA_INFO("");

    recall_estimator_t recall_estimator;

    for (uint32_t queue_size = g_config.queue_start; queue_size <= g_config.queue_end; queue_size += g_config.queue_step) {
        // Create hierarchical router with current queue size
        hierarchical_graph_router_t<graph_mode_t::search_mode> router(
            base_vecs,
            dist_func,
            hierarchical_search_graph,
            g_config.topk,
            queue_size
        );
        router.initialize();

        // Query all vectors
        start_time = std::chrono::high_resolution_clock::now();
        knn_results_t results = router.batch_query(query_vecs);
        end_time = std::chrono::high_resolution_clock::now();

        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // Compute metrics
        QueryResult result;
        result.candidate_queue_size = queue_size;
        result.query_time_ms = duration.count() / 1000.0;
        result.avg_query_time_us = static_cast<double>(duration.count()) / query_vecs.get_num_vecs();
        result.throughput_qps = query_vecs.get_num_vecs() * 1000000.0 / duration.count();
        result.recall = recall_estimator.calculate_recall_at_k(results, groundtruth, g_config.topk, query_vecs.get_num_vecs());

        g_results.query_results.push_back(result);

        ARTEA_INFO(fmt::format("CandidateQueue={:3}: Recall@{}={:.4f}, QPS={:8.2f}, AvgTime={:.2f}ms",
            queue_size, g_config.topk, result.recall, result.throughput_qps, result.query_time_ms));
    }

    // Expect reasonable recall for at least one configuration
    bool has_good_recall = false;
    for (const auto& result : g_results.query_results) {
        if (result.recall >= 0.5f) {
            has_good_recall = true;
            break;
        }
    }
    EXPECT_TRUE(has_good_recall) << "At least one configuration should achieve recall >= 50%";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_artea_graph");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));

    // Vertices builder parameters
    program.add_argument("--beta").default_value(1.44f).scan<'g', float>();
    program.add_argument("--coverage-ratio").default_value(0.999f).scan<'g', float>();
    program.add_argument("--confidence").default_value(0.950f).scan<'g', float>();
    program.add_argument("--max-result-ratio").default_value(0.2f).scan<'g', float>();
    program.add_argument("--sampling-batch-size").default_value(2048u).scan<'u', uint32_t>();

    // Bottom layer config
    program.add_argument("--bl-max-nbr-size").default_value(96u).scan<'u', uint32_t>();
    // Upper layer config
    program.add_argument("--ul-max-nbr-size").default_value(96u).scan<'u', uint32_t>();

    // Bottom layer pruning config
    program.add_argument("--bl-scale-coeffs").default_value(1.10f).scan<'g', float>();
    program.add_argument("--bl-shifted-coeffs").default_value(0.10f).scan<'g', float>();

    // Upper layer pruning config
    program.add_argument("--ul-scale-coeffs").default_value(1.10f).scan<'g', float>();
    program.add_argument("--ul-shifted-coeffs").default_value(0.10f).scan<'g', float>();

    // Propagate config (shared between layers)
    program.add_argument("--num-build-loops").default_value(5u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters").default_value(12u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio").default_value(0.34f).scan<'g', float>();
    program.add_argument("--num-routing-loops").default_value(1u).scan<'u', uint32_t>();

    // Router parameters
    program.add_argument("-k", "--topk").default_value(20u).scan<'u', uint32_t>();
    program.add_argument("--candidate-queue-config")
        .default_value(std::string("40,100,20"))
        .help("Candidate queue size grid search: start,end,step (default: 40,100,20)");
    program.add_argument("--bl-extracted-nbr-size").default_value(64u).scan<'u', uint32_t>()
        .help("Bottom layer extracted neighbor size (defaults to 64)");
    program.add_argument("--ul-extracted-nbr-size").default_value(32u).scan<'u', uint32_t>()
        .help("Upper layer extracted neighbor size (defaults to 32)");

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

    g_config.vertices_config.beta = program.get<float>("--beta");
    g_config.vertices_config.coverage_ratio = program.get<float>("--coverage-ratio");
    g_config.vertices_config.confidence = program.get<float>("--confidence");
    g_config.vertices_config.max_result_ratio = program.get<float>("--max-result-ratio");
    g_config.vertices_config.sampling_batch_size = program.get<uint32_t>("--sampling-batch-size");

    g_config.bottom_layer_config.max_nbr_size = program.get<uint32_t>("--bl-max-nbr-size");
    g_config.bottom_layer_config.reserved_nbr_size =
        static_cast<uint32_t>(g_config.bottom_layer_config.max_nbr_size * 1.5);

    g_config.upper_layer_config.max_nbr_size = program.get<uint32_t>("--ul-max-nbr-size");
    g_config.upper_layer_config.reserved_nbr_size =
        static_cast<uint32_t>(g_config.upper_layer_config.max_nbr_size * 1.5);

    g_config.bottom_pruning_config.scale_coeffs = program.get<float>("--bl-scale-coeffs");
    g_config.bottom_pruning_config.shifted_coeffs = program.get<float>("--bl-shifted-coeffs");

    g_config.upper_pruning_config.scale_coeffs = program.get<float>("--ul-scale-coeffs");
    g_config.upper_pruning_config.shifted_coeffs = program.get<float>("--ul-shifted-coeffs");

    g_config.propagate_config.num_build_loops = program.get<uint32_t>("--num-build-loops");
    g_config.propagate_config.num_triu_iters = program.get<uint32_t>("--num-triu-iters");
    g_config.propagate_config.prefill_ratio = program.get<float>("--prefill-ratio");
    g_config.propagate_config.num_routing_loops = program.get<uint32_t>("--num-routing-loops");

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

    g_config.bl_extracted_nbr_size = program.get<uint32_t>("--bl-extracted-nbr-size");
    g_config.ul_extracted_nbr_size = program.get<uint32_t>("--ul-extracted-nbr-size");

    g_config.verbose = program.get<bool>("--verbose");

    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Config path: " << g_config.config_path << std::endl;
    std::cout << "\n--- Vertices Builder Parameters ---" << std::endl;
    std::cout << "Beta: " << g_config.vertices_config.beta << std::endl;
    std::cout << "Coverage ratio: " << g_config.vertices_config.coverage_ratio << std::endl;
    std::cout << "Confidence: " << g_config.vertices_config.confidence << std::endl;
    std::cout << "Max result ratio: " << g_config.vertices_config.max_result_ratio << std::endl;
    std::cout << "Sampling batch size: " << g_config.vertices_config.sampling_batch_size << std::endl;
    std::cout << "\n--- Layer Configuration ---" << std::endl;
    std::cout << "Bottom layer max nbr size: " << g_config.bottom_layer_config.max_nbr_size << std::endl;
    std::cout << "Upper layer max nbr size: " << g_config.upper_layer_config.max_nbr_size << std::endl;
    std::cout << "\n--- Pruning Configuration ---" << std::endl;
    std::cout << "Bottom layer scale coeffs: " << g_config.bottom_pruning_config.scale_coeffs << std::endl;
    std::cout << "Bottom layer shifted coeffs: " << g_config.bottom_pruning_config.shifted_coeffs << std::endl;
    std::cout << "Upper layer scale coeffs: " << g_config.upper_pruning_config.scale_coeffs << std::endl;
    std::cout << "Upper layer shifted coeffs: " << g_config.upper_pruning_config.shifted_coeffs << std::endl;
    std::cout << "\n--- Propagate Configuration ---" << std::endl;
    std::cout << "Build loops: " << g_config.propagate_config.num_build_loops << std::endl;
    std::cout << "Triangle updater iterations: " << g_config.propagate_config.num_triu_iters << std::endl;
    std::cout << "Prefill ratio: " << g_config.propagate_config.prefill_ratio << std::endl;
    std::cout << "Routing loops: " << g_config.propagate_config.num_routing_loops << std::endl;
    std::cout << "\n--- Router Configuration ---" << std::endl;
    std::cout << "Top-k: " << g_config.topk << std::endl;
    std::cout << "Candidate queue config: " << g_config.queue_start << "," << g_config.queue_end << "," << g_config.queue_step << std::endl;
    std::cout << "Bottom layer extracted nbr size: " << g_config.bl_extracted_nbr_size << std::endl;
    std::cout << "Upper layer extracted nbr size: " << g_config.ul_extracted_nbr_size << std::endl;
    std::cout << "Verbose: " << (g_config.verbose ? "true" : "false") << std::endl;
    std::cout << "==========================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    // Print summary
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "                        ARTEA HIERARCHICAL GRAPH TEST SUMMARY" << std::endl;
    std::cout << std::string(100, '=') << std::endl;

    std::cout << "\n=== Construction Timing ===" << std::endl;
    std::cout << fmt::format("  Graph Construction:     {:.2f} ms", g_results.total_construction_time_ms) << std::endl;

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
    std::cout << fmt::format("  Top-k:                  {}", g_config.topk) << std::endl;
    std::cout << "\n--- Grid Search Results ---" << std::endl;
    std::cout << fmt::format("{:<15} {:<12} {:<12} {:<12}",
        "CandidateQueue", "Recall@k", "QPS", "AvgTime(ms)") << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (const auto& result : g_results.query_results) {
        std::cout << fmt::format("{:<15} {:<12.4f} {:<12.2f} {:<12.2f}",
            result.candidate_queue_size,
            result.recall,
            result.throughput_qps,
            result.query_time_ms) << std::endl;
    }

    std::cout << std::string(100, '=') << std::endl;

    return result;
}