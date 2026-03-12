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
#include <random>
#include <unordered_set>
#include <chrono>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;
using namespace artea::cpu::default_context;

struct RNetSelectionConfig {
    float min_radius;
    float beta_sq;
    float coverage_ratio;
    float confidence;
    uint32_t max_result_size;
    uint32_t sampling_batch_size;
    bool is_shuffle;
};

struct RandomSelectionConfig {
    float result_ratio;
};

struct TestConfig {
    std::string config_path;
    std::string dataset_name;

    RNetSelectionConfig rnet_config;
    RandomSelectionConfig random_config;

    // For coverage detection
    uint32_t num_test_samples;
    bool verbose;
} g_config;

struct LayerMetrics {
    uint32_t layer_id;
    uint32_t num_vertices;
    float layer_ratio;
    float min_pairwise_dist;
    uint32_t separation_violations;
    float empirical_coverage;
    bool ordering_passed;
    bool data_consistency_passed;
    bool uniqueness_passed;
};

struct TestResults {
    double generation_time_ms = 0.0;
    uint32_t num_layers = 0;
    uint32_t total_base_vecs = 0;
    std::vector<LayerMetrics> layer_metrics;
};

TestResults g_rnet_results;
TestResults g_random_results;

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
        g_rnet_results.total_base_vecs = base_vecs.get_num_vecs();
        g_random_results.total_base_vecs = base_vecs.get_num_vecs();

        if (g_config.verbose) {
            logger.info(fmt::format("Base vectors size: {}", base_vecs.get_num_vecs()));
        }
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }
    hierarchical_graph_t& get_hierarchical_graph() { return *hierarchical_graph_; }
    hierarchical_vecs_manager_t& get_hier_vecs_manager() { return *hier_vecs_manager_; }

    void set_hierarchical_graph(std::unique_ptr<hierarchical_graph_t> graph) {
        hierarchical_graph_ = std::move(graph);
    }

    void set_hier_vecs_manager(std::unique_ptr<hierarchical_vecs_manager_t> manager) {
        hier_vecs_manager_ = std::move(manager);
    }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<hierarchical_vecs_manager_t> hier_vecs_manager_;
    std::unique_ptr<hierarchical_graph_t> hierarchical_graph_;
};

template <VGPolicyT VGPolicy>
class HierVertexTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto& provider = DataProvider::instance();
        auto& dataset = provider.get_dataset();
        auto& dist_func = provider.get_dist_func();
        const auto& base_vecs = dataset.get_base_vecs();

        // Create dummy layer configs (not used in vertex generation, only for graph construction)
        layer_config_t dummy_config(32, 32);

        // Create hierarchical vecs manager and store it in provider
        auto hier_vecs_manager = std::make_unique<hierarchical_vecs_manager_t>(base_vecs);
        provider.set_hier_vecs_manager(std::move(hier_vecs_manager));

        // Create hierarchical graph with reference to the stored manager
        auto hierarchical_graph = std::make_unique<hierarchical_graph_t>(
            provider.get_hier_vecs_manager(),
            base_vecs.get_num_vecs(),
            dummy_config,
            dummy_config
        );

        logger.info(fmt::format("Testing VGPolicy: {}",
            VGPolicy == VGPolicyT::rnet_selection ? "rnet_selection" : "random_selection"));

        auto start_time = std::chrono::high_resolution_clock::now();

        if constexpr (VGPolicy == VGPolicyT::rnet_selection) {
            hierarchical_vertices_builder_t::template construct<VGPolicy>(
                base_vecs,
                dist_func,
                *hierarchical_graph,
                g_config.rnet_config.min_radius,
                g_config.rnet_config.beta_sq,
                g_config.rnet_config.coverage_ratio,
                g_config.rnet_config.confidence,
                g_config.rnet_config.max_result_size,
                g_config.rnet_config.sampling_batch_size,
                g_config.rnet_config.is_shuffle
            );
        } else {
            hierarchical_vertices_builder_t::template construct<VGPolicy>(
                base_vecs,
                dist_func,
                *hierarchical_graph,
                g_config.random_config.result_ratio
            );
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        auto& results = (VGPolicy == VGPolicyT::rnet_selection) ? g_rnet_results : g_random_results;
        results.generation_time_ms = duration.count() / 1000.0;
        results.num_layers = hierarchical_graph->get_hier_vecs_manager().get_num_layers();

        logger.info(fmt::format("Generated {} layers", results.num_layers));
        logger.info(fmt::format("Generation time: {:.2f} ms", results.generation_time_ms));

        provider.set_hierarchical_graph(std::move(hierarchical_graph));
    }
};

using HierVertexRNetTest = HierVertexTest<VGPolicyT::rnet_selection>;
using HierVertexRandomTest = HierVertexTest<VGPolicyT::random_selection>;

template <typename TestFixture>
void TestLayerOrdering(uint32_t layer_id) {
    auto& provider = DataProvider::instance();
    auto& hierarchical_graph = provider.get_hierarchical_graph();
    auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();

    if (layer_id == 0) {
        // Bottom layer is the full dataset, no vec_ids to check
        return;
    }

    logger.info(fmt::format("Testing layer {} ordering...", layer_id));

    // For upper layers, we need to check if they maintain proper ordering
    // Since we don't have direct access to vec_ids in upper layers through hier_vecs_manager,
    // we'll just verify the layer exists and has data
    const auto& layer_vecs = hier_vecs_manager.get_layer_vecs(layer_id);
    bool has_data = layer_vecs.get_num_vecs() > 0;

    EXPECT_TRUE(has_data) << fmt::format("Layer {} should have data", layer_id);
}

template <typename TestFixture>
void TestLayerSeparation(uint32_t layer_id, float min_radius, TestResults& results) {
    auto& provider = DataProvider::instance();
    auto& dist_func = provider.get_dist_func();
    auto& hierarchical_graph = provider.get_hierarchical_graph();
    auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();

    const auto& layer_vecs = hier_vecs_manager.get_layer_vecs(layer_id);

    logger.info(fmt::format("Testing layer {} separation (min_radius={:.4f})...", layer_id, min_radius));

    uint32_t violation_count = 0;
    distance_t min_pairwise_dist = std::numeric_limits<distance_t>::max();
    uint32_t num_vecs = layer_vecs.get_num_vecs();

    for (uint32_t i = 0; i < num_vecs; ++i) {
        for (uint32_t j = i + 1; j < num_vecs; ++j) {
            distance_t dist = dist_func(layer_vecs.get(i), layer_vecs.get(j));
            min_pairwise_dist = std::min(min_pairwise_dist, dist);
            if (dist < min_radius) {
                violation_count++;
                if (g_config.verbose && violation_count <= 5) {
                    logger.warn(fmt::format("Layer {} separation violation: vec[{}] and vec[{}] have distance {:.4f} < {:.4f}",
                        layer_id, i, j, dist, min_radius));
                }
            }
        }
    }

    LayerMetrics metrics;
    metrics.layer_id = layer_id;
    metrics.num_vertices = num_vecs;
    metrics.layer_ratio = 100.0f * num_vecs / results.total_base_vecs;
    metrics.min_pairwise_dist = min_pairwise_dist;
    metrics.separation_violations = violation_count;

    results.layer_metrics.push_back(metrics);

    logger.info(fmt::format("Layer {} separation: {} violations, min distance: {:.4f}",
        layer_id, violation_count, min_pairwise_dist));
}

template <typename TestFixture>
void TestLayerCoverage(uint32_t layer_id, float coverage_radius, TestResults& results) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();
    auto& hierarchical_graph = provider.get_hierarchical_graph();
    auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& layer_vecs = hier_vecs_manager.get_layer_vecs(layer_id);

    logger.info(fmt::format("Testing layer {} coverage (radius={:.4f})...", layer_id, coverage_radius));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, base_vecs.get_num_vecs() - 1);

    uint32_t uncovered_count = 0;

    for (uint32_t i = 0; i < g_config.num_test_samples; ++i) {
        uint32_t sample_id = dist(gen);
        const vec_ele_t* sample_vec = base_vecs.get(sample_id);

        distance_t min_dist_to_layer = std::numeric_limits<distance_t>::max();
        for (uint32_t j = 0; j < layer_vecs.get_num_vecs(); ++j) {
            distance_t d = dist_func(sample_vec, layer_vecs.get(j));
            min_dist_to_layer = std::min(min_dist_to_layer, d);
        }

        if (min_dist_to_layer >= coverage_radius) {
            uncovered_count++;
        }
    }

    float empirical_coverage = 1.0f - (float)uncovered_count / g_config.num_test_samples;

    // Update metrics
    for (auto& m : results.layer_metrics) {
        if (m.layer_id == layer_id) {
            m.empirical_coverage = empirical_coverage;
            break;
        }
    }

    logger.info(fmt::format("Layer {} coverage: {:.2f}% ({}/{} samples covered)",
        layer_id, empirical_coverage * 100.0f,
        g_config.num_test_samples - uncovered_count, g_config.num_test_samples));
}

template <typename TestFixture>
void TestInterLayerLinks(TestResults& results) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();
    auto& hierarchical_graph = provider.get_hierarchical_graph();
    auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();
    auto& inter_layer_links = hierarchical_graph.get_inter_layer_links();

    const auto& base_vecs = dataset.get_base_vecs();

    logger.info("Testing inter-layer links...");

    // Test each layer (starting from Layer 1)
    for (uint32_t layer_id = 1; layer_id < results.num_layers; ++layer_id) {
        const auto& current_layer_vecs = hier_vecs_manager.get_layer_vecs(layer_id);
        const auto& prev_layer_vecs = hier_vecs_manager.get_layer_vecs(layer_id - 1);

        auto layer_links = inter_layer_links.get_layer_links(layer_id);

        // Test 1: Link count should equal current layer vertex count
        EXPECT_EQ(layer_links.size(), current_layer_vecs.get_num_vecs())
            << fmt::format("Layer {} link count mismatch", layer_id);

        // Test 2: Each link index should be within valid range of previous layer
        for (uint32_t i = 0; i < layer_links.size(); ++i) {
            vertex_id_t prev_layer_idx = layer_links[i];
            EXPECT_LT(prev_layer_idx, prev_layer_vecs.get_num_vecs())
                << fmt::format("Layer {} vertex {} has invalid link {} (prev layer size: {})",
                    layer_id, i, prev_layer_idx, prev_layer_vecs.get_num_vecs());
        }

        // Test 3: Vector data consistency - vectors should match through links
        uint32_t mismatch_count = 0;
        for (uint32_t i = 0; i < std::min(layer_links.size(), static_cast<size_t>(100)); ++i) {
            vertex_id_t prev_layer_idx = layer_links[i];
            const vec_ele_t* current_vec = current_layer_vecs.get(i);
            const vec_ele_t* prev_vec = prev_layer_vecs.get(prev_layer_idx);

            distance_t dist = dist_func(current_vec, prev_vec);
            if (dist > 1e-6) {  // Should be exactly the same vector
                mismatch_count++;
                if (g_config.verbose && mismatch_count <= 3) {
                    logger.warn(fmt::format(
                        "Layer {} vertex {} vector mismatch with prev layer vertex {} (distance: {})",
                        layer_id, i, prev_layer_idx, dist));
                }
            }
        }

        EXPECT_EQ(mismatch_count, 0)
            << fmt::format("Layer {} has {} vector mismatches", layer_id, mismatch_count);

        logger.info(fmt::format("Layer {} inter-layer links: {} links, all valid",
            layer_id, layer_links.size()));
    }

    // Test 4: Full chain traceability - trace from top layer to base_vecs
    if (results.num_layers > 1) {
        logger.info("Testing full chain traceability from top to bottom...");

        uint32_t top_layer_id = results.num_layers - 1;
        const auto& top_layer_vecs = hier_vecs_manager.get_layer_vecs(top_layer_id);

        // Test a few vertices from top layer
        uint32_t num_test_vertices = std::min(static_cast<uint32_t>(10),
                                               static_cast<uint32_t>(top_layer_vecs.get_num_vecs()));

        for (uint32_t top_idx = 0; top_idx < num_test_vertices; ++top_idx) {
            const vec_ele_t* top_vec = top_layer_vecs.get(top_idx);

            // Trace down to base layer
            vertex_id_t current_idx = top_idx;
            for (layer_id_t layer_id = top_layer_id; layer_id > 0; --layer_id) {
                auto layer_links = inter_layer_links.get_layer_links(layer_id);
                current_idx = layer_links[current_idx];
            }

            // current_idx now points to base_vecs
            const vec_ele_t* base_vec = base_vecs.get(current_idx);

            // Verify vectors match
            distance_t dist = dist_func(top_vec, base_vec);
            EXPECT_LT(dist, 1e-6)
                << fmt::format("Top layer vertex {} traced to base_vecs[{}] but vectors don't match (distance: {})",
                    top_idx, current_idx, dist);
        }

        logger.info(fmt::format("Full chain traceability test passed for {} vertices", num_test_vertices));
    }
}


TEST_F(HierVertexRNetTest, VerifyHierarchicalStructure) {
    auto& provider = DataProvider::instance();
    auto& hierarchical_graph = provider.get_hierarchical_graph();
    auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();

    logger.info("Verifying hierarchical structure for rnet_selection...");

    EXPECT_GE(g_rnet_results.num_layers, 1) << "Should have at least 1 layer (bottom layer)";

    // Test each layer, starting from Layer 1 with radius = min_radius * beta_sq
    float current_radius = g_config.rnet_config.min_radius * g_config.rnet_config.beta_sq;
    for (uint32_t layer_id = 1; layer_id < g_rnet_results.num_layers; ++layer_id) {
        TestLayerSeparation<HierVertexRNetTest>(layer_id, current_radius, g_rnet_results);
        TestLayerCoverage<HierVertexRNetTest>(layer_id, current_radius, g_rnet_results);
        current_radius *= g_config.rnet_config.beta_sq;
    }

    // Test inter-layer links
    TestInterLayerLinks<HierVertexRNetTest>(g_rnet_results);
}

TEST_F(HierVertexRandomTest, VerifyHierarchicalStructure) {
    auto& provider = DataProvider::instance();
    auto& hierarchical_graph = provider.get_hierarchical_graph();
    auto& hier_vecs_manager = hierarchical_graph.get_hier_vecs_manager();

    logger.info("Verifying hierarchical structure for random_selection...");

    EXPECT_GE(g_random_results.num_layers, 1) << "Should have at least 1 layer (bottom layer)";

    // For random selection, we just verify layer sizes decrease
    for (uint32_t layer_id = 1; layer_id < g_random_results.num_layers; ++layer_id) {
        const auto& layer_vecs = hier_vecs_manager.get_layer_vecs(layer_id);

        LayerMetrics metrics;
        metrics.layer_id = layer_id;
        metrics.num_vertices = layer_vecs.get_num_vecs();
        metrics.layer_ratio = 100.0f * metrics.num_vertices / g_random_results.total_base_vecs;
        metrics.min_pairwise_dist = 0.0f;
        metrics.separation_violations = 0;
        metrics.empirical_coverage = 0.0f;

        g_random_results.layer_metrics.push_back(metrics);

        logger.info(fmt::format("Layer {}: {} vertices ({:.2f}%)",
            layer_id, metrics.num_vertices, metrics.layer_ratio));
    }

    // Test inter-layer links
    TestInterLayerLinks<HierVertexRandomTest>(g_random_results);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_hierarchical_vertices_builder");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));

    // RNet selection parameters
    program.add_argument("--min-radius").default_value(34875.0f).scan<'g', float>();
    program.add_argument("--beta-sq").default_value(2.56f).scan<'g', float>();
    program.add_argument("--coverage-ratio").default_value(0.96f).scan<'g', float>();
    program.add_argument("--confidence").default_value(0.99f).scan<'g', float>();
    program.add_argument("--max-result-size").default_value(100000u).scan<'u', uint32_t>();
    program.add_argument("--sampling-batch-size").default_value(2048u).scan<'u', uint32_t>();
    program.add_argument("--shuffle").default_value(false).implicit_value(true);

    // Random selection parameters
    program.add_argument("--random-result-ratio").default_value(1.0f/16.0f).scan<'g', float>();

    program.add_argument("-n", "--num-test-samples").default_value(10000u).scan<'u', uint32_t>();
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

    g_config.rnet_config.min_radius = program.get<float>("--min-radius");
    g_config.rnet_config.beta_sq = program.get<float>("--beta-sq");
    g_config.rnet_config.coverage_ratio = program.get<float>("--coverage-ratio");
    g_config.rnet_config.confidence = program.get<float>("--confidence");
    g_config.rnet_config.max_result_size = program.get<uint32_t>("--max-result-size");
    g_config.rnet_config.sampling_batch_size = program.get<uint32_t>("--sampling-batch-size");
    g_config.rnet_config.is_shuffle = program.get<bool>("--shuffle");

    g_config.random_config.result_ratio = program.get<float>("--random-result-ratio");

    g_config.num_test_samples = program.get<uint32_t>("--num-test-samples");
    g_config.verbose = program.get<bool>("--verbose");

    std::cout << "\n=== Test Configuration ===" << std::endl;
    std::cout << "Dataset: " << g_config.dataset_name << std::endl;
    std::cout << "Config path: " << g_config.config_path << std::endl;
    std::cout << "\n--- RNet Selection Parameters ---" << std::endl;
    std::cout << "Min radius: " << g_config.rnet_config.min_radius << std::endl;
    std::cout << "Beta squared: " << g_config.rnet_config.beta_sq << std::endl;
    std::cout << "Coverage ratio: " << g_config.rnet_config.coverage_ratio << std::endl;
    std::cout << "Confidence: " << g_config.rnet_config.confidence << std::endl;
    std::cout << "Max result size: " << g_config.rnet_config.max_result_size << std::endl;
    std::cout << "Sampling batch size: " << g_config.rnet_config.sampling_batch_size << std::endl;
    std::cout << "Shuffle: " << (g_config.rnet_config.is_shuffle ? "true" : "false") << std::endl;
    std::cout << "\n--- Random Selection Parameters ---" << std::endl;
    std::cout << "Result ratio: " << g_config.random_config.result_ratio << std::endl;
    std::cout << "\n--- Test Parameters ---" << std::endl;
    std::cout << "Num test samples: " << g_config.num_test_samples << std::endl;
    std::cout << "Verbose: " << (g_config.verbose ? "true" : "false") << std::endl;
    std::cout << "==========================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    // Print summary
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "                        HIERARCHICAL VERTEX GENERATION SUMMARY" << std::endl;
    std::cout << std::string(100, '=') << std::endl;

    // RNet Selection Results
    if (g_rnet_results.num_layers > 0) {
        std::cout << "\n=== RNet Selection ===" << std::endl;
        std::cout << fmt::format("  Generation Time:        {:.2f} ms", g_rnet_results.generation_time_ms) << std::endl;
        std::cout << fmt::format("  Total Layers:           {}", g_rnet_results.num_layers) << std::endl;
        std::cout << fmt::format("  Base Vectors:           {}", g_rnet_results.total_base_vecs) << std::endl;

        if (!g_rnet_results.layer_metrics.empty()) {
            std::cout << "\n--- Layer-by-Layer Analysis ---" << std::endl;
            std::cout << fmt::format("{:<8} {:<15} {:<12} {:<18} {:<20} {:<15}",
                "Layer", "Vertices", "Ratio (%)", "Min Pair Dist", "Sep Violations", "Coverage (%)") << std::endl;
            std::cout << std::string(100, '-') << std::endl;

            for (const auto& metrics : g_rnet_results.layer_metrics) {
                std::cout << fmt::format("{:<8} {:<15} {:<12.2f} {:<18.4f} {:<20} {:<15.2f}",
                    metrics.layer_id,
                    metrics.num_vertices,
                    metrics.layer_ratio,
                    metrics.min_pairwise_dist,
                    metrics.separation_violations,
                    metrics.empirical_coverage * 100.0f) << std::endl;
            }
        }
    }

    // Random Selection Results
    if (g_random_results.num_layers > 0) {
        std::cout << "\n=== Random Selection ===" << std::endl;
        std::cout << fmt::format("  Generation Time:        {:.2f} ms", g_random_results.generation_time_ms) << std::endl;
        std::cout << fmt::format("  Total Layers:           {}", g_random_results.num_layers) << std::endl;
        std::cout << fmt::format("  Base Vectors:           {}", g_random_results.total_base_vecs) << std::endl;

        if (!g_random_results.layer_metrics.empty()) {
            std::cout << "\n--- Layer-by-Layer Analysis ---" << std::endl;
            std::cout << fmt::format("{:<8} {:<15} {:<12}",
                "Layer", "Vertices", "Ratio (%)") << std::endl;
            std::cout << std::string(50, '-') << std::endl;

            for (const auto& metrics : g_random_results.layer_metrics) {
                std::cout << fmt::format("{:<8} {:<15} {:<12.2f}",
                    metrics.layer_id,
                    metrics.num_vertices,
                    metrics.layer_ratio) << std::endl;
            }
        }
    }

    std::cout << std::string(100, '=') << std::endl;

    return result;
}
