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
#include <cmath>
#include <chrono>
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
    uint32_t max_nbr_size;
    float prefill_ratio;
    uint32_t num_samples;
    bool verbose;
} g_config;

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

        // Build KNN graph
        const auto& base_vecs = dataset_->get_base_vecs();
        layer_config_t layer_config(g_config.max_nbr_size, static_cast<uint32_t>(g_config.max_nbr_size * 1.5));
        knn_graph::pruning_config_t pruning_config(1.0f, 0.0f);
        knn_graph::propagate_config_t propagate_config(5, 12, g_config.prefill_ratio, 1);

        ARTEA_INFO("Building KNN graph...");
        auto t0 = std::chrono::high_resolution_clock::now();
        knn_graph_ = std::make_unique<knn_graph::index_t>(knn_graph::factory_t::construct_graph(
            base_vecs, layer_config, pruning_config, propagate_config
        ));
        auto t1 = std::chrono::high_resolution_clock::now();
        double build_time_s = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        ARTEA_INFO(fmt::format("KNN graph built with {} vertices in {:.2f} s", knn_graph_->get_num_vertices(), build_time_s));
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }
    knn_graph::index_t& get_knn_graph() { return *knn_graph_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
    std::unique_ptr<knn_graph::index_t> knn_graph_;
};

class RadiusProberTest : public ::testing::Test {};

/**
 * @brief Compare RadiusProber (graph-based) vs BruteforceRouter (exact) on sampled vertices.
 *
 * Validation logic:
 *   1. RadiusProber reads nbrs[0].distance from the approximate KNN graph for every vertex.
 *   2. BruteforceRouter computes the exact NN distance for a uniform sample of vertices.
 *   3. For each sample, compute relative error = |graph_nn - bf_nn| / bf_nn.
 *      - Duplicate vectors (bf_nn == 0) are detected and skipped (verified by recomputing distance).
 *      - Exact matches (rel_err == 0) are counted separately as "match ratio".
 *   4. Report match ratio and average relative error. No hard threshold — the graph is approximate,
 *      so error depends on dataset dimensionality and build parameters.
 */
TEST_F(RadiusProberTest, CompareWithBruteforce) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();
    auto& knn_graph = provider.get_knn_graph();

    const auto& base_vecs = dataset.get_base_vecs();
    const vertex_num_t num_vertices = base_vecs.get_num_vecs();

    // --- Step 1: Run RadiusProber on the KNN graph ---
    // Reads nbrs[0].distance for all vertices, returns the requested quantile.
    radius_prober_t prober;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto probe_result = prober.probe(knn_graph, 1, 0.0f);
    auto t1 = std::chrono::high_resolution_clock::now();
    double probe_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    ARTEA_INFO(fmt::format("RadiusProber min NN distance: {:.6f} (probe time: {:.2f} ms)", probe_result.radius, probe_time_ms));

    // --- Step 2: Compute exact NN distances via BruteforceRouter ---
    // topk=3: need extra slots because bruteforce includes self (distance=0),
    // and there may also be a duplicate vector (distance=0) occupying another slot.
    const uint32_t num_samples = std::min(g_config.num_samples, static_cast<uint32_t>(num_vertices));
    bruteforce_router_t bf_router(base_vecs, dist_func, 3);
    bf_router.initialize();

    // Uniformly sample vertices with stride = num_vertices / num_samples.
    std::vector<distance_t> bf_nn_distances(num_samples);
    std::vector<vertex_id_t> results_cache(num_samples);
    uint32_t stride = num_vertices / num_samples;
    for (uint32_t i = 0; i < num_samples; ++i) {
        vertex_id_t vid = i * stride;
        const auto* query_vec = base_vecs.get(vid);
        auto results = bf_router.query(query_vec);
        // Skip self (bruteforce returns the query vertex itself with distance=0).
        for (const auto& entry : results) {
            if (entry.get_base_id() != vid) {
                bf_nn_distances[i] = entry.get_distance();
                results_cache[i] = entry.get_base_id();
                break;
            }
        }
    }

    // --- Step 3: Per-sample comparison ---
    double total_relative_error = 0.0;
    uint32_t valid_count = 0;
    uint32_t exact_match_count = 0;
    uint32_t duplicate_vec_count = 0;
    for (uint32_t i = 0; i < num_samples; ++i) {
        vertex_id_t vid = i * stride;
        // Graph NN: the closest neighbor found by the approximate KNN graph.
        distance_t graph_nn_dist = knn_graph.fetch_nbrs(vid)[0].get_distance();
        // Bruteforce NN: the exact closest non-self neighbor.
        distance_t bf_nn_dist = bf_nn_distances[i];

        // Detect duplicate vectors in the dataset (exact NN distance is 0).
        // Verify by recomputing distance between the two different vertex IDs.
        if (bf_nn_dist == 0.0f) {
            vertex_id_t bf_nn_id = results_cache[i];
            distance_t recomputed = dist_func(base_vecs.get(vid), base_vecs.get(bf_nn_id));
            ARTEA_INFO(fmt::format("  [duplicate] vertex {}: bf_nn_id={}, recomputed_dist={:.6f}",
                vid, bf_nn_id, recomputed));
            duplicate_vec_count++;
            continue;
        }

        // Relative error: how far the approximate graph NN is from the exact NN.
        // Always >= 0 since graph_nn_dist >= bf_nn_dist (approximate can't beat exact).
        double rel_err = std::abs(static_cast<double>(graph_nn_dist) - static_cast<double>(bf_nn_dist))
                       / static_cast<double>(bf_nn_dist);
        total_relative_error += rel_err;
        valid_count++;

        if (rel_err == 0.0) { exact_match_count++; }

        if (g_config.verbose && i < 10) {
            ARTEA_INFO(fmt::format("  vertex {}: graph_nn={:.6f}, bf_nn={:.6f}, rel_err={:.4f}%",
                vid, graph_nn_dist, bf_nn_dist, rel_err * 100.0));
        }
    }

    // --- Step 4: Summary statistics ---
    double avg_relative_error = (valid_count > 0) ? total_relative_error / valid_count : 0.0;
    double match_ratio = (valid_count > 0) ? static_cast<double>(exact_match_count) / valid_count : 0.0;

    ARTEA_INFO(fmt::format("Samples: {}, non-duplicate: {}, exact_match: {}, duplicate_vecs: {}",
        num_samples, valid_count, exact_match_count, duplicate_vec_count));
    ARTEA_INFO(fmt::format("Match ratio: {:.2f}% ({}/{})", match_ratio * 100.0, exact_match_count, valid_count));
    ARTEA_INFO(fmt::format("Average relative error over {} samples: {:.4f}%", valid_count, avg_relative_error * 100.0));

    EXPECT_GT(valid_count, 0u) << "Should have at least some valid (non-duplicate) samples";
}

TEST_F(RadiusProberTest, QuantileOrdering) {
    auto& provider = DataProvider::instance();
    auto& knn_graph = provider.get_knn_graph();

    radius_prober_t prober;

    std::vector<float> quantiles = {0.0f, 0.01f, 0.05f, 0.1f, 0.5f, 0.9f};
    std::vector<distance_t> radii;

    ARTEA_INFO("Quantile results (nbr_rank=1, nearest neighbor):");
    for (float q : quantiles) {
        auto result = prober.probe(knn_graph, 1, q);
        radii.push_back(result.radius);
        ARTEA_INFO(fmt::format("  quantile={:.2f}: radius={:.6f}", q, result.radius));
    }

    // Verify monotonically non-decreasing
    for (size_t i = 1; i < radii.size(); ++i) {
        EXPECT_LE(radii[i - 1], radii[i]);
    }
}

/**
 * @brief Probe distance quantiles at nbr_rank=16 (the 16th nearest neighbor).
 */
TEST_F(RadiusProberTest, Rank16QuantileOrdering) {
    auto& provider = DataProvider::instance();
    auto& knn_graph = provider.get_knn_graph();

    radius_prober_t prober;

    const vertex_num_t nbr_rank = 16;
    std::vector<float> quantiles = {0.0f, 0.01f, 0.05f, 0.1f, 0.5f, 0.9f};
    std::vector<distance_t> radii;

    ARTEA_INFO(fmt::format("Quantile results (nbr_rank={}):", nbr_rank));
    for (float q : quantiles) {
        auto result = prober.probe(knn_graph, nbr_rank, q);
        radii.push_back(result.radius);
        ARTEA_INFO(fmt::format("  quantile={:.2f}: radius={:.6f} (valid_vertices={})",
            q, result.radius, result.num_vertices));
    }

    // Verify monotonically non-decreasing
    for (size_t i = 1; i < radii.size(); ++i) {
        EXPECT_LE(radii[i - 1], radii[i]);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_radius_prober");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--max-nbr-size").default_value(96u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio").default_value(0.34f).scan<'g', float>();
    program.add_argument("--num-samples").default_value(100u).scan<'u', uint32_t>();
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
    g_config.prefill_ratio = program.get<float>("--prefill-ratio");
    g_config.num_samples = program.get<uint32_t>("--num-samples");
    g_config.verbose = program.get<bool>("--verbose");

    DataProvider::instance().init();

    return RUN_ALL_TESTS();
}
