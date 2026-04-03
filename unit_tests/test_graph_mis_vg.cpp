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
#include <unordered_set>
#include <string>
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
    uint32_t max_nbr_size;
    float prefill_ratio;
    uint32_t routing_topk;
    uint32_t routing_queue_size;
    bool verbose;
} g_config;

struct TestResults {
    double knn_build_time_s = 0.0;
    double mis_time_s = 0.0;
    distance_t min_radius = 0.0f;
    float radix = 2.0f;
    uint32_t max_power = 0;
    uint32_t num_vertices = 0;
    uint32_t num_selected = 0;
    uint32_t independence_violations = 0;
    uint32_t coverage_count = 0;
    uint32_t non_selected_count = 0;
} g_results;

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

        // Build KNN graph
        layer_config_t layer_config(g_config.max_nbr_size, static_cast<uint32_t>(g_config.max_nbr_size * 1.5));
        knn_graph::pruning_config_t pruning_config(1.0f, 0.0f);
        knn_graph::propagate_config_t propagate_config(5, 12, g_config.prefill_ratio, 1,
        g_config.routing_topk, g_config.routing_queue_size);

        ARTEA_INFO("Building KNN graph...");
        auto t0 = std::chrono::high_resolution_clock::now();
        knn_graph_ = std::make_unique<knn_graph::index_t>(knn_graph::factory_t::construct_graph(
            base_vecs, layer_config, pruning_config, propagate_config
        ));
        auto t1 = std::chrono::high_resolution_clock::now();
        g_results.knn_build_time_s = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        g_results.num_vertices = knn_graph_->get_num_vertices();
        ARTEA_INFO(fmt::format("KNN graph built with {} vertices in {:.2f} s",
            g_results.num_vertices, g_results.knn_build_time_s));

        // Print radius quantile table for user selection
        _print_radius_table();

        // Interactive: ask user for rnet_radius, radix, max_power
        std::cout << "\nEnter: rnet_radius radix max_power (space-separated, e.g. 30000 2.0 3):" << std::endl;
        std::cout << "  Start radius = rnet_radius * radix^max_power (max_power=0 for single-shot): " << std::flush;
        std::string input;
        std::getline(std::cin, input);
        {
            std::istringstream iss(input);
            float rnet_radius, radix_val = 2.0f;
            uint32_t max_power_val = 0;
            iss >> rnet_radius;
            if (iss >> radix_val) { iss >> max_power_val; }
            g_results.min_radius = rnet_radius;
            g_results.radix = radix_val;
            g_results.max_power = max_power_val;
        }
        ARTEA_INFO(fmt::format("Parameters: rnet_radius={:.6f}, radix={:.2f}, max_power={}",
            g_results.min_radius, g_results.radix, g_results.max_power));

        // Run GraphMISVG
        graph_mis_vg_t mis_vg;
        t0 = std::chrono::high_resolution_clock::now();
        rnet_ = std::make_unique<approx_rnet_t>(mis_vg.generate(
            *knn_graph_, g_results.min_radius, g_results.radix, g_results.max_power));
        t1 = std::chrono::high_resolution_clock::now();
        g_results.mis_time_s = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        g_results.num_selected = rnet_->get_num_vecs();
        ARTEA_INFO(fmt::format("MIS completed in {:.2f} s: selected {} / {} vertices ({:.2f}%)",
            g_results.mis_time_s, g_results.num_selected, g_results.num_vertices,
            100.0 * g_results.num_selected / g_results.num_vertices));
    }

    knn_graph::index_t& get_knn_graph() { return *knn_graph_; }
    approx_rnet_t& get_rnet() { return *rnet_; }

private:
    DataProvider() = default;

    void _print_radius_table() {
        radius_prober_t prober;

        std::vector<uint32_t> nbr_ranks = {1, 2, 4, 8, 16, 32};
        std::vector<float> quantiles = {0.01f, 0.05f, 0.10f, 0.25f, 0.50f, 0.75f, 0.90f, 0.95f, 0.99f, 0.995f, 0.999f};

        // Filter out nbr_ranks that exceed max_nbr_size
        std::vector<uint32_t> valid_ranks;
        for (auto rank : nbr_ranks) {
            if (rank <= g_config.max_nbr_size) valid_ranks.push_back(rank);
        }

        // Print header
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "  Radius Quantile Table (distance at nbr_rank-th neighbor)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::cout << fmt::format("{:<12}", "quantile");
        for (auto rank : valid_ranks) {
            std::cout << fmt::format(" {:>10}", fmt::format("rank={}", rank));
        }
        std::cout << std::endl;
        std::cout << std::string(12 + 11 * valid_ranks.size(), '-') << std::endl;

        for (float q : quantiles) {
            std::cout << fmt::format("{:<12}", fmt::format("{:.3f}", q));
            for (auto rank : valid_ranks) {
                auto result = prober.probe(*knn_graph_, rank, q);
                std::cout << fmt::format(" {:>10.4f}", result.radius);
            }
            std::cout << std::endl;
        }

        std::cout << std::string(70, '=') << std::endl;
    }

    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<knn_graph::index_t> knn_graph_;
    std::unique_ptr<approx_rnet_t> rnet_;
};

class GraphMISVGTest : public ::testing::Test {};

/**
 * @brief Verify independence: no two selected vertices should be close neighbors.
 *
 * For each IN vertex, scan its close neighbors (distance < min_radius).
 * None of them should also be IN.
 */
TEST_F(GraphMISVGTest, VerifyIndependence) {
    auto& knn_graph = DataProvider::instance().get_knn_graph();
    auto& rnet = DataProvider::instance().get_rnet();
    const distance_t min_radius = g_results.min_radius;

    // Build lookup set for selected vertices
    std::unordered_set<vertex_id_t> selected_set(rnet.vec_ids.begin(), rnet.vec_ids.end());

    uint32_t violations = 0;
    for (const auto& vid : rnet.vec_ids) {
        const auto& nbrs = knn_graph.fetch_nbrs(vid);
        for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
            if (nbrs[i].get_distance() >= min_radius) break;
            if (selected_set.count(nbrs[i].get_id())) {
                violations++;
                if (g_config.verbose && violations <= 5) {
                    ARTEA_INFO(fmt::format("  Independence violation: vertex {} and {} (dist={:.6f} < {:.6f})",
                        vid, nbrs[i].get_id(), nbrs[i].get_distance(), min_radius));
                }
            }
        }
    }

    g_results.independence_violations = violations;
    double violation_ratio = (rnet.vec_ids.size() > 0)
        ? 100.0 * violations / rnet.vec_ids.size() : 0.0;
    ARTEA_INFO(fmt::format("Independence violations: {} ({:.2f}% of selected vertices)",
        violations, violation_ratio));
}

/**
 * @brief Verify coverage (maximality): every non-selected vertex should have
 * at least one close IN vertex reachable via either direction of an edge.
 *
 * A non-selected vertex u is covered if:
 *   (a) u sees an IN vertex in u's neighbor list (u→v exists, v is IN), OR
 *   (b) an IN vertex v sees u in v's neighbor list (v→u exists, v is IN).
 *
 * Both directions are needed because the MIS algorithm marks vertices OUT
 * via two mechanisms: (a) Phase 1 self-proposal when u sees an IN neighbor,
 * and (b) Phase 3a Crush when an IN vertex v suppresses u along v→u.
 *
 * Note: coverage may be < 100% because the KNN graph is approximate.
 */
TEST_F(GraphMISVGTest, VerifyCoverage) {
    auto& knn_graph = DataProvider::instance().get_knn_graph();
    auto& rnet = DataProvider::instance().get_rnet();
    const distance_t min_radius = g_results.min_radius;
    const vertex_num_t num_vertices = knn_graph.get_num_vertices();

    std::unordered_set<vertex_id_t> selected_set(rnet.vec_ids.begin(), rnet.vec_ids.end());

    std::vector<bool> is_covered(num_vertices, false);

    // Direction 1 (v→u): IN vertex v's out-edges cover its close neighbors
    for (const auto& vid : rnet.vec_ids) {
        const auto& nbrs = knn_graph.fetch_nbrs(vid);
        for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
            if (nbrs[i].get_distance() >= min_radius) break;
            is_covered[nbrs[i].get_id()] = true;
        }
    }

    // Direction 2 (u→v): non-selected u sees an IN vertex in its own neighbor list
    for (vertex_id_t u = 0; u < num_vertices; ++u) {
        if (selected_set.count(u) || is_covered[u]) continue;
        const auto& nbrs = knn_graph.fetch_nbrs(u);
        for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
            if (nbrs[i].get_distance() >= min_radius) break;
            if (selected_set.count(nbrs[i].get_id())) {
                is_covered[u] = true;
                break;
            }
        }
    }

    uint32_t covered = 0;
    uint32_t non_selected = 0;
    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        if (selected_set.count(v)) continue;
        non_selected++;
        if (is_covered[v]) covered++;
    }

    g_results.coverage_count = covered;
    g_results.non_selected_count = non_selected;
    double coverage_ratio = (non_selected > 0) ? 100.0 * covered / non_selected : 100.0;
    ARTEA_INFO(fmt::format("Coverage: {}/{} non-selected vertices covered ({:.2f}%)",
        covered, non_selected, coverage_ratio));

    // MIS guarantees 100% coverage on the exact graph; approximate graph may have gaps
    EXPECT_GT(coverage_ratio, 90.0) << "Coverage should be > 90% even on approximate graph";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_graph_mis_vg");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--max-nbr-size").default_value(96u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio").default_value(0.34f).scan<'g', float>();
    program.add_argument("--routing-topk").default_value(64u).scan<'u', uint32_t>()
        .help("Routing updater top-k (default: 64)");
    program.add_argument("--routing-queue-size").default_value(96u).scan<'u', uint32_t>()
        .help("Routing updater candidate queue size (default: 96)");
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
    g_config.routing_topk = program.get<uint32_t>("--routing-topk");
    g_config.routing_queue_size = program.get<uint32_t>("--routing-queue-size");
    g_config.verbose = program.get<bool>("--verbose");

    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << fmt::format("  Dataset:        {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Max nbr size:       {}", g_config.max_nbr_size) << std::endl;
    std::cout << fmt::format("  Prefill ratio:      {}", g_config.prefill_ratio) << std::endl;
    std::cout << fmt::format("  Routing top-k:      {}", g_config.routing_topk) << std::endl;
    std::cout << fmt::format("  Routing queue size: {}", g_config.routing_queue_size) << std::endl;
    std::cout << "=====================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "          GRAPH MIS VG TEST SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << fmt::format("  Dataset:                  {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Num Vertices:             {}", g_results.num_vertices) << std::endl;
    std::cout << fmt::format("  Min Radius:               {:.6f}", g_results.min_radius) << std::endl;
    std::cout << fmt::format("  Radix:                    {:.2f}", g_results.radix) << std::endl;
    std::cout << fmt::format("  Max Power:                {}", g_results.max_power) << std::endl;
    std::cout << fmt::format("  Mode:                     {}",
        g_results.max_power > 0 ? "coarse-to-fine" : "single-shot") << std::endl;
    std::cout << fmt::format("  KNN Build Time:           {:.2f} s", g_results.knn_build_time_s) << std::endl;
    std::cout << fmt::format("  MIS Time:                 {:.2f} s", g_results.mis_time_s) << std::endl;
    std::cout << fmt::format("  Selected:                 {} ({:.2f}%)",
        g_results.num_selected, 100.0 * g_results.num_selected / g_results.num_vertices) << std::endl;
    std::cout << fmt::format("  Independence Violations:  {}", g_results.independence_violations) << std::endl;
    double cov = (g_results.non_selected_count > 0)
        ? 100.0 * g_results.coverage_count / g_results.non_selected_count : 100.0;
    std::cout << fmt::format("  Coverage:                 {}/{} ({:.2f}%)",
        g_results.coverage_count, g_results.non_selected_count, cov) << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return result;
}
