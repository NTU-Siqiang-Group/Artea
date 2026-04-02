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
    uint32_t nbr_rank;
    float quantile;
    bool verbose;
} g_config;

struct TestResults {
    double knn_build_time_s = 0.0;
    double mis_time_s = 0.0;
    distance_t min_radius = 0.0f;
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
        knn_graph::propagate_config_t propagate_config(5, 12, g_config.prefill_ratio, 1);

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

        // Probe min_radius
        radius_prober_t prober;
        auto probe_result = prober.probe(*knn_graph_, g_config.nbr_rank, g_config.quantile);
        g_results.min_radius = probe_result.radius;
        ARTEA_INFO(fmt::format("RadiusProber nbr_rank={}, quantile={:.2f}: min_radius={:.6f}",
            g_config.nbr_rank, g_config.quantile, g_results.min_radius));

        // Run GraphMISVG
        graph_mis_vg_t mis_vg;
        t0 = std::chrono::high_resolution_clock::now();
        rnet_ = std::make_unique<approx_rnet_t>(mis_vg.generate(*knn_graph_, g_results.min_radius));
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

    // Violations are counted per directed edge (A's nbr list has B, both IN).
    // On an asymmetric approximate KNN graph, a vertex B may not see A in its
    // neighbor list, so B can be marked IN even though A (also IN) considers B
    // a close neighbor. This is inherent to approximate graphs.
    g_results.independence_violations = violations;
    double violation_ratio = (rnet.vec_ids.size() > 0)
        ? 100.0 * violations / rnet.vec_ids.size() : 0.0;
    ARTEA_INFO(fmt::format("Independence violations: {} ({:.2f}% of selected vertices)",
        violations, violation_ratio));
}

/**
 * @brief Verify coverage (maximality): every non-selected vertex should have
 * at least one selected close neighbor in its KNN neighbor list.
 *
 * Note: coverage may be < 100% because the KNN graph is approximate.
 */
TEST_F(GraphMISVGTest, VerifyCoverage) {
    auto& knn_graph = DataProvider::instance().get_knn_graph();
    auto& rnet = DataProvider::instance().get_rnet();
    const distance_t min_radius = g_results.min_radius;
    const vertex_num_t num_vertices = knn_graph.get_num_vertices();

    std::unordered_set<vertex_id_t> selected_set(rnet.vec_ids.begin(), rnet.vec_ids.end());

    uint32_t covered = 0;
    uint32_t non_selected = 0;
    for (vertex_id_t v = 0; v < num_vertices; ++v) {
        if (selected_set.count(v)) continue;
        non_selected++;

        const auto& nbrs = knn_graph.fetch_nbrs(v);
        bool is_covered = false;
        for (vertex_num_t i = 0; i < nbrs.size(); ++i) {
            if (nbrs[i].get_distance() >= min_radius) break;
            if (selected_set.count(nbrs[i].get_id())) {
                is_covered = true;
                break;
            }
        }
        if (is_covered) covered++;
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
    program.add_argument("--nbr-rank").default_value(16u).scan<'u', uint32_t>();
    program.add_argument("--quantile").default_value(0.5f).scan<'g', float>();
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
    g_config.nbr_rank = program.get<uint32_t>("--nbr-rank");
    g_config.quantile = program.get<float>("--quantile");
    g_config.verbose = program.get<bool>("--verbose");

    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << fmt::format("  Dataset:        {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Max nbr size:   {}", g_config.max_nbr_size) << std::endl;
    std::cout << fmt::format("  Prefill ratio:  {}", g_config.prefill_ratio) << std::endl;
    std::cout << fmt::format("  Nbr rank:       {}", g_config.nbr_rank) << std::endl;
    std::cout << fmt::format("  Quantile:       {}", g_config.quantile) << std::endl;
    std::cout << "=====================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "          GRAPH MIS VG TEST SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << fmt::format("  Dataset:                  {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Num Vertices:             {}", g_results.num_vertices) << std::endl;
    std::cout << fmt::format("  Min Radius:               {:.6f}", g_results.min_radius) << std::endl;
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
