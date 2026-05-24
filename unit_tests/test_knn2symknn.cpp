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

static auto compute_average_degree(const symmetric_knn_graph::index_t& graph) -> double {
    const auto& nbrs_arr = graph.get_nbrs_arr();
    uint64_t total_edges = 0;
    for (const auto& nbrs : nbrs_arr) {
        total_edges += nbrs.size();
    }
    return static_cast<double>(total_edges) / nbrs_arr.size();
}

struct TestConfig {
    std::string config_path;
    std::string dataset_name;

    layer_config_t knn_layer_config{64};
    knn_graph::propagate_config_t knn_propagate_config{4, 14};

    bool verbose;
} g_config;

struct TestResults {
    double knn_build_time_s = 0.0;
    double symknn_build_time_s = 0.0;
    double avg_degree = 0.0;
    uint32_t num_vertices = 0;
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
        g_test_results.num_vertices = static_cast<uint32_t>(base_vecs.get_num_vecs());

        // Part 1: Build KNN graph
        ARTEA_INFO("Part 1: Building KNN graph...");
        dispatcher_ = std::make_unique<simd_dispatcher_t>(base_vecs.get_vec_dim());
        auto t0 = std::chrono::high_resolution_clock::now();

        knn_graph::index_t knn_index = dispatcher_->dispatch([&](const auto& dist_func) {
            return std::move(knn_graph::factory_t::construct_graph(
                base_vecs,
                g_config.knn_layer_config,
                g_config.knn_propagate_config,
                dist_func
            ).graph);
        });

        auto t1 = std::chrono::high_resolution_clock::now();
        g_test_results.knn_build_time_s =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        ARTEA_INFO(fmt::format("KNN graph built in {:.2f} s", g_test_results.knn_build_time_s));

        // Part 2: Convert knn_graph -> symmetric_knn_graph via move + reverse
        ARTEA_INFO("Part 2: Converting KNN graph to symmetric KNN graph (move + reverse)...");
        t0 = std::chrono::high_resolution_clock::now();

        symknn_graph_ = std::make_unique<symmetric_knn_graph::index_t>(
            dispatcher_->dispatch([&](const auto& dist_func) {
                return std::move(symmetric_knn_graph::factory_t::construct_graph(
                    std::move(knn_index), dist_func
                ).graph);
            })
        );

        t1 = std::chrono::high_resolution_clock::now();
        g_test_results.symknn_build_time_s =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        ARTEA_INFO(fmt::format("Symmetric KNN graph built in {:.2f} s", g_test_results.symknn_build_time_s));

        // Compute average degree
        g_test_results.avg_degree = compute_average_degree(*symknn_graph_);
        ARTEA_INFO(fmt::format("Symmetric KNN graph average degree: {:.2f}", g_test_results.avg_degree));
    }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<simd_dispatcher_t> dispatcher_;
    std::unique_ptr<symmetric_knn_graph::index_t> symknn_graph_;
};

class Knn2SymKnnTest : public ::testing::Test {};

TEST_F(Knn2SymKnnTest, BuildAndReportDegree) {
    EXPECT_GT(g_test_results.avg_degree, 0.0) << "Average degree should be positive";
    EXPECT_GE(g_test_results.avg_degree, g_config.knn_layer_config.max_nbr_size())
        << "Symmetric graph degree should be >= original max_nbr_size due to reverse edges";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_knn2symknn");
    program.add_argument("-c", "--config").default_value(artea::default_dataset_config_path());
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));

    // KNN graph params
    program.add_argument("--knn-max-nbr-size").default_value(96u).scan<'u', uint32_t>();
    program.add_argument("--knn-num-build-loops").default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--knn-num-triu-iters").default_value(14u).scan<'u', uint32_t>();
    program.add_argument("--knn-prefill-ratio").default_value(0.34f).scan<'g', float>();
    program.add_argument("--knn-num-routing-loops").default_value(1u).scan<'u', uint32_t>();
    program.add_argument("--routing-topk").default_value(64u).scan<'u', uint32_t>()
        .help("Routing updater top-k (default: 96)");
    program.add_argument("--routing-queue-size").default_value(96u).scan<'u', uint32_t>()
        .help("Routing updater candidate queue size (default: 128)");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");

    uint32_t knn_max_nbr_size = program.get<uint32_t>("--knn-max-nbr-size");
    g_config.knn_layer_config = layer_config_t(knn_max_nbr_size);
    g_config.knn_propagate_config = knn_graph::propagate_config_t(
        program.get<uint32_t>("--knn-num-build-loops"),
        program.get<uint32_t>("--knn-num-triu-iters"),
        program.get<float>("--knn-prefill-ratio"),
        program.get<uint32_t>("--knn-num-routing-loops"),
        program.get<uint32_t>("--routing-topk"),
        program.get<uint32_t>("--routing-queue-size")
    );

    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << fmt::format("  Dataset:                  {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Config path:              {}", g_config.config_path) << std::endl;
    std::cout << fmt::format("  KNN max nbr size:         {}", g_config.knn_layer_config.max_nbr_size()) << std::endl;
    std::cout << fmt::format("  KNN build loops:          {}", g_config.knn_propagate_config.num_build_loops()) << std::endl;
    std::cout << fmt::format("  KNN triangle updater its: {}", g_config.knn_propagate_config.num_triu_iters()) << std::endl;
    std::cout << fmt::format("  KNN prefill ratio:        {}", g_config.knn_propagate_config.prefill_ratio()) << std::endl;
    std::cout << fmt::format("  KNN routing loops:        {}", g_config.knn_propagate_config.num_routing_loops()) << std::endl;
    std::cout << "=====================\n" << std::endl;

    DataProvider::instance().init();

    int result = RUN_ALL_TESTS();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "              TEST RESULTS SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << fmt::format("  Dataset:            {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Num Vertices:       {}", g_test_results.num_vertices) << std::endl;
    std::cout << fmt::format("  KNN Build Time:     {:.2f} s", g_test_results.knn_build_time_s) << std::endl;
    std::cout << fmt::format("  SymKNN Build Time:  {:.2f} s", g_test_results.symknn_build_time_s) << std::endl;
    std::cout << fmt::format("  Total Build Time:   {:.2f} s", g_test_results.knn_build_time_s + g_test_results.symknn_build_time_s) << std::endl;
    std::cout << fmt::format("  Average Degree:     {:.2f}", g_test_results.avg_degree) << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return result;
}
