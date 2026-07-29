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

template <DistanceMetricsT Metric, vec_dim_t Dim>
static auto compute_average_degree(const symmetric_knn_graph::index_t<Metric, Dim>& graph) -> double {
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
    std::string metric;

    // layer_config_t is metric/dim-independent (IndexTraits) and can live here.
    layer_config_t knn_layer_config{64};

    // knn_graph::propagate_config_t is now a <Metric, Dim> alias, so we keep the raw
    // build params here and construct the metric-dependent config object inside
    // the dispatched body (metric from --metric, padded dim from the dataset).
    uint32_t knn_num_build_loops;
    uint32_t knn_num_triu_iters;
    float    knn_prefill_ratio;
    uint32_t knn_num_routing_loops;
    uint32_t knn_routing_topk;
    uint32_t knn_routing_queue_size;

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

        // Metric from the --metric input, padded dim from the loaded dataset;
        // the metric-dependent build runs behind <Metric, Dim>. The dataset itself
        // is metric/dim-independent and stays out here.
        const auto dataset_info = DatasetInfra{parse_metric(g_config.metric), base_vecs.get_vec_dim()};

        infra_dispatch(dataset_info, ARTEA_METRIC_LAMBDA(void) {
            knn_graph::propagate_config_t<Metric, Dim> knn_propagate_config(
                g_config.knn_num_build_loops,
                g_config.knn_num_triu_iters,
                g_config.knn_prefill_ratio,
                g_config.knn_num_routing_loops,
                g_config.knn_routing_topk,
                g_config.knn_routing_queue_size);

            // Part 1: Build KNN graph
            ARTEA_INFO("Part 1: Building KNN graph...");
            auto t0 = std::chrono::high_resolution_clock::now();

            knn_graph::index_t<Metric, Dim> knn_index = std::move(
                knn_graph::factory_t<Metric, Dim>::construct_graph(
                    base_vecs,
                    g_config.knn_layer_config,
                    knn_propagate_config
                ).graph
            );

            auto t1 = std::chrono::high_resolution_clock::now();
            g_test_results.knn_build_time_s =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
            ARTEA_INFO(fmt::format("KNN graph built in {:.2f} s", g_test_results.knn_build_time_s));

            // Part 2: Convert knn_graph -> symmetric_knn_graph via move + reverse
            ARTEA_INFO("Part 2: Converting KNN graph to symmetric KNN graph (move + reverse)...");
            t0 = std::chrono::high_resolution_clock::now();

            symmetric_knn_graph::index_t<Metric, Dim> symknn_graph = std::move(
                symmetric_knn_graph::factory_t<Metric, Dim>::construct_graph(std::move(knn_index)).graph
            );

            t1 = std::chrono::high_resolution_clock::now();
            g_test_results.symknn_build_time_s =
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
            ARTEA_INFO(fmt::format("Symmetric KNN graph built in {:.2f} s", g_test_results.symknn_build_time_s));

            // Compute average degree
            g_test_results.avg_degree = compute_average_degree<Metric, Dim>(symknn_graph);
            ARTEA_INFO(fmt::format("Symmetric KNN graph average degree: {:.2f}", g_test_results.avg_degree));
        });
    }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
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
    program.add_argument("--metric").default_value(std::string("euclidean")).help("Distance metric: 'euclidean', 'inner_product', or 'cosine'");

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
    g_config.metric = program.get<std::string>("--metric");

    uint32_t knn_max_nbr_size = program.get<uint32_t>("--knn-max-nbr-size");
    g_config.knn_layer_config = layer_config_t(knn_max_nbr_size);
    g_config.knn_num_build_loops    = program.get<uint32_t>("--knn-num-build-loops");
    g_config.knn_num_triu_iters     = program.get<uint32_t>("--knn-num-triu-iters");
    g_config.knn_prefill_ratio      = program.get<float>("--knn-prefill-ratio");
    g_config.knn_num_routing_loops  = program.get<uint32_t>("--knn-num-routing-loops");
    g_config.knn_routing_topk       = program.get<uint32_t>("--routing-topk");
    g_config.knn_routing_queue_size = program.get<uint32_t>("--routing-queue-size");

    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << fmt::format("  Dataset:                  {}", g_config.dataset_name) << std::endl;
    std::cout << fmt::format("  Config path:              {}", g_config.config_path) << std::endl;
    std::cout << fmt::format("  KNN max nbr size:         {}", g_config.knn_layer_config.max_nbr_size()) << std::endl;
    std::cout << fmt::format("  KNN build loops:          {}", g_config.knn_num_build_loops) << std::endl;
    std::cout << fmt::format("  KNN triangle updater its: {}", g_config.knn_num_triu_iters) << std::endl;
    std::cout << fmt::format("  KNN prefill ratio:        {}", g_config.knn_prefill_ratio) << std::endl;
    std::cout << fmt::format("  KNN routing loops:        {}", g_config.knn_num_routing_loops) << std::endl;
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
