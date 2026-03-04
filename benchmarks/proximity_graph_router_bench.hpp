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

/*
 * @FilePath: /Artea/benchmarks/proximity_graph_router_bench.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Proximity graph router benchmark utilities
 */

#pragma once

#include <arena_benchmark/arena_benchmark.hpp>
#include <benchmark/benchmark.h>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/default_context.hpp>
#include <nlohmann/json.hpp>
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace artea {
namespace benchmarks {

using namespace artea::cpu;
using namespace artea::cpu::default_context;

struct SearchParams {
    uint32_t topk;
    uint32_t candidate_queue_size;
    std::string index_path;
    std::string dataset_name;
    std::string algorithm;
    uint32_t max_nbr_size;
    uint32_t extracted_nbr_size;
    double scale_coeffs;
    double shifted_coeffs;
    uint32_t num_outer_iters;
    uint32_t num_inner_iters;
};

struct BenchConfig {
    std::string config_path;
    std::string export_path;
    int64_t repetitions;
    int64_t warm_up;
    std::vector<SearchParams> param_sets;
};

// Global configuration
extern BenchConfig g_config;

// Map from dataset name to DataProvider instance
extern std::map<std::string, std::unique_ptr<vector_dataset_t>> g_datasets;

// Store search results for recall calculation
extern std::map<std::string, idlist_array_t> g_search_results;

class DataProvider {
public:
    static void init(const std::string& dataset_name) {
        if (g_datasets.find(dataset_name) != g_datasets.end()) {
            return; // Already loaded
        }

        auto dataset = std::make_unique<vector_dataset_t>(
            g_config.config_path,
            dataset_name
        );

        vec_dim_t dim = dataset->get_vec_dim();
        vertex_num_t num_base_vecs = dataset->get_num_base_vecs();
        vertex_num_t num_query_vecs = dataset->get_num_query_vecs();

        logger.info(fmt::format("Dataset '{}' loaded:", dataset_name));
        logger.info(fmt::format("  Dimension: {}", dim));
        logger.info(fmt::format("  Base vectors: {}", num_base_vecs));
        logger.info(fmt::format("  Query vectors: {}", num_query_vecs));

        g_datasets[dataset_name] = std::move(dataset);
    }

    static const vector_dataset_t& get_dataset(const std::string& dataset_name) {
        return *g_datasets.at(dataset_name);
    }
};

class IndexProvider {
public:
    static IndexProvider& instance() {
        static IndexProvider inst;
        return inst;
    }

    void load_index(const std::string& index_path, const std::string& dataset_name) {
        const auto& dataset = DataProvider::get_dataset(dataset_name);

        logger.info(fmt::format("Loading search graph from {}...", index_path));
        _search_graph = std::make_unique<search_graph_t>(
            search_graph_t::restore(index_path, dataset.get_base_vecs())
        );

        vertex_num_t num_vertices = _search_graph->get_num_vertices();
        vertex_num_t extracted_nbr_size = _search_graph->get_extracted_nbr_size();

        logger.info(fmt::format("Search graph loaded:"));
        logger.info(fmt::format("  Vertices: {}", num_vertices));
        logger.info(fmt::format("  Extracted neighbors: {}", extracted_nbr_size));
    }

    const search_graph_t& get_search_graph() const { return *_search_graph; }

private:
    std::unique_ptr<search_graph_t> _search_graph;
};

inline auto make_benchmark_func(const SearchParams& params, const std::string& bench_name) {
    return [params, bench_name](benchmark::State& state) {
        const auto& dataset = DataProvider::get_dataset(params.dataset_name);
        const auto& search_graph = IndexProvider::instance().get_search_graph();
        const vec_dim_t dim = dataset.get_vec_dim();
        const vertex_num_t num_queries = dataset.get_num_query_vecs();

        // Create distance function
        dist_func_t dist_func(dim);

        // Create router with correct template parameters
        using router_traits_t = RouterTraits<computer_traits_t, index_traits_t, false>;
        using router_t = ProximityGraphRouter<router_traits_t>;

        router_t router(
            dataset.get_base_vecs(),
            dist_func,
            search_graph,
            params.topk,
            params.candidate_queue_size
        );

        // Initialize router (warmup)
        router.initialize();

        // Get query vectors
        const auto& query_vecs = dataset.get_query_vecs();

        idlist_array_t results;
        for (auto _ : state) {
            // Run batch query
            results = router.batch_query(query_vecs);

            // Prevent optimization from removing the work
            benchmark::DoNotOptimize(results);
            benchmark::ClobberMemory();
        }

        // Store the last result for recall calculation
        g_search_results[bench_name] = std::move(results);

        state.SetItemsProcessed(state.iterations() * num_queries);
    };
}

}  // namespace benchmarks
}  // namespace artea
