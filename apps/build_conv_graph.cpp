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
 * @FilePath: /Artea/examples/build_conv_graph.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Build convergent graph from dataset
 */

#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace artea;
using namespace artea::cpu;

struct GraphParams {
    vertex_num_t max_nbr_size = 32;
    vertex_num_t reserved_nbr_size = 64;
    ratio_t scale_coeffs = 1.00;
    ratio_t shifted_coeffs = 0.00;
    iter_t num_outer_iters = 4;
    iter_t num_inner_iters = 14;
};

int main(int argc, char** argv) {
    argparse::ArgumentParser program("build_conv_graph");
    program.add_description("Build convergent graph from dataset");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    program.add_argument("-o", "--output")
        .default_value(std::string("graph_index_repo/artea/"))
        .help("Output directory for the graph file");

    // Graph construction parameters
    program.add_argument("--max-nbr-size")
        .default_value(uint32_t(64))
        .scan<'u', uint32_t>()
        .help("Maximum number of neighbors per vertex");

    program.add_argument("--scale-coeffs")
        .default_value(1.00)
        .scan<'g', double>()
        .help("Scale coefficient for triangle updater");

    program.add_argument("--shifted-coeffs")
        .default_value(0.00)
        .scan<'g', double>()
        .help("Shifted coefficient for triangle updater");

    program.add_argument("--num-outer-iters")
        .default_value(uint32_t(4))
        .scan<'u', uint32_t>()
        .help("Number of outer iterations");

    program.add_argument("--num-inner-iters")
        .default_value(uint32_t(14))
        .scan<'u', uint32_t>()
        .help("Number of inner iterations");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Parse configuration
    std::string config_path = program.get<std::string>("--config");
    std::string dataset_name = program.get<std::string>("--dataset");
    std::string output_dir = program.get<std::string>("--output");

    GraphParams params;
    params.max_nbr_size = program.get<uint32_t>("--max-nbr-size");
    params.reserved_nbr_size = static_cast<uint32_t>(params.max_nbr_size * 1.5);
    params.scale_coeffs = program.get<double>("--scale-coeffs");
    params.shifted_coeffs = program.get<double>("--shifted-coeffs");
    params.num_outer_iters = program.get<uint32_t>("--num-outer-iters");
    params.num_inner_iters = program.get<uint32_t>("--num-inner-iters");

    logger.info(fmt::format("Graph Construction Configuration:"));
    logger.info(fmt::format("  Dataset: {}", dataset_name));
    logger.info(fmt::format("  Config path: {}", config_path));
    logger.info(fmt::format("  Output directory: {}", output_dir));
    logger.info(fmt::format("  Max neighbors: {}", params.max_nbr_size));
    logger.info(fmt::format("  Reserved neighbors: {}", params.reserved_nbr_size));
    logger.info(fmt::format("  Scale coefficient: {:.2f}", params.scale_coeffs));
    logger.info(fmt::format("  Shifted coefficient: {:.2f}", params.shifted_coeffs));
    logger.info(fmt::format("  Outer iterations: {}", params.num_outer_iters));
    logger.info(fmt::format("  Inner iterations: {}", params.num_inner_iters));

    // Load dataset
    logger.info("Loading dataset...");
    vector_dataset_t dataset(config_path, dataset_name);

    vec_dim_t dim = dataset.get_vec_dim();
    vertex_num_t num_base_vecs = dataset.get_num_base_vecs();

    logger.info(fmt::format("Dataset loaded:"));
    logger.info(fmt::format("  Dimension: {}", dim));
    logger.info(fmt::format("  Base vectors: {}", num_base_vecs));

    // Construct graph
    logger.info("Constructing convergent graph...");
    auto start_time = std::chrono::high_resolution_clock::now();

    layer_config_t layer_config(params.max_nbr_size, params.reserved_nbr_size);
    conv_graph::edges_builder_config_t edges_builder_config(
        params.scale_coeffs,
        params.shifted_coeffs,
        params.num_outer_iters,
        params.num_inner_iters
    );

    conv_graph_factory_t conv_graph_factory;
    flat_graph_t flat_graph = conv_graph_factory.construct_graph(
        dataset,
        layer_config,
        edges_builder_config
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    logger.info(fmt::format("Graph construction completed in {:.2f} seconds", duration.count() / 1000.0));

    // Calculate and output index size
    index_size_calculator_t index_size_calc;
    auto index_size_info = index_size_calc.calculate_size(flat_graph);

    logger.info(fmt::format("Index size: {:.2f} MB ({} bytes)",
        index_size_info.total_mb, index_size_info.total_bytes));

    // Generate directory name with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::ostringstream dirname_stream;
    dirname_stream << "conv_graph_" << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    std::string dirname = dirname_stream.str();

    std::string subdir = "conv_graph." + dataset_name;
    std::filesystem::path output_path = std::filesystem::path(output_dir) / subdir / dirname;

    // Print construction summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    CONVERGENT GRAPH BUILD SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "\n--- Dataset ---" << std::endl;
    std::cout << fmt::format("  Name:                   {}", dataset_name) << std::endl;
    std::cout << fmt::format("  Base vectors:           {}", num_base_vecs) << std::endl;
    std::cout << fmt::format("  Vector dimension:       {}", dim) << std::endl;
    std::cout << "\n--- Graph Configuration ---" << std::endl;
    std::cout << fmt::format("  Max nbr size:           {}", params.max_nbr_size) << std::endl;
    std::cout << fmt::format("  Scale coeffs:           {:.2f}", params.scale_coeffs) << std::endl;
    std::cout << fmt::format("  Shifted coeffs:         {:.2f}", params.shifted_coeffs) << std::endl;
    std::cout << fmt::format("  Num outer iters:        {}", params.num_outer_iters) << std::endl;
    std::cout << fmt::format("  Num inner iters:        {}", params.num_inner_iters) << std::endl;
    std::cout << "\n--- Build Results ---" << std::endl;
    std::cout << fmt::format("  Num vertices:           {}", flat_graph.get_num_vertices()) << std::endl;
    std::cout << fmt::format("  Index size:             {:.2f} MB ({} bytes)", index_size_info.total_mb, index_size_info.total_bytes) << std::endl;
    std::cout << fmt::format("  Construction time:      {:.2f} s", duration.count() / 1000.0) << std::endl;
    std::cout << "\n--- Output ---" << std::endl;
    std::cout << fmt::format("  Index path:             {}", output_path.string()) << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;

    logger.info(fmt::format("Saving flat graph to {}...", output_path.string()));

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_path);

    // Prepare metadata for snapshot
    nlohmann::json metadata;
    metadata["dataset"] = dataset_name;
    metadata["vec_dim"] = dataset.get_base_vecs().get_vec_dim();
    metadata["build_params"] = {
        {"scale_coeffs", params.scale_coeffs},
        {"shifted_coeffs", params.shifted_coeffs},
        {"num_outer_iters", params.num_outer_iters},
        {"num_inner_iters", params.num_inner_iters}
    };

    // Get current timestamp in ISO format
    std::ostringstream timestamp_stream;
    timestamp_stream << std::put_time(std::gmtime(&time_t_now), "%Y-%m-%dT%H:%M:%SZ");
    metadata["timestamp"] = timestamp_stream.str();

    flat_graph_file_manager_t::snapshot(flat_graph, output_path.string(), metadata);

    logger.info("Graph saved successfully");

    // Update index registry JSON
    std::filesystem::path registry_path = std::filesystem::path(output_dir) / "index_registry.json";

    // Prepare parameters JSON
    nlohmann::json index_params;
    index_params["dataset"] = dataset_name;
    index_params["max_nbr_size"] = params.max_nbr_size;
    index_params["scale_coeffs"] = params.scale_coeffs;
    index_params["shifted_coeffs"] = params.shifted_coeffs;
    index_params["num_outer_iters"] = params.num_outer_iters;
    index_params["num_inner_iters"] = params.num_inner_iters;

    // Use relative path from project root: output_dir/subdir/dirname
    std::filesystem::path relative_index_path = std::filesystem::path(output_dir) / subdir / dirname;

    // Register the index
    index_register_util_t::register_index(
        registry_path,
        "conv_graph",
        index_params,
        relative_index_path.string()
    );

    return 0;
}