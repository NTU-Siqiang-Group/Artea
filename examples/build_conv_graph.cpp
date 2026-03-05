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
#include <artea/cpu/framework/default_context.hpp>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace artea;
using namespace artea::cpu;
using namespace artea::cpu::default_context;

struct GraphParams {
    vertex_num_t max_nbr_size = 32;
    vertex_num_t reserved_nbr_size = 64;
    ratio_t scale_coeffs = 1.10;
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

    program.add_argument("--extracted-nbr-size")
        .default_value(uint32_t(32))
        .scan<'u', uint32_t>()
        .help("Fixed number of neighbors for flat search graph (extracted from flat graph)");

    // Graph construction parameters
    program.add_argument("--max-nbr-size")
        .default_value(uint32_t(64))
        .scan<'u', uint32_t>()
        .help("Maximum number of neighbors per vertex");

    program.add_argument("--reserved-nbr-size")
        .default_value(uint32_t(64))
        .scan<'u', uint32_t>()
        .help("Reserved neighbor size for memory allocation");

    program.add_argument("--scale-coeffs")
        .default_value(1.10)
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
    vertex_num_t extracted_nbr_size = program.get<uint32_t>("--extracted-nbr-size");

    GraphParams params;
    params.max_nbr_size = program.get<uint32_t>("--max-nbr-size");
    params.reserved_nbr_size = program.get<uint32_t>("--reserved-nbr-size");
    params.scale_coeffs = program.get<double>("--scale-coeffs");
    params.shifted_coeffs = program.get<double>("--shifted-coeffs");
    params.num_outer_iters = program.get<uint32_t>("--num-outer-iters");
    params.num_inner_iters = program.get<uint32_t>("--num-inner-iters");

    logger.info(fmt::format("Graph Construction Configuration:"));
    logger.info(fmt::format("  Dataset: {}", dataset_name));
    logger.info(fmt::format("  Config path: {}", config_path));
    logger.info(fmt::format("  Output directory: {}", output_dir));
    logger.info(fmt::format("  Extracted neighbors: {}", extracted_nbr_size));
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

    conv_graph_factory_t conv_graph_factory;
    flat_graph_t flat_graph = conv_graph_factory.construct_graph(
        dataset,
        params.max_nbr_size,
        params.reserved_nbr_size,
        params.scale_coeffs,
        params.shifted_coeffs,
        params.num_outer_iters,
        params.num_inner_iters
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    logger.info(fmt::format("Graph construction completed in {:.2f} seconds", duration.count() / 1000.0));

    // Convert to flat search graph
    logger.info("Converting to flat search graph...");
    flat_search_graph_factory_t flat_search_graph_factory;
    flat_search_graph_t flat_search_graph = flat_search_graph_factory.from_flat_graph(flat_graph, extracted_nbr_size);

    // Save graph
    std::string subdir = "conv_graph." + dataset_name;

    // Format coefficients with 2 decimal places
    std::ostringstream sc_stream, sh_stream;
    sc_stream << std::fixed << std::setprecision(2) << params.scale_coeffs;
    sh_stream << std::fixed << std::setprecision(2) << params.shifted_coeffs;

    std::string filename = "mns" + std::to_string(params.max_nbr_size) +
                          "_ens" + std::to_string(extracted_nbr_size) +
                          "_sc" + sc_stream.str() +
                          "_sh" + sh_stream.str() +
                          "_noi" + std::to_string(params.num_outer_iters) +
                          "_nii" + std::to_string(params.num_inner_iters) + ".index";
    std::filesystem::path output_path = std::filesystem::path(output_dir) / subdir / filename;
    logger.info(fmt::format("Saving graph to {}...", output_path.string()));

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_path.parent_path());

    flat_search_graph.snapshot(output_path.string());

    logger.info("Graph saved successfully");

    // Update index registry JSON
    std::filesystem::path registry_path = std::filesystem::path(output_dir) / "index_registry.json";

    // Prepare parameters JSON
    nlohmann::json index_params;
    index_params["dataset"] = dataset_name;
    index_params["max_nbr_size"] = params.max_nbr_size;
    index_params["extracted_nbr_size"] = extracted_nbr_size;
    index_params["scale_coeffs"] = params.scale_coeffs;
    index_params["shifted_coeffs"] = params.shifted_coeffs;
    index_params["num_outer_iters"] = params.num_outer_iters;
    index_params["num_inner_iters"] = params.num_inner_iters;

    // Use relative path from project root: output_dir/subdir/filename
    std::filesystem::path relative_index_path = std::filesystem::path(output_dir) / subdir / filename;

    // Register the index
    index_register_util_t::register_index(
        registry_path,
        "conv_graph",
        index_params,
        relative_index_path.string()
    );

    return 0;
}