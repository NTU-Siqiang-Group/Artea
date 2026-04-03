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
 * @FilePath: /Artea/apps/build_artea_graph.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Build hierarchical Artea graph from dataset
 */

#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>

using namespace artea;
using namespace artea::cpu;

int main(int argc, char** argv) {
    argparse::ArgumentParser program("build_artea_graph");
    program.add_description("Build hierarchical Artea graph from dataset");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    program.add_argument("-o", "--output")
        .default_value(std::string("graph_index_repo/artea/"))
        .help("Output directory for the graph index");

    program.add_argument("--shuffle-seed").scan<'u', uint32_t>()
        .help("Shuffle seed (if not specified, uses random seed)");

    // Bottom layer config
    program.add_argument("--bl-max-nbr-size").default_value(96u).scan<'u', uint32_t>();
    program.add_argument("--bl-scale-coeffs").default_value(1.10f).scan<'g', float>();
    program.add_argument("--bl-shifted-coeffs").default_value(0.10f).scan<'g', float>();

    // Upper layer config
    program.add_argument("--ul-max-nbr-size").default_value(96u).scan<'u', uint32_t>();
    program.add_argument("--ul-scale-coeffs").default_value(1.10f).scan<'g', float>();
    program.add_argument("--ul-shifted-coeffs").default_value(0.10f).scan<'g', float>();

    // Propagate config (shared between layers)
    program.add_argument("--num-build-loops").default_value(5u).scan<'u', uint32_t>();
    program.add_argument("--num-triu-iters").default_value(12u).scan<'u', uint32_t>();
    program.add_argument("--prefill-ratio").default_value(0.34f).scan<'g', float>();
    program.add_argument("--num-routing-loops").default_value(1u).scan<'u', uint32_t>();
    program.add_argument("--routing-topk").default_value(64u).scan<'u', uint32_t>();
    program.add_argument("--routing-queue-size").default_value(96u).scan<'u', uint32_t>();

    // R-net config
    program.add_argument("--mis-radix").default_value(1.2f).scan<'g', float>()
        .help("MIS coarse-to-fine radix (default: 1.2)");
    program.add_argument("--mis-max-power").default_value(12u).scan<'u', uint32_t>()
        .help("MIS coarse-to-fine max power (default: 12)");
    program.add_argument("--beta").default_value(1.44f).scan<'g', float>()
        .help("Radius growth between layers (default: 1.44)");

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

    // Load dataset
    ARTEA_INFO(fmt::format("Loading dataset: {} from {}", dataset_name, config_path));
    vector_dataset_t dataset(config_path, dataset_name);

    const auto& base_vecs = dataset.get_base_vecs();
    ARTEA_INFO(fmt::format("Dataset loaded: {} vectors, {} dims",
        base_vecs.get_num_vecs(), base_vecs.get_vec_dim()));

    // Shuffle dataset (always enabled)
    ARTEA_INFO("Shuffling dataset...");
    uint32_t shuffle_seed = program.is_used("--shuffle-seed")
        ? program.get<uint32_t>("--shuffle-seed")
        : std::random_device{}();
    shuffle_seed = dataset.shuffle_in_place(shuffle_seed);

    // Create layer configs
    uint32_t bl_max_nbr_size = program.get<uint32_t>("--bl-max-nbr-size");
    uint32_t ul_max_nbr_size = program.get<uint32_t>("--ul-max-nbr-size");

    layer_config_t bottom_layer_config(bl_max_nbr_size, static_cast<uint32_t>(bl_max_nbr_size * 1.5));
    layer_config_t upper_layer_config(ul_max_nbr_size, static_cast<uint32_t>(ul_max_nbr_size * 1.5));

    artea_graph::pruning_config_t bottom_pruning_config(
        program.get<float>("--bl-scale-coeffs"),
        program.get<float>("--bl-shifted-coeffs"));
    artea_graph::pruning_config_t upper_pruning_config(
        program.get<float>("--ul-scale-coeffs"),
        program.get<float>("--ul-shifted-coeffs"));

    artea_graph::propagate_config_t propagate_config(
        program.get<uint32_t>("--num-build-loops"),
        program.get<uint32_t>("--num-triu-iters"),
        program.get<float>("--prefill-ratio"),
        program.get<uint32_t>("--num-routing-loops"),
        program.get<uint32_t>("--routing-topk"),
        program.get<uint32_t>("--routing-queue-size"));

    artea_graph::rnet_config_t rnet_config(
        program.get<float>("--mis-radix"),
        program.get<uint32_t>("--mis-max-power"),
        program.get<float>("--beta"));

    // Construct hierarchical graph via factory
    ARTEA_INFO("Constructing hierarchical Artea graph...");
    auto construction_start = std::chrono::high_resolution_clock::now();

    auto hierarchical_graph = artea_graph::factory_t::construct_graph(
        base_vecs,
        bottom_layer_config, upper_layer_config,
        bottom_pruning_config, upper_pruning_config,
        propagate_config, rnet_config
    );

    auto construction_end = std::chrono::high_resolution_clock::now();
    auto construction_duration = std::chrono::duration_cast<std::chrono::milliseconds>(construction_end - construction_start);

    ARTEA_INFO(fmt::format("Graph construction completed: {} layers in {:.2f} s",
        hierarchical_graph.get_num_layers(), construction_duration.count() / 1000.0));

    // Output layer vertices information
    for (uint32_t layer_id = 0; layer_id < hierarchical_graph.get_num_layers(); ++layer_id) {
        const auto& layer_vecs = hierarchical_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);
        ARTEA_INFO(fmt::format("  Layer {}: {} vertices", layer_id, layer_vecs.get_num_vecs()));
    }

    // Calculate and output index size
    index_size_calculator_t index_size_calc;
    auto index_size_info = index_size_calc.calculate_hierarchical_graph_size(hierarchical_graph);

    ARTEA_INFO(fmt::format("Index size: {:.2f} MB ({} bytes)",
        index_size_info.total_mb, index_size_info.total_bytes));

    // Generate directory name with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::ostringstream dirname_stream;
    dirname_stream << "artea_graph_" << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    std::string dirname = dirname_stream.str();

    std::string subdir = "artea_graph." + dataset_name;
    std::filesystem::path output_path = std::filesystem::path(output_dir) / subdir / dirname;

    // Print construction summary
    double total_time_s = construction_duration.count() / 1000.0;
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "                    ARTEA HIERARCHICAL GRAPH BUILD SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "\n--- Dataset ---" << std::endl;
    std::cout << fmt::format("  Name:                   {}", dataset_name) << std::endl;
    std::cout << fmt::format("  Base vectors:           {}", base_vecs.get_num_vecs()) << std::endl;
    std::cout << fmt::format("  Vector dimension:       {}", base_vecs.get_vec_dim()) << std::endl;
    std::cout << fmt::format("  Shuffle seed:           {}", shuffle_seed) << std::endl;
    std::cout << "\n--- Graph Structure ---" << std::endl;
    std::cout << fmt::format("  Num layers:             {}", hierarchical_graph.get_num_layers()) << std::endl;
    std::cout << fmt::format("  Index size:             {:.2f} MB ({} bytes)", index_size_info.total_mb, index_size_info.total_bytes) << std::endl;

    // Output layer-by-layer statistics
    std::cout << "\n  Layer Statistics:" << std::endl;
    std::cout << fmt::format("  {:<8} {:<15} {:<15}", "Layer", "Vertices", "Edges") << std::endl;
    std::cout << "  " << std::string(40, '-') << std::endl;

    for (uint32_t layer_id = 0; layer_id < hierarchical_graph.get_num_layers(); ++layer_id) {
        const auto& layer_vecs = hierarchical_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);
        uint32_t num_vertices = layer_vecs.get_num_vecs();

        uint64_t num_edges = 0;
        const auto& layer_graph = hierarchical_graph.get_layer_graph(layer_id);
        for (uint32_t v = 0; v < num_vertices; ++v) {
            num_edges += layer_graph.fetch_nbrs(v).size();
        }

        std::cout << fmt::format("  {:<8} {:<15} {:<15}",
            layer_id, num_vertices, num_edges) << std::endl;
    }

    std::cout << "\n--- Layer Configuration ---" << std::endl;
    std::cout << fmt::format("  Bottom max nbr size:    {}", bl_max_nbr_size) << std::endl;
    std::cout << fmt::format("  Upper max nbr size:     {}", ul_max_nbr_size) << std::endl;
    std::cout << "  Bottom pruning:" << std::endl;
    std::cout << fmt::format("    Scale coeffs:         {}", program.get<float>("--bl-scale-coeffs")) << std::endl;
    std::cout << fmt::format("    Shifted coeffs:       {}", program.get<float>("--bl-shifted-coeffs")) << std::endl;
    std::cout << "  Upper pruning:" << std::endl;
    std::cout << fmt::format("    Scale coeffs:         {}", program.get<float>("--ul-scale-coeffs")) << std::endl;
    std::cout << fmt::format("    Shifted coeffs:       {}", program.get<float>("--ul-shifted-coeffs")) << std::endl;
    std::cout << "\n--- Propagate Config ---" << std::endl;
    std::cout << fmt::format("  Build loops:            {}", program.get<uint32_t>("--num-build-loops")) << std::endl;
    std::cout << fmt::format("  Triangle updater iters: {}", program.get<uint32_t>("--num-triu-iters")) << std::endl;
    std::cout << fmt::format("  Prefill ratio:          {}", program.get<float>("--prefill-ratio")) << std::endl;
    std::cout << fmt::format("  Routing loops:          {}", program.get<uint32_t>("--num-routing-loops")) << std::endl;
    std::cout << fmt::format("  Routing top-k:          {}", program.get<uint32_t>("--routing-topk")) << std::endl;
    std::cout << fmt::format("  Routing queue size:     {}", program.get<uint32_t>("--routing-queue-size")) << std::endl;
    std::cout << "\n--- R-Net Config ---" << std::endl;
    std::cout << fmt::format("  MIS radix:              {}", program.get<float>("--mis-radix")) << std::endl;
    std::cout << fmt::format("  MIS max power:          {}", program.get<uint32_t>("--mis-max-power")) << std::endl;
    std::cout << fmt::format("  R-net beta:             {}", program.get<float>("--beta")) << std::endl;
    std::cout << "\n--- Construction Time ---" << std::endl;
    std::cout << fmt::format("  Graph construction:     {:.2f} s", total_time_s) << std::endl;
    std::cout << "\n--- Output ---" << std::endl;
    std::cout << fmt::format("  Index path:             {}", output_path.string()) << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;

    ARTEA_INFO(fmt::format("Saving hierarchical graph to {}...", output_path.string()));
    std::filesystem::create_directories(output_path);

    // Prepare metadata
    nlohmann::json metadata;
    metadata["dataset"] = dataset_name;
    metadata["vec_dim"] = base_vecs.get_vec_dim();
    metadata["num_base_vecs"] = base_vecs.get_num_vecs();
    metadata["shuffle_seed"] = shuffle_seed;

    std::ostringstream timestamp_stream;
    timestamp_stream << std::put_time(std::gmtime(&time_t_now), "%Y-%m-%dT%H:%M:%SZ");
    metadata["timestamp"] = timestamp_stream.str();

    // Save hierarchical graph
    hierarchical_graph_file_manager_t::snapshot(hierarchical_graph, output_path.string(), metadata);

    ARTEA_INFO("Graph saved successfully");

    // Update index registry
    std::filesystem::path registry_path = std::filesystem::path(output_dir) / "index_registry.json";

    nlohmann::json index_params;
    index_params["dataset"] = dataset_name;
    index_params["mis_radix"] = program.get<float>("--mis-radix");
    index_params["mis_max_power"] = program.get<uint32_t>("--mis-max-power");
    index_params["rnet_beta"] = std::round(program.get<float>("--beta") * 100.0f) / 100.0f;
    index_params["bl_max_nbr_size"] = program.get<uint32_t>("--bl-max-nbr-size");
    index_params["ul_max_nbr_size"] = program.get<uint32_t>("--ul-max-nbr-size");

    std::filesystem::path relative_index_path = std::filesystem::path(output_dir) / subdir / dirname;

    index_register_util_t::register_index(
        registry_path,
        "artea_graph",
        index_params,
        relative_index_path.string()
    );

    return 0;
}
