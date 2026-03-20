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
using namespace artea::cpu::default_context;

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

    // Vertices builder parameters
    program.add_argument("--beta").default_value(1.69f).scan<'g', float>();
    program.add_argument("--coverage-ratio").default_value(0.999f).scan<'g', float>();
    program.add_argument("--confidence").default_value(0.950f).scan<'g', float>();
    program.add_argument("--max-result-ratio").default_value(0.2f).scan<'g', float>();
    program.add_argument("--sampling-batch-size").default_value(2048u).scan<'u', uint32_t>();
    program.add_argument("--shuffle-seed").scan<'u', uint32_t>()
        .help("Shuffle seed (if not specified, uses random seed)");

    // Bottom layer config
    program.add_argument("--bl-max-nbr-size").default_value(32u).scan<'u', uint32_t>();
    program.add_argument("--bl-reserved-nbr-size").scan<'u', uint32_t>()
        .help("Bottom layer reserved neighbor size (defaults to bl-max-nbr-size * 1.5)");

    // Upper layer config
    program.add_argument("--ul-max-nbr-size").default_value(24u).scan<'u', uint32_t>();
    program.add_argument("--ul-reserved-nbr-size").scan<'u', uint32_t>()
        .help("Upper layer reserved neighbor size (defaults to ul-max-nbr-size * 1.5)");

    // Bottom edges builder config
    program.add_argument("--bl-scale-coeffs").default_value(1.0f).scan<'g', float>();
    program.add_argument("--bl-shifted-coeffs").default_value(0.0f).scan<'g', float>();
    program.add_argument("--bl-num-outer-iters").default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--bl-num-inner-iters").default_value(14u).scan<'u', uint32_t>();

    // Upper edges builder config
    program.add_argument("--ul-scale-coeffs").default_value(1.0f).scan<'g', float>();
    program.add_argument("--ul-shifted-coeffs").default_value(0.0f).scan<'g', float>();
    program.add_argument("--ul-num-outer-iters").default_value(4u).scan<'u', uint32_t>();
    program.add_argument("--ul-num-inner-iters").default_value(14u).scan<'u', uint32_t>();

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
    logger.info(fmt::format("Loading dataset: {} from {}", dataset_name, config_path));
    vector_dataset_t dataset(config_path, dataset_name);
    dist_func_t dist_func(dataset.get_base_vecs().get_vec_dim());

    const auto& base_vecs = dataset.get_base_vecs();
    logger.info(fmt::format("Dataset loaded: {} vectors, {} dims",
        base_vecs.get_num_vecs(), base_vecs.get_vec_dim()));

    // Shuffle dataset (always enabled)
    logger.info("Shuffling dataset...");
    uint32_t shuffle_seed = program.is_used("--shuffle-seed")
        ? program.get<uint32_t>("--shuffle-seed")
        : std::random_device{}();
    shuffle_seed = dataset.shuffle_in_place(shuffle_seed);

    // Probe min_radius using radius prober
    logger.info("Probing min_radius from dataset...");
    constexpr float QUANTILE = 0.001f;
    constexpr float CONFIDENCE = 0.95f;
    constexpr float RELATIVE_ERR = 0.05f;

    radius_prober_t prober(dist_func);
    auto probe_start = std::chrono::high_resolution_clock::now();
    auto probe_result = prober.probe(base_vecs, QUANTILE, CONFIDENCE, RELATIVE_ERR);
    auto probe_end = std::chrono::high_resolution_clock::now();
    auto probe_duration = std::chrono::duration_cast<std::chrono::milliseconds>(probe_end - probe_start);

    float min_radius = probe_result.radius;
    logger.info(fmt::format("Probed min_radius: {:.6f} (quantile: {:.4f}, samples: {}, time: {:.2f}s)",
        min_radius, probe_result.quantile, probe_result.num_dists_sampled, probe_duration.count() / 1000.0));

    // Create layer configs
    uint32_t bl_max_nbr_size = program.get<uint32_t>("--bl-max-nbr-size");
    uint32_t bl_reserved_nbr_size = program.is_used("--bl-reserved-nbr-size")
        ? program.get<uint32_t>("--bl-reserved-nbr-size")
        : static_cast<uint32_t>(bl_max_nbr_size * 1.5);

    uint32_t ul_max_nbr_size = program.get<uint32_t>("--ul-max-nbr-size");
    uint32_t ul_reserved_nbr_size = program.is_used("--ul-reserved-nbr-size")
        ? program.get<uint32_t>("--ul-reserved-nbr-size")
        : static_cast<uint32_t>(ul_max_nbr_size * 1.5);

    layer_config_t bottom_layer_config(bl_max_nbr_size, bl_reserved_nbr_size);
    layer_config_t upper_layer_config(ul_max_nbr_size, ul_reserved_nbr_size);

    // Create edges builder configs
    edges_builder_config_t bottom_edges_config(
        program.get<float>("--bl-scale-coeffs"),
        program.get<float>("--bl-shifted-coeffs"),
        program.get<uint32_t>("--bl-num-outer-iters"),
        program.get<uint32_t>("--bl-num-inner-iters")
    );
    edges_builder_config_t upper_edges_config(
        program.get<float>("--ul-scale-coeffs"),
        program.get<float>("--ul-shifted-coeffs"),
        program.get<uint32_t>("--ul-num-outer-iters"),
        program.get<uint32_t>("--ul-num-inner-iters")
    );

    // Create vertices builder config
    greedy_vertices_builder_config_t vertices_builder_config(
        min_radius,
        program.get<float>("--beta"),
        program.get<float>("--coverage-ratio"),
        program.get<float>("--confidence"),
        program.get<float>("--max-result-ratio"),
        program.get<uint32_t>("--sampling-batch-size")
    );

    // Create hierarchical graph
    hierarchical_graph_t hierarchical_graph(
        base_vecs,
        bottom_layer_config,
        upper_layer_config,
        bottom_edges_config,
        upper_edges_config,
        vertices_builder_config
    );

    // Step 1: Construct vertices
    logger.info("Step 1: Constructing hierarchical vertices...");
    auto vertices_start = std::chrono::high_resolution_clock::now();

    hierarchical_vertices_builder_t::template construct<VGPolicyT::rnet_selection>(
        dist_func,
        hierarchical_graph,
        vertices_builder_config
    );

    auto vertices_end = std::chrono::high_resolution_clock::now();
    auto vertices_duration = std::chrono::duration_cast<std::chrono::milliseconds>(vertices_end - vertices_start);

    logger.info(fmt::format("Vertices construction completed: {} layers in {:.2f} s",
        hierarchical_graph.get_num_layers(), vertices_duration.count() / 1000.0));

    // Output layer vertices information
    for (uint32_t layer_id = 0; layer_id < hierarchical_graph.get_num_layers(); ++layer_id) {
        const auto& layer_vecs = hierarchical_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);
        logger.info(fmt::format("  Layer {}: {} vertices", layer_id, layer_vecs.get_num_vecs()));
    }

    // Step 2: Construct edges
    logger.info("Step 2: Constructing hierarchical edges...");
    auto edges_start = std::chrono::high_resolution_clock::now();

    hierarchical_edges_builder_t::template construct<EGPolicyT::conv_graph_descent>(
        dist_func,
        hierarchical_graph
    );

    auto edges_end = std::chrono::high_resolution_clock::now();
    auto edges_duration = std::chrono::duration_cast<std::chrono::milliseconds>(edges_end - edges_start);

    logger.info(fmt::format("Edges construction completed in {:.2f} s", edges_duration.count() / 1000.0));

    // Calculate and output index size
    index_size_calculator_t index_size_calc;
    auto index_size_info = index_size_calc.calculate_size(hierarchical_graph);

    logger.info(fmt::format("Index size: {:.2f} MB ({} bytes)",
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
    double total_time_s = (vertices_duration.count() + edges_duration.count()) / 1000.0;
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
    std::cout << fmt::format("  {:<8} {:<15} {:<15} {:<20}", "Layer", "Vertices", "Edges", "R-Net Radius") << std::endl;
    std::cout << "  " << std::string(60, '-') << std::endl;

    float current_radius = min_radius * program.get<float>("--beta");
    for (uint32_t layer_id = 0; layer_id < hierarchical_graph.get_num_layers(); ++layer_id) {
        const auto& layer_vecs = hierarchical_graph.get_hier_vecs_manager().get_layer_vecs(layer_id);
        uint32_t num_vertices = layer_vecs.get_num_vecs();

        // Count edges for this layer
        uint64_t num_edges = 0;
        const auto& layer_graph = hierarchical_graph.get_layer_graph(layer_id);
        for (uint32_t v = 0; v < num_vertices; ++v) {
            num_edges += layer_graph.fetch_nbrs(v).size();
        }

        // Display layer info
        if (layer_id == 0) {
            std::cout << fmt::format("  {:<8} {:<15} {:<15} {:<20}",
                layer_id, num_vertices, num_edges, "N/A (base layer)") << std::endl;
        } else {
            std::cout << fmt::format("  {:<8} {:<15} {:<15} {:<20.6f}",
                layer_id, num_vertices, num_edges, current_radius) << std::endl;
            current_radius *= program.get<float>("--beta");
        }
    }
    std::cout << std::endl;
    std::cout << "  Bottom layer:" << std::endl;
    std::cout << fmt::format("    Max nbr size:         {}", bl_max_nbr_size) << std::endl;
    std::cout << fmt::format("    Reserved nbr size:    {}", bl_reserved_nbr_size) << std::endl;
    std::cout << fmt::format("    Scale coeffs:         {}", program.get<float>("--bl-scale-coeffs")) << std::endl;
    std::cout << fmt::format("    Shifted coeffs:       {}", program.get<float>("--bl-shifted-coeffs")) << std::endl;
    std::cout << fmt::format("    Num outer iters:      {}", program.get<uint32_t>("--bl-num-outer-iters")) << std::endl;
    std::cout << fmt::format("    Num inner iters:      {}", program.get<uint32_t>("--bl-num-inner-iters")) << std::endl;
    std::cout << "  Upper layers:" << std::endl;
    std::cout << fmt::format("    Max nbr size:         {}", ul_max_nbr_size) << std::endl;
    std::cout << fmt::format("    Reserved nbr size:    {}", ul_reserved_nbr_size) << std::endl;
    std::cout << fmt::format("    Scale coeffs:         {}", program.get<float>("--ul-scale-coeffs")) << std::endl;
    std::cout << fmt::format("    Shifted coeffs:       {}", program.get<float>("--ul-shifted-coeffs")) << std::endl;
    std::cout << fmt::format("    Num outer iters:      {}", program.get<uint32_t>("--ul-num-outer-iters")) << std::endl;
    std::cout << fmt::format("    Num inner iters:      {}", program.get<uint32_t>("--ul-num-inner-iters")) << std::endl;
    std::cout << "--- Vertices Builder Config ---" << std::endl;
    std::cout << fmt::format("  Min radius:             {:.6f} (auto-probed)", min_radius) << std::endl;
    std::cout << fmt::format("  Beta:                   {}", program.get<float>("--beta")) << std::endl;
    std::cout << fmt::format("  Coverage ratio:         {}", program.get<float>("--coverage-ratio")) << std::endl;
    std::cout << fmt::format("  Confidence:             {}", program.get<float>("--confidence")) << std::endl;
    std::cout << fmt::format("  Max result ratio:       {}", program.get<float>("--max-result-ratio")) << std::endl;
    std::cout << fmt::format("  Sampling batch size:    {}", program.get<uint32_t>("--sampling-batch-size")) << std::endl;
    std::cout << "\n--- Construction Time ---" << std::endl;
    std::cout << fmt::format("  Vertices construction:  {:.2f} s", vertices_duration.count() / 1000.0) << std::endl;
    std::cout << fmt::format("  Edges construction:     {:.2f} s", edges_duration.count() / 1000.0) << std::endl;
    std::cout << fmt::format("  Total time:             {:.2f} s", total_time_s) << std::endl;
    std::cout << "\n--- Output ---" << std::endl;
    std::cout << fmt::format("  Index path:             {}", output_path.string()) << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;

    logger.info(fmt::format("Saving hierarchical graph to {}...", output_path.string()));
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

    logger.info("Graph saved successfully");

    // Update index registry
    std::filesystem::path registry_path = std::filesystem::path(output_dir) / "index_registry.json";

    nlohmann::json index_params;
    index_params["dataset"] = dataset_name;
    index_params["min_radius"] = min_radius;
    index_params["beta"] = std::round(program.get<float>("--beta") * 100.0f) / 100.0f;
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
