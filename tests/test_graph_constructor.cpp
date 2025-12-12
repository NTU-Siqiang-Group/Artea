/*
 * @FilePath: /Artea/tests/test_graph_constructor.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-12-11 21:38:29
 * @Date: 2025-12-11 21:40:00
 * @Description: Performance test for GraphConstructor without verification.
 */

#include <iostream>
#include <string>
#include <chrono>
#include <stdexcept>

// Third-party libraries
#include <fmt/format.h>
#include <argparse/argparse.hpp>

// Artea headers
#include <artea/cpu/index/graph_constructor.hpp>
#include <artea/cpu/containers/vector_dataset.hpp>
#include <artea/definitions.hpp>
#include <artea/common/logger.hpp>

int main(int argc, char *argv[]) {
    // 1. Setup Argument Parser
    argparse::ArgumentParser program("test_graph_constructor");

    program.add_argument("--dataset")
        .default_value(std::string("sift-1m"))
        .help("Name of the dataset (e.g., sift-1m)");

    program.add_argument("--config")
        .default_value(std::string("/home/yeweitang/Artea/datasets.json"))
        .help("Path to the dataset configuration JSON file");

    program.add_argument("--iters")
        .default_value(10)
        .scan<'i', int>()
        .help("Number of iterations for propagation");

    program.add_argument("--init_k")
        .default_value(32)
        .scan<'i', int>()
        .help("Initial number of neighbors (init_nbrs_size)");

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Extract arguments
    std::string dataset_name = program.get<std::string>("--dataset");
    std::string config_path = program.get<std::string>("--config");
    int num_iters = program.get<int>("--iters");
    int init_k = program.get<int>("--init_k");

    std::cout << "Starting GraphConstructor performance test..." << std::endl;
    artea::logger.info(fmt::format("Configuration: Dataset={}, Iters={}, Init_K={}, Config={}",
                                   dataset_name, num_iters, init_k, config_path));

    try {
        using VertexT = uint32_t;
        using ElementT = float;

        // --- Step 1: Load Dataset ---
        // We need to load the dataset first as the new GraphConstructor requires a reference to it.
        artea::logger.info("Loading dataset...");
        auto load_start = std::chrono::high_resolution_clock::now();

        artea::cpu::VectorDataset<VertexT, ElementT> dataset(config_path, dataset_name);

        auto load_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> load_ms = load_end - load_start;
        artea::logger.info(fmt::format("Dataset loaded in {:.2f} ms.", load_ms.count()));
        artea::logger.info(fmt::format("Number of base vectors: {}", dataset.get_base_vecs().get_num_vecs()));

        // --- Step 2: Initialize GraphConstructor ---
        artea::logger.info("Initializing GraphConstructor...");
        artea::cpu::GraphConstructor<VertexT, ElementT> constructor(dataset);

        // --- Step 3: Construct Graph and measure time ---
        artea::logger.info("Starting graph construction...");
        auto build_start = std::chrono::high_resolution_clock::now();

        // The construct_graph method performs the heavy lifting (propagation)
        auto index_graph = constructor.construct_graph(
            static_cast<VertexT>(init_k),
            static_cast<artea::iter_t>(num_iters)
        );

        auto build_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> build_ms = build_end - build_start;
        double build_sec = build_ms.count() / 1000.0;

        // --- Step 4: Report Results ---
        artea::logger.success("Graph construction completed successfully.");
        artea::logger.info("--------------------------------------------------");
        artea::logger.info(fmt::format("Dataset:    {}", dataset_name));
        artea::logger.info(fmt::format("Iterations: {}", num_iters));
        artea::logger.info(fmt::format("Init K:     {}", init_k));
        artea::logger.info(fmt::format("Build Time: {:.2f} ms ({:.4f} s)", build_ms.count(), build_sec));
        artea::logger.info("--------------------------------------------------");

    } catch (const std::exception& e) {
        artea::logger.error(fmt::format("\nAn error occurred during execution: {}", e.what()));
        return 1;
    } catch (...) {
        artea::logger.error("\nAn unknown error occurred.");
        return 1;
    }

    return 0;
}