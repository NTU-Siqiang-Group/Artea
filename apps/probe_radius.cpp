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
 * @FilePath: /Artea/apps/probe_radius.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Probe distance distribution quantiles from dataset
 */

#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/default_context.hpp>
#include <iostream>
#include <iomanip>

using namespace artea;
using namespace artea::cpu;
using namespace artea::cpu::default_context;

int main(int argc, char** argv) {
    argparse::ArgumentParser program("probe_radius");
    program.add_description("Probe distance distribution quantiles from dataset");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(std::string("./configs/datasets.json"))
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    // Probing parameters
    program.add_argument("-q", "--quantile")
        .default_value(0.001f)
        .scan<'g', float>()
        .help("Target quantile (e.g., 0.001 for 0.1%, 0.01 for 1%, 0.05 for 5%)");

    program.add_argument("-n", "--num-samples")
        .default_value(uint32_t(1152))
        .scan<'u', uint32_t>()
        .help("Number of vectors to sample from the dataset");

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
    float quantile = program.get<float>("--quantile");
    vertex_num_t num_samples = program.get<uint32_t>("--num-samples");

    logger.info(fmt::format("Radius Probing Configuration:"));
    logger.info(fmt::format("  Dataset: {}", dataset_name));
    logger.info(fmt::format("  Config path: {}", config_path));
    logger.info(fmt::format("  Quantile: {:.4f} ({:.2f}%)", quantile, quantile * 100));
    logger.info(fmt::format("  Number of samples: {}", num_samples));

    // Load dataset
    logger.info("Loading dataset...");
    vector_dataset_t dataset(config_path, dataset_name);

    vec_dim_t dim = dataset.get_vec_dim();
    vertex_num_t num_base_vecs = dataset.get_num_base_vecs();

    logger.info(fmt::format("Dataset loaded:"));
    logger.info(fmt::format("  Dimension: {}", dim));
    logger.info(fmt::format("  Base vectors: {}", num_base_vecs));

    // Create distance function
    dist_func_t dist_func(dim);

    // Create radius prober
    radius_prober_t prober(dist_func);

    // Probe radius
    logger.info("Probing radius...");
    auto result = prober.probe(dataset.get_base_vecs(), quantile, num_samples);

    // Display results
    logger.info(fmt::format("Probing Results:"));
    logger.info(fmt::format("  Quantile: {:.4f} ({:.2f}%)", result.quantile, result.quantile * 100));
    logger.info(fmt::format("  Radius: {:.6f}", result.radius));
    logger.info(fmt::format("  Vectors sampled: {}", result.num_vecs_sampled));
    logger.info(fmt::format("  Distances computed: {}", result.num_distances_computed));

    return 0;
}
