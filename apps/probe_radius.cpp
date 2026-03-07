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
#include <chrono>

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
        .default_value(0.0005f)
        .scan<'g', float>()
        .help("Target quantile (e.g., 0.0005 for 0.05%, 0.001 for 0.1%, 0.01 for 1%)");

    // Option 1: Specify number of distance samples directly
    program.add_argument("-m", "--num-distances")
        .scan<'u', uint32_t>()
        .help("Number of independent distance samples (overrides confidence/relative-err)");

    // Option 2: Specify confidence and relative error (auto-compute num-distances)
    program.add_argument("--confidence")
        .default_value(0.99f)
        .scan<'g', float>()
        .help("Confidence level (e.g., 0.95 for 95%, 0.99 for 99%)");

    program.add_argument("--relative-err")
        .default_value(0.1f)
        .scan<'g', float>()
        .help("Relative error (e.g., 0.1 for 10%, 0.2 for 20%)");

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

    // Check which mode is being used
    bool has_num_distances = program.is_used("--num-distances");
    bool has_confidence = program.is_used("--confidence");
    bool has_relative_err = program.is_used("--relative-err");

    // Validate parameter combinations
    if (has_num_distances && (has_confidence || has_relative_err)) {
        logger.error("Cannot specify both --num-distances and (--confidence/--relative-err)");
        logger.error("Use either:");
        logger.error("  Option 1: -m/--num-distances (manual)");
        logger.error("  Option 2: --confidence and --relative-err (auto-compute)");
        return 1;
    }

    if ((has_confidence && !has_relative_err) || (!has_confidence && has_relative_err)) {
        logger.error("--confidence and --relative-err must be specified together");
        return 1;
    }

    // Determine which mode to use
    vertex_num_t num_distances;
    float confidence = 0.0f;
    float relative_err = 0.0f;

    if (has_num_distances) {
        // Mode 1: User specified num-distances directly
        num_distances = program.get<uint32_t>("--num-distances");
        logger.info(fmt::format("Radius Probing Configuration:"));
        logger.info(fmt::format("  Dataset: {}", dataset_name));
        logger.info(fmt::format("  Config path: {}", config_path));
        logger.info(fmt::format("  Quantile: {:.4f} ({:.2f}%)", quantile, quantile * 100));
        logger.info(fmt::format("  Number of distance samples: {}", num_distances));
    } else if (has_confidence && has_relative_err) {
        // Mode 2: Auto-compute from confidence and relative error
        confidence = program.get<float>("--confidence");
        relative_err = program.get<float>("--relative-err");
        num_distances = radius_prober_t::compute_num_dists_sampled(quantile, confidence, relative_err);

        logger.info(fmt::format("Radius Probing Configuration:"));
        logger.info(fmt::format("  Dataset: {}", dataset_name));
        logger.info(fmt::format("  Config path: {}", config_path));
        logger.info(fmt::format("  Quantile: {:.4f} ({:.2f}%)", quantile, quantile * 100));
        logger.info(fmt::format("  Confidence: {:.2f}%", confidence * 100));
        logger.info(fmt::format("  Relative error: {:.2f}%", relative_err * 100));
        logger.info(fmt::format("  Computed distance samples: {}", num_distances));
    } else {
        // Default mode: use default confidence and relative error
        confidence = program.get<float>("--confidence");
        relative_err = program.get<float>("--relative-err");
        num_distances = radius_prober_t::compute_num_dists_sampled(quantile, confidence, relative_err);

        logger.info(fmt::format("Radius Probing Configuration:"));
        logger.info(fmt::format("  Dataset: {}", dataset_name));
        logger.info(fmt::format("  Config path: {}", config_path));
        logger.info(fmt::format("  Quantile: {:.4f} ({:.2f}%)", quantile, quantile * 100));
        logger.info(fmt::format("  Confidence: {:.2f}% (default)", confidence * 100));
        logger.info(fmt::format("  Relative error: {:.2f}% (default)", relative_err * 100));
        logger.info(fmt::format("  Computed distance samples: {}", num_distances));
    }

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
    auto start_time = std::chrono::high_resolution_clock::now();

    auto result = prober.probe(dataset.get_base_vecs(), quantile, num_distances);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Display results
    logger.info(fmt::format("Probing Results:"));
    logger.info(fmt::format("  Quantile: {:.4f} ({:.2f}%)", result.quantile, result.quantile * 100));
    logger.info(fmt::format("  Radius: {:.6f}", result.radius));
    logger.info(fmt::format("  Distance samples: {}", result.num_dists_sampled));
    logger.info(fmt::format("  Time elapsed: {:.3f} seconds", duration.count() / 1000.0));

    return 0;
}
