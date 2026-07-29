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
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace artea;
using namespace artea::cpu;

int main(int argc, char** argv) {
    argparse::ArgumentParser program("probe_radius");
    program.add_description("Probe distance distribution quantiles from dataset");

    // Dataset configuration
    program.add_argument("-c", "--config")
        .default_value(artea::default_dataset_config_path())
        .help("Path to dataset configuration file");

    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");

    program.add_argument("--metric")
        .default_value(std::string("euclidean"))
        .help("Distance metric: 'euclidean', 'inner_product', or 'cosine'");

    // Probing parameters
    program.add_argument("-q", "--quantile")
        .scan<'g', float>()
        .help("Target quantile (e.g., 0.0005 for 0.05%%, 0.001 for 0.1%%, 0.01 for 1%%). If not specified, probes multiple quantiles.");

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
        .default_value(0.05f)
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
    std::string metric_str = program.get<std::string>("--metric");
    bool has_quantile = program.is_used("--quantile");
    float quantile = has_quantile ? program.get<float>("--quantile") : 0.0f;

    // Check which mode is being used
    bool has_num_distances = program.is_used("--num-distances");
    bool has_confidence = program.is_used("--confidence");
    bool has_relative_err = program.is_used("--relative-err");

    // Validate parameter combinations
    if (has_quantile && has_num_distances && (has_confidence || has_relative_err)) {
        ARTEA_ERROR("Cannot specify both --num-distances and (--confidence/--relative-err)");
    }

    if (has_quantile && ((has_confidence && !has_relative_err) || (!has_confidence && has_relative_err))) {
        ARTEA_ERROR("--confidence and --relative-err must be specified together");
    }

    // Load dataset (metric/dim-independent).
    ARTEA_INFO("Loading dataset...");
    vector_dataset_t dataset(config_path, dataset_name);

    vec_dim_t dim = dataset.get_vec_dim();
    vertex_num_t num_base_vecs = dataset.get_num_base_vecs();

    ARTEA_INFO(fmt::format("Dataset loaded:"));
    ARTEA_INFO(fmt::format("  Dimension: {}", dim));
    ARTEA_INFO(fmt::format("  Base vectors: {}", num_base_vecs));

    // Resolve both compile-time axes: metric from the --metric input, padded
    // dim from the loaded dataset.
    const auto dataset_info = DatasetInfra{parse_metric(metric_str), dim};

    // Bridge runtime (metric, dim) -> compile-time <Metric, Dim>; the distance
    // function and DistanceProber are metric/dim-dependent.
    return infra_dispatch(dataset_info, ARTEA_METRIC_LAMBDA(int) {
    // Create distance function (stateless: dim is a compile-time trait now)
    dist_func_t<Metric, Dim> dist_func;

    // Create radius prober
    distance_prober_t<Metric, Dim> prober(dist_func);

    // Check if multi-quantile mode
    if (!has_quantile) {
        // Multi-quantile mode
        float confidence = program.get<float>("--confidence");
        float relative_err = program.get<float>("--relative-err");

        ARTEA_INFO("Multi-Quantile Probing Configuration:");
        ARTEA_INFO(fmt::format("  Dataset: {}", dataset_name));
        ARTEA_INFO(fmt::format("  Config path: {}", config_path));
        ARTEA_INFO(fmt::format("  Confidence: {:.2f}%", confidence * 100));
        ARTEA_INFO(fmt::format("  Relative error: {:.2f}%", relative_err * 100));

        // Probe multiple quantiles
        ARTEA_INFO("Probing multiple quantiles...");
        auto start_time = std::chrono::high_resolution_clock::now();

        auto result = prober.probe_multi_quantiles(dataset.get_base_vecs(), confidence, relative_err);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Display results
        ARTEA_INFO(fmt::format("Probing Results:"));
        ARTEA_INFO(fmt::format("  Distance samples: {}", result.num_dists_sampled));
        ARTEA_INFO(fmt::format("  Time elapsed: {:.3f} seconds", duration.count() / 1000.0));

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "                    MULTI-QUANTILE PROBING SUMMARY" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << fmt::format("\n{:<15} {:<20} {:<20}", "Quantile", "Percentage", "Radius") << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        for (size_t i = 0; i < result.quantiles.size(); ++i) {
            std::cout << fmt::format("{:<15.4f} {:<20} {:<20.6f}",
                result.quantiles[i],
                fmt::format("{:.2f}%", result.quantiles[i] * 100),
                result.radii[i]) << std::endl;
        }

        std::cout << std::string(80, '=') << std::endl;

        return 0;
    }

    // Single quantile mode (original behavior)
    // Determine which mode to use
    vertex_num_t num_distances;
    float confidence = 0.0f;
    float relative_err = 0.0f;

    if (has_num_distances) {
        // Mode 1: User specified num-distances directly
        num_distances = program.get<uint32_t>("--num-distances");
        ARTEA_INFO(fmt::format("Radius Probing Configuration:"));
        ARTEA_INFO(fmt::format("  Dataset: {}", dataset_name));
        ARTEA_INFO(fmt::format("  Config path: {}", config_path));
        ARTEA_INFO(fmt::format("  Quantile: {:.4f} ({:.2f}%)", quantile, quantile * 100));
        ARTEA_INFO(fmt::format("  Number of distance samples: {}", num_distances));
    } else if (has_confidence && has_relative_err) {
        // Mode 2: Auto-compute from confidence and relative error
        confidence = program.get<float>("--confidence");
        relative_err = program.get<float>("--relative-err");
        num_distances = distance_prober_t<Metric, Dim>::compute_num_dists_sampled(quantile, confidence, relative_err);

        ARTEA_INFO(fmt::format("Radius Probing Configuration:"));
        ARTEA_INFO(fmt::format("  Dataset: {}", dataset_name));
        ARTEA_INFO(fmt::format("  Config path: {}", config_path));
        ARTEA_INFO(fmt::format("  Quantile: {:.4f} ({:.2f}%)", quantile, quantile * 100));
        ARTEA_INFO(fmt::format("  Confidence: {:.2f}%", confidence * 100));
        ARTEA_INFO(fmt::format("  Relative error: {:.2f}%", relative_err * 100));
        ARTEA_INFO(fmt::format("  Computed distance samples: {}", num_distances));
    } else {
        // Default mode: use default confidence and relative error
        confidence = program.get<float>("--confidence");
        relative_err = program.get<float>("--relative-err");
        num_distances = distance_prober_t<Metric, Dim>::compute_num_dists_sampled(quantile, confidence, relative_err);

        ARTEA_INFO(fmt::format("Radius Probing Configuration:"));
        ARTEA_INFO(fmt::format("  Dataset: {}", dataset_name));
        ARTEA_INFO(fmt::format("  Config path: {}", config_path));
        ARTEA_INFO(fmt::format("  Quantile: {:.4f} ({:.2f}%)", quantile, quantile * 100));
        ARTEA_INFO(fmt::format("  Confidence: {:.2f}% (default)", confidence * 100));
        ARTEA_INFO(fmt::format("  Relative error: {:.2f}% (default)", relative_err * 100));
        ARTEA_INFO(fmt::format("  Computed distance samples: {}", num_distances));
    }

    // Probe radius
    ARTEA_INFO("Probing radius...");
    auto start_time = std::chrono::high_resolution_clock::now();

    auto result = prober.probe(dataset.get_base_vecs(), quantile, num_distances);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Display results
    ARTEA_INFO(fmt::format("Probing Results:"));
    ARTEA_INFO(fmt::format("  Quantile: {:.4f} ({:.2f}%)", result.quantile, result.quantile * 100));
    ARTEA_INFO(fmt::format("  Radius: {:.6f}", result.radius));
    ARTEA_INFO(fmt::format("  Distance samples: {}", result.num_dists_sampled));
    ARTEA_INFO(fmt::format("  Time elapsed: {:.3f} seconds", duration.count() / 1000.0));

    return 0;
    });
}
