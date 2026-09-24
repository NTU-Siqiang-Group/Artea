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
 * @FilePath: /Artea/apps/probe_dataset.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Probe dataset nearest/farthest distance distributions, LID,
 *               query ground-truth distances, and approximate aspect ratio.
 */

#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace artea;
using namespace artea::cpu;

namespace {

// Quantiles reported by the dataset probing modes.
const std::vector<float> quantiles = {
    0.0001f, 0.001f, 0.01f, 0.025f, 0.05f, 0.1f, 0.25f,
    0.5f, 0.9f, 0.95f, 0.99f, 0.999f, 0.9999f
};

void print_header(const std::string& label, const std::vector<float>& qs) {
    std::string header = fmt::format("{:>10}", label);
    for (float q : qs) {
        header += fmt::format(" {:>12.3f}%", q * 100.0f);
    }
    std::cout << header << '\n';
}

template <typename Result>
void print_nn_table(const Result& result) {
    print_header("rank", result.quantiles);
    for (size_t r = 0; r < result.nn_ranks.size(); ++r) {
        std::cout << fmt::format("{:>10}", result.nn_ranks[r]);
        for (auto distance : result.table[r]) {
            std::cout << fmt::format(" {:>13.6f}", distance);
        }
        std::cout << '\n';
    }
}

template <typename NearResult, typename FarResult>
void print_aspect_ratio(const NearResult& nearest, const FarResult& farthest) {
    const auto median_idx = static_cast<size_t>(
        std::find(quantiles.begin(), quantiles.end(), 0.5f) - quantiles.begin());
    const float near_low = nearest.table[0][0];
    const float near_median = nearest.table[0][median_idx];
    const float far_median = farthest.farthest[median_idx];
    const auto ratio = [](float f, float n) {
        return n > 0.0f ? f / n : std::numeric_limits<float>::infinity();
    };

    // The test uses q=0.0001 as a closest-pair proxy, not the exact minimum.
    ARTEA_INFO(fmt::format("Nearest (rank=1): q=0.0001={:.6f}, median={:.6f}",
        near_low, near_median));
    ARTEA_INFO(fmt::format("Farthest: median={:.6f}, max={:.6f}",
        far_median, farthest.max_farthest));
    ARTEA_INFO(fmt::format(
        "Approximate aspect ratio (extreme = max_farthest / nearest_q0.0001): {:.6f}",
        ratio(farthest.max_farthest, near_low)));
    ARTEA_INFO(fmt::format(
        "Approximate aspect ratio (median = median_farthest / median_nearest): {:.6f}",
        ratio(far_median, near_median)));
}

} // namespace

int main(int argc, char** argv) {
    argparse::ArgumentParser program("probe_dataset");
    program.add_description(
        "Probe dataset nearest/farthest distance quantiles, LID, query ground truth, and aspect ratio.");
    program.add_argument("-c", "--config")
        .default_value(artea::default_dataset_config_path())
        .help("Path to dataset configuration file");
    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name");
    program.add_argument("--metric")
        .default_value(std::string("euclidean_sqr"))
        .help("Distance metric: euclidean (l2), euclidean_sqr (l2_sqr), inner_product, or cosine");
    program.add_argument("-n", "--num-samples")
        .default_value(1000u)
        .scan<'u', uint32_t>()
        .help("Number of base vertices sampled for nearest/farthest probes (query uses all queries)");
    program.add_argument("--mode")
        .default_value(std::string("all"))
        .nargs(1)
        .choices("all", "nearest", "query", "farthest", "aspect-ratio")
        .help("Statistics: all, nearest, query, farthest, or aspect-ratio (all reuses probe results)");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << '\n' << program;
        return 1;
    }

    try {
        const auto config_path = program.get<std::string>("--config");
        const auto dataset_name = program.get<std::string>("--dataset");
        const auto metric = parse_metric(program.get<std::string>("--metric"));
        const auto num_samples = program.get<uint32_t>("--num-samples");
        const auto mode = program.get<std::string>("--mode");
        if (num_samples == 0) {
            throw std::invalid_argument("--num-samples must be at least 1");
        }

        vector_dataset_t dataset(config_path, dataset_name);
        const auto& base_vecs = dataset.get_base_vecs();
        ARTEA_INFO(fmt::format(
            "Dataset: {}, metric: {}, dimension: {}, base vectors: {}, mode: {}, samples: {}",
            dataset_name, metric_name(metric), dataset.get_vec_dim(),
            dataset.get_num_base_vecs(), mode, num_samples));

        const auto start = std::chrono::steady_clock::now();
        infra_dispatch(DatasetInfra{metric, base_vecs.get_vec_dim()}, ARTEA_METRIC_LAMBDA(void) {
            dist_func_t<Metric, Dim> dist_func;
            using prober_t = dataset_prober_t<Metric, Dim>;
            prober_t prober(base_vecs, dist_func);
            std::optional<typename prober_t::ProbeResult> nearest;
            std::optional<typename prober_t::FarthestProbeResult> farthest;

            if (mode == "all" || mode == "nearest" || mode == "aspect-ratio") {
                ARTEA_INFO(fmt::format("Probing {} sampled vertices x 128 nearest-neighbor ranks...",
                    num_samples));
                nearest = prober.probe(quantiles, num_samples);
                ARTEA_INFO(fmt::format("Estimated LID (Levina-Bickel, k=128, metric={}): {:.6f}",
                    metric_name(metric), nearest->lid));
                if (mode != "aspect-ratio") {
                    ARTEA_INFO("Base nearest-neighbor distance quantiles:");
                    print_nn_table(*nearest);
                }
            }

            if (mode == "all" || mode == "query") {
                ARTEA_INFO("Probing query ground-truth distances...");
                const auto result = prober.probe_query(
                    dataset.get_query_vecs(), dataset.get_gt_vecs(), quantiles);
                ARTEA_INFO(fmt::format("Query distance quantiles: {} queries x {} ranks",
                    result.num_queries, result.nn_ranks.size()));
                print_nn_table(result);
            }

            if (mode == "all" || mode == "farthest" || mode == "aspect-ratio") {
                ARTEA_INFO(fmt::format("Probing {} sampled vertices for farthest distances...",
                    num_samples));
                farthest = prober.probe_farthest(quantiles, num_samples);
                ARTEA_INFO(fmt::format("Per-sample farthest distance: min={:.6f}, max={:.6f}",
                    farthest->min_farthest, farthest->max_farthest));
                if (mode != "aspect-ratio") {
                    print_header("stat", farthest->quantiles);
                    std::cout << fmt::format("{:>10}", "farthest");
                    for (auto distance : farthest->farthest) {
                        std::cout << fmt::format(" {:>13.6f}", distance);
                    }
                    std::cout << '\n';
                }
            }

            if (mode == "all" || mode == "aspect-ratio") {
                print_aspect_ratio(*nearest, *farthest);
            }
        });
        const double elapsed_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        ARTEA_INFO(fmt::format("Dataset probing completed in {:.3f} seconds", elapsed_s));
        return 0;
    } catch (const std::exception& err) {
        std::cerr << "probe_dataset: " << err.what() << '\n';
        return 1;
    }
}
