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

#include <iostream>
#include <vector>
#include <memory>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    uint32_t num_samples;
    float confidence;
    float relative_err;
} g_config;

class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }
        ARTEA_INFO(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset_ = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dist_func_ = std::make_unique<dist_func_t>(dataset_->get_base_vecs().get_vec_dim());
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    dist_func_t& get_dist_func() { return *dist_func_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<dist_func_t> dist_func_;
};

class DistanceProberTest : public ::testing::Test {};

/**
 * @brief Sample pairwise distances and print quantiles from the minimum up to 90%.
 *
 * Samples num_samples independent vector pairs, computes their distances,
 * sorts them, and prints quantile values including very small ones
 * (0.0001%, 0.001%, ...) and the absolute minimum.
 */
TEST_F(DistanceProberTest, ProbeMultiQuantiles) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();

    const auto& base_vecs = dataset.get_base_vecs();
    const vec_num_t total_vecs = base_vecs.get_num_vecs();
    const vec_num_t num_pairs = g_config.num_samples;

    ARTEA_INFO(fmt::format("Sampling {} independent vector pairs from {} vectors...", num_pairs, total_vecs));

    // Sample two independent sets of random vector IDs
    random_seq_t rand_gen;
    std::vector<vec_id_t> ids_1(num_pairs), ids_2(num_pairs);

    tbb::parallel_for(
        tbb::blocked_range<vec_num_t>(0, num_pairs),
        [&](const tbb::blocked_range<vec_num_t>& r) {
            rand_gen.generate(ids_1.data() + r.begin(), total_vecs, r.size());
        }
    );
    tbb::parallel_for(
        tbb::blocked_range<vec_num_t>(0, num_pairs),
        [&](const tbb::blocked_range<vec_num_t>& r) {
            rand_gen.generate(ids_2.data() + r.begin(), total_vecs, r.size());
        }
    );

    // Compute pairwise distances
    std::vector<distance_t> distances(num_pairs);
    auto t0 = std::chrono::high_resolution_clock::now();

    tbb::parallel_for(
        tbb::blocked_range<vec_num_t>(0, num_pairs),
        [&](const tbb::blocked_range<vec_num_t>& r) {
            for (vec_num_t i = r.begin(); i != r.end(); ++i) {
                distances[i] = dist_func(base_vecs.get(ids_1[i]), base_vecs.get(ids_2[i]));
            }
        }
    );

    // Sort distances
    std::sort(distances.begin(), distances.end());

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    ARTEA_INFO(fmt::format("Distance computation + sort completed in {:.2f} ms", elapsed_ms));

    // Define quantiles: from very small to large, plus the minimum
    std::vector<float> quantiles = {
        0.000001f, 0.00001f, 0.0001f, 0.001f, 0.005f, 0.01f, 0.05f, 0.1f,
        0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f
    };

    // Print header
    ARTEA_INFO(fmt::format("Pairwise distance quantiles ({} samples):", num_pairs));
    ARTEA_INFO(fmt::format("  {:>12s}  {:>15s}  {:>10s}", "Quantile", "Radius", "Index"));
    ARTEA_INFO(fmt::format("  {:>12s}  {:>15s}  {:>10s}", "--------", "------", "-----"));

    // Print the absolute minimum
    ARTEA_INFO(fmt::format("  {:>12s}  {:>15.6f}  {:>10d}", "min", distances.front(), 0));

    // Print each quantile
    std::vector<distance_t> radii;
    for (float q : quantiles) {
        vec_num_t idx = static_cast<vec_num_t>(q * distances.size());
        if (idx >= distances.size()) { idx = distances.size() - 1; }
        // Skip quantiles that resolve to index 0 (same as min)
        if (idx == 0) {
            ARTEA_INFO(fmt::format("  {:>11.4f}%  {:>15.6f}  {:>10d} (= min)", q * 100.0f, distances[idx], idx));
        } else {
            ARTEA_INFO(fmt::format("  {:>11.4f}%  {:>15.6f}  {:>10d}", q * 100.0f, distances[idx], idx));
        }
        radii.push_back(distances[idx]);
    }

    // Print the absolute maximum
    ARTEA_INFO(fmt::format("  {:>12s}  {:>15.6f}  {:>10d}", "max", distances.back(), static_cast<int>(distances.size() - 1)));

    // Verify monotonically non-decreasing
    for (size_t i = 1; i < radii.size(); ++i) {
        EXPECT_LE(radii[i - 1], radii[i]);
    }
}

/**
 * @brief Probe a single quantile with explicit sample count.
 *
 * Uses the user-specified num_samples to probe the median (0.5 quantile).
 */
TEST_F(DistanceProberTest, ProbeSingleQuantile) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& dist_func = provider.get_dist_func();

    const auto& base_vecs = dataset.get_base_vecs();

    distance_prober_t prober(dist_func);

    float quantile = 0.5f;
    vec_num_t num_distances = g_config.num_samples;

    ARTEA_INFO(fmt::format("Probing single quantile {:.2f} with {} distance samples...", quantile, num_distances));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = prober.probe(base_vecs, quantile, num_distances);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    ARTEA_INFO(fmt::format("Median distance: {:.6f} (sampled {} distances in {:.2f} ms)",
        result.radius, result.num_dists_sampled, elapsed_ms));

    EXPECT_GT(result.radius, 0.0f) << "Median distance should be positive";
    EXPECT_EQ(result.num_dists_sampled, num_distances);
    EXPECT_FLOAT_EQ(result.quantile, quantile);
}

/**
 * @brief Verify compute_num_dists_sampled returns reasonable sample sizes.
 */
TEST_F(DistanceProberTest, ComputeNumDistsSampled) {
    // Smaller quantile or higher confidence should require more samples
    vec_num_t n1 = distance_prober_t::compute_num_dists_sampled(0.01f, 0.95f, 0.1f);
    vec_num_t n2 = distance_prober_t::compute_num_dists_sampled(0.001f, 0.95f, 0.1f);
    vec_num_t n3 = distance_prober_t::compute_num_dists_sampled(0.001f, 0.99f, 0.1f);

    ARTEA_INFO(fmt::format("Sample sizes: q=0.01/c=0.95/e=0.1 -> {}", n1));
    ARTEA_INFO(fmt::format("Sample sizes: q=0.001/c=0.95/e=0.1 -> {}", n2));
    ARTEA_INFO(fmt::format("Sample sizes: q=0.001/c=0.99/e=0.1 -> {}", n3));

    EXPECT_GT(n1, 0u);
    EXPECT_GT(n2, n1) << "Smaller quantile should need more samples";
    EXPECT_GT(n3, n2) << "Higher confidence should need more samples";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_distance_prober");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--num-samples").default_value(10000u).scan<'u', uint32_t>();
    program.add_argument("--confidence").default_value(0.95f).scan<'g', float>();
    program.add_argument("--relative-err").default_value(0.1f).scan<'g', float>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.num_samples = program.get<uint32_t>("--num-samples");
    g_config.confidence = program.get<float>("--confidence");
    g_config.relative_err = program.get<float>("--relative-err");

    DataProvider::instance().init();

    return RUN_ALL_TESTS();
}