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
#include <cmath>
#include <limits>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>

using namespace artea;
using namespace artea::cpu;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    std::string metric;
    uint32_t num_samples;
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

        // Resolve BOTH compile-time axes: metric from the --metric input, padded
        // dim from the loaded dataset; the dataset stays metric/dim-independent
        // here, and the stateless dist_func + dataset_prober_t are rebuilt inside
        // each dispatched body.
        const auto& base_vecs = dataset_->get_base_vecs();
        dataset_info_ = DatasetInfra{parse_metric(g_config.metric), base_vecs.get_vec_dim()};
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    DatasetInfra get_dataset_info() const { return dataset_info_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    DatasetInfra dataset_info_{};
};

class DatasetProberTest : public ::testing::Test {};

/**
 * @brief Probe full table: nn_rank 1..128 x all quantiles, plus LID estimate.
 *        Print in chunks of at most 10 nn_rank columns.
 */
TEST_F(DatasetProberTest, Probe) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();

    const auto& base_vecs = dataset.get_base_vecs();

    const vec_num_t num_samples = g_config.num_samples;

    std::vector<float> quantiles = {
        0.0001f, 0.001f, 0.01f, 0.05f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.95f, 0.99f, 0.999f, 0.9999f
    };

    ARTEA_INFO(fmt::format("Probing dataset with {} samples...", num_samples));

    infra_dispatch(provider.get_dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        // Stateless functor: the dimension is a compile-time trait now.
        dist_func_t<Metric, Dim> dist_func;
        dataset_prober_t<Metric, Dim> prober(base_vecs, dist_func);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = prober.probe(quantiles, num_samples);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

        ARTEA_INFO(fmt::format("Probing completed in {:.2f} s ({} samples x 128 ranks)", elapsed_s, num_samples));
        ARTEA_INFO(fmt::format("Estimated LID (Levina-Bickel, k=128): {:.4f}", result.lid));

        // Print table: one row per nn_rank, one column per quantile.
        const uint32_t total_ranks = static_cast<uint32_t>(result.nn_ranks.size());

        // Header: rank label + quantile columns
        std::string header = fmt::format("  {:>6s}", "rank");
        for (size_t qi = 0; qi < result.quantiles.size(); ++qi) {
            header += fmt::format(" {:>9.3f}%", result.quantiles[qi] * 100.0f);
        }
        ARTEA_INFO(header);

        // Data rows: one per rank
        for (uint32_t r = 0; r < total_ranks; ++r) {
            std::string row = fmt::format("  {:>6d}", result.nn_ranks[r]);
            for (size_t qi = 0; qi < result.quantiles.size(); ++qi) {
                row += fmt::format(" {:>10.2f}", result.table[r][qi]);
            }
            ARTEA_INFO(row);
        }

        // Verify LID is positive and reasonable
        EXPECT_GT(result.lid, 0.0f) << "LID should be positive";

        // Verify: for each quantile, radii should be non-decreasing across nn_ranks
        for (size_t qi = 0; qi < result.quantiles.size(); ++qi) {
            for (uint32_t r = 1; r < total_ranks; ++r) {
                EXPECT_LE(result.table[r - 1][qi], result.table[r][qi])
                    << fmt::format("rank ordering violated at quantile={:.3f}: rank {}={:.6f} > rank {}={:.6f}",
                        result.quantiles[qi], r, result.table[r - 1][qi], r + 1, result.table[r][qi]);
            }
        }

        // Verify: for each nn_rank, radii should be non-decreasing across quantiles
        for (uint32_t r = 0; r < total_ranks; ++r) {
            for (size_t qi = 1; qi < result.quantiles.size(); ++qi) {
                EXPECT_LE(result.table[r][qi - 1], result.table[r][qi])
                    << fmt::format("quantile ordering violated at rank {}: q={:.3f} ({:.6f}) > q={:.3f} ({:.6f})",
                        r + 1, result.quantiles[qi - 1], result.table[r][qi - 1],
                        result.quantiles[qi], result.table[r][qi]);
            }
        }
    });
}

/**
 * @brief Probe query set using ground truth: nn_rank 1..k x all quantiles.
 *        For each query, compute distances to its top-k ground truth IDs
 *        with _dist_func and print the column-wise quantiles.
 */
TEST_F(DatasetProberTest, ProbeQuery) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();

    const auto& base_vecs = dataset.get_base_vecs();
    const auto& query_vecs = dataset.get_query_vecs();
    const auto& gt_vecs = dataset.get_gt_vecs();

    std::vector<float> quantiles = {
        0.0001f, 0.001f, 0.01f, 0.05f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.95f, 0.99f, 0.999f, 0.9999f
    };

    ARTEA_INFO(fmt::format(
        "Probing query set: {} queries x {} ground-truth NNs per query...",
        query_vecs.get_num_vecs(), gt_vecs.get_vec_dim()));

    infra_dispatch(provider.get_dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        // Stateless functor: the dimension is a compile-time trait now.
        dist_func_t<Metric, Dim> dist_func;
        dataset_prober_t<Metric, Dim> prober(base_vecs, dist_func);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = prober.probe_query(query_vecs, gt_vecs, quantiles);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

        ARTEA_INFO(fmt::format(
            "Query probe completed in {:.2f} s ({} queries x {} ranks)",
            elapsed_s, result.num_queries, result.nn_ranks.size()));

        // Print table: one row per nn_rank, one column per quantile.
        const uint32_t total_ranks = static_cast<uint32_t>(result.nn_ranks.size());

        // Header: rank label + quantile columns
        std::string header = fmt::format("  {:>6s}", "rank");
        for (size_t qi = 0; qi < result.quantiles.size(); ++qi) {
            header += fmt::format(" {:>9.3f}%", result.quantiles[qi] * 100.0f);
        }
        ARTEA_INFO(header);

        // Data rows: one per rank
        for (uint32_t r = 0; r < total_ranks; ++r) {
            std::string row = fmt::format("  {:>6d}", result.nn_ranks[r]);
            for (size_t qi = 0; qi < result.quantiles.size(); ++qi) {
                row += fmt::format(" {:>10.2f}", result.table[r][qi]);
            }
            ARTEA_INFO(row);
        }

        // Verify: shape is consistent
        EXPECT_EQ(result.nn_ranks.size(), gt_vecs.get_vec_dim());
        EXPECT_EQ(result.table.size(), gt_vecs.get_vec_dim());
        EXPECT_EQ(result.num_queries, query_vecs.get_num_vecs());

        // Verify: for each quantile, distances should be non-decreasing across nn_ranks
        for (size_t qi = 0; qi < result.quantiles.size(); ++qi) {
            for (uint32_t r = 1; r < total_ranks; ++r) {
                EXPECT_LE(result.table[r - 1][qi], result.table[r][qi])
                    << fmt::format("rank ordering violated at quantile={:.3f}: rank {}={:.6f} > rank {}={:.6f}",
                        result.quantiles[qi], r, result.table[r - 1][qi], r + 1, result.table[r][qi]);
            }
        }

        // Verify: for each nn_rank, distances should be non-decreasing across quantiles
        for (uint32_t r = 0; r < total_ranks; ++r) {
            for (size_t qi = 1; qi < result.quantiles.size(); ++qi) {
                EXPECT_LE(result.table[r][qi - 1], result.table[r][qi])
                    << fmt::format("quantile ordering violated at rank {}: q={:.3f} ({:.6f}) > q={:.3f} ({:.6f})",
                        r + 1, result.quantiles[qi - 1], result.table[r][qi - 1],
                        result.quantiles[qi], result.table[r][qi]);
            }
        }
    });
}

/**
 * @brief Probe the per-sample farthest distance distribution.
 *        For each sampled vertex, scan the full base set and record the single
 *        farthest distance; then report quantiles across samples plus the
 *        global min/max. Characterises the dataset's outer scale (diameter),
 *        complementing the near-NN table from the Probe test.
 */
TEST_F(DatasetProberTest, ProbeFarthest) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();

    const auto& base_vecs = dataset.get_base_vecs();

    const vec_num_t num_samples = g_config.num_samples;

    std::vector<float> quantiles = {
        0.0001f, 0.001f, 0.01f, 0.05f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 0.95f, 0.99f, 0.999f, 0.9999f
    };

    ARTEA_INFO(fmt::format("Probing farthest distance with {} samples...", num_samples));

    infra_dispatch(provider.get_dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        // Stateless functor: the dimension is a compile-time trait now.
        dist_func_t<Metric, Dim> dist_func;
        dataset_prober_t<Metric, Dim> prober(base_vecs, dist_func);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = prober.probe_farthest(quantiles, num_samples);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

        ARTEA_INFO(fmt::format("Farthest probe completed in {:.2f} s ({} samples x full scan)",
            elapsed_s, num_samples));
        ARTEA_INFO(fmt::format("Per-sample farthest distance: min={:.2f}  max={:.2f}",
            result.min_farthest, result.max_farthest));

        // Print a one-row quantile table for the per-sample farthest distance.
        std::string header = fmt::format("  {:>10s}", "stat");
        for (size_t qi = 0; qi < result.quantiles.size(); ++qi) {
            header += fmt::format(" {:>9.3f}%", result.quantiles[qi] * 100.0f);
        }
        ARTEA_INFO(header);

        std::string row = fmt::format("  {:>10s}", "farthest");
        for (size_t qi = 0; qi < result.farthest.size(); ++qi) {
            row += fmt::format(" {:>10.2f}", result.farthest[qi]);
        }
        ARTEA_INFO(row);

        // Verify: farthest distances are positive and min <= max.
        EXPECT_GT(result.max_farthest, 0.0f) << "Farthest distance should be positive";
        EXPECT_LE(result.min_farthest, result.max_farthest);

        // Verify: quantiles are non-decreasing across the sorted per-sample maxima.
        for (size_t qi = 1; qi < result.farthest.size(); ++qi) {
            EXPECT_LE(result.farthest[qi - 1], result.farthest[qi])
                << fmt::format("farthest quantile ordering violated: q={:.3f} ({:.6f}) > q={:.3f} ({:.6f})",
                    result.quantiles[qi - 1], result.farthest[qi - 1],
                    result.quantiles[qi], result.farthest[qi]);
        }
    });
}

/**
 * @brief Report the dataset's approximate aspect ratio = farthest / nearest.
 *        "Nearest" is the 1-NN (nn_rank=1) distance probed over @c num_samples
 *        sampled vertices; "farthest" is the per-sample farthest distance from
 *        a full scan (probe_farthest). Two ratios are reported:
 *          - extreme: max farthest / min nearest  (≈ diameter / closest-pair,
 *            the classic aspect ratio),
 *          - median : median farthest / median nearest  (robust to outliers /
 *            duplicate points whose 1-NN distance is 0).
 *        Distances are in the prober's own metric (e.g. squared L2); the ratio
 *        is reported in those same units.
 */
TEST_F(DatasetProberTest, AspectRatio) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();

    const auto& base_vecs = dataset.get_base_vecs();

    const vec_num_t num_samples = g_config.num_samples;

    // Extremes (for diameter / closest-pair) plus the median (robust ratio).
    std::vector<float> quantiles = {0.0001f, 0.5f, 0.9999f};

    ARTEA_INFO(fmt::format("Probing approximate aspect ratio with {} samples...", num_samples));

    infra_dispatch(provider.get_dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        // Stateless functor: the dimension is a compile-time trait now.
        dist_func_t<Metric, Dim> dist_func;
        dataset_prober_t<Metric, Dim> prober(base_vecs, dist_func);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto near_res = prober.probe(quantiles, num_samples);            // nn_rank=1 → nearest
        auto far_res  = prober.probe_farthest(quantiles, num_samples);   // per-sample farthest
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;
        ARTEA_INFO(fmt::format("Aspect-ratio probe completed in {:.2f} s", elapsed_s));

        // Index of a given quantile in a result's echoed quantile list.
        auto qidx = [](const std::vector<float>& qs, float target) -> size_t {
            for (size_t i = 0; i < qs.size(); ++i) {
                if (std::abs(qs[i] - target) < 1e-9f) return i;
            }
            return 0;
        };

        ASSERT_FALSE(near_res.nn_ranks.empty()) << "probe returned no nn_ranks";
        const size_t near_lo  = qidx(near_res.quantiles, 0.0001f);  // ≈ min 1-NN (closest pair)
        const size_t near_med = qidx(near_res.quantiles, 0.5f);
        const float near_min = near_res.table[0][near_lo];          // table[0] is nn_rank=1
        const float near_p50 = near_res.table[0][near_med];

        const size_t far_med = qidx(far_res.quantiles, 0.5f);
        const float far_p50  = far_res.farthest[far_med];
        const float far_max  = far_res.max_farthest;

        auto safe_ratio = [](float f, float n) -> float {
            return n > 0.0f ? f / n : std::numeric_limits<float>::infinity();
        };
        const float ar_extreme = safe_ratio(far_max, near_min);
        const float ar_median  = safe_ratio(far_p50, near_p50);

        ARTEA_INFO(fmt::format("Nearest (nn=1):  min={:.2f}  median={:.2f}", near_min, near_p50));
        ARTEA_INFO(fmt::format("Farthest:        median={:.2f}  max={:.2f}", far_p50, far_max));
        ARTEA_INFO(fmt::format(
            "Approximate aspect ratio (extreme = max_farthest / min_nearest): {:.2f}", ar_extreme));
        ARTEA_INFO(fmt::format(
            "Approximate aspect ratio (median  = med_farthest / med_nearest): {:.2f}", ar_median));

        // Sanity: farthest must dominate nearest, and the median ratio is finite.
        EXPECT_GT(far_max, 0.0f) << "Farthest distance should be positive";
        EXPECT_GE(far_max, near_min) << "max farthest must be >= min nearest";
        EXPECT_GT(near_p50, 0.0f) << "median 1-NN distance should be positive";
        EXPECT_GE(ar_median, 1.0f) << "aspect ratio should be >= 1";
    });
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_dataset_prober");
    program.add_argument("-c", "--config").default_value(artea::default_dataset_config_path());
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("--metric").default_value(std::string("euclidean"))
        .help("Distance metric: 'euclidean', 'inner_product', or 'cosine'");
    program.add_argument("--num-samples").default_value(1000u).scan<'u', uint32_t>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.metric = program.get<std::string>("--metric");
    g_config.num_samples = program.get<uint32_t>("--num-samples");

    DataProvider::instance().init();

    return RUN_ALL_TESTS();
}
