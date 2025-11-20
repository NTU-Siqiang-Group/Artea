/*
 * @FilePath: /Artea/tests/test_kmeans_partition.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Benchmark for KmeansPartition with different sampling ratios,
 *               analyzing both performance and clustering quality.
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <cmath>

#include <fmt/format.h>

#include <artea/cpu/vector_dataset.hpp>
#include <artea/cpu/kmeans_partition.hpp>
#include <artea/definitions.hpp>
#include <artea/logger.hpp>

using namespace artea;
using namespace artea::cpu;

// Structure to hold results for a single trial
struct BenchResult {
    double fit_time_ms;
    double eval_time_ms;
    double part_time_ms;
    double inertia;
    double silhouette;
    double imbalance_stddev;
};

// Helper to calculate mean of a specific field in BenchResult
template <typename Func>
double calculate_mean(const std::vector<BenchResult>& results, Func extractor) {
    double sum = 0.0;
    for (const auto& res : results) {
        sum += extractor(res);
    }
    return sum / results.size();
}

int main() {
    // NOTE: Adjust these paths according to your environment
    const std::string config_path = "/home/yeweitang/Artea/datasets.json";
    const std::string dataset_name = "sift-1m";

    logger.info("Starting KmeansPartition Benchmark & Quality Analysis...");

    try {
        // 1. Load Dataset
        logger.info("Loading dataset...");
        VectorDataset<uint32_t, float> dataset(config_path, dataset_name);
        const auto* base_vecs = dataset.get_base_vecs();

        uint32_t num_vecs = base_vecs->get_num_vecs();
        uint32_t dim = base_vecs->get_vec_dim();

        // Benchmark Configuration
        const uint32_t num_clusters = 1024; // Target IVF1024
        const int num_trials = 5;
        // Compare small sampling vs full dataset
        const std::vector<float> ratios = {0.05f, 0.1f, 1.0f};

        logger.info(fmt::format("Dataset: {} vectors, {} dims. Target Clusters: {}", num_vecs, dim, num_clusters));

        // 2. Run Benchmarks
        for (float ratio : ratios) {
            logger.info(fmt::format(">>> Testing Sampling Ratio: {:.1f}% ({}) <<<",
                ratio * 100.0f, static_cast<uint32_t>(num_vecs * ratio)));

            std::vector<BenchResult> trial_results;
            trial_results.reserve(num_trials);

            for (int i = 0; i < num_trials; ++i) {
                // Reset Kmeans object each time to ensure clean state
                KmeansPartition<uint32_t, float> kmeans(num_vecs, num_clusters);

                // A. Measure Fit (Training)
                auto t1 = std::chrono::high_resolution_clock::now();
                kmeans.fit(*base_vecs, ratio);
                auto t2 = std::chrono::high_resolution_clock::now();

                // B. Measure Evaluation (Quality Analysis)
                // Note: This scans the whole dataset to compute metrics
                auto t3 = std::chrono::high_resolution_clock::now();
                auto metrics = kmeans.evaluate(*base_vecs);
                auto t4 = std::chrono::high_resolution_clock::now();

                // C. Measure Partition & Reorder (Throughput)
                auto t5 = std::chrono::high_resolution_clock::now();
                auto [reordered, offsets] = kmeans.partition_and_reorder(*base_vecs);
                auto t6 = std::chrono::high_resolution_clock::now();

                // Record Metrics
                BenchResult res;
                res.fit_time_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                res.eval_time_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
                res.part_time_ms = std::chrono::duration<double, std::milli>(t6 - t5).count();
                res.inertia = metrics.inertia;
                res.silhouette = metrics.silhouette_score;
                res.imbalance_stddev = metrics.part_sizes_stddev;

                trial_results.push_back(res);

                logger.debug(fmt::format(
                    "Trial {}: Fit={:.1f}ms, Part={:.1f}ms | Sil={:.4f}, Imbal={:.2f}",
                    i + 1, res.fit_time_ms, res.part_time_ms, res.silhouette, res.imbalance_stddev
                ));
            }

            // 3. Aggregate and Report
            double avg_fit = calculate_mean(trial_results, [](const BenchResult& r){ return r.fit_time_ms; });
            double avg_eval = calculate_mean(trial_results, [](const BenchResult& r){ return r.eval_time_ms; });
            double avg_part = calculate_mean(trial_results, [](const BenchResult& r){ return r.part_time_ms; });
            double avg_sil = calculate_mean(trial_results, [](const BenchResult& r){ return r.silhouette; });
            double avg_imbal = calculate_mean(trial_results, [](const BenchResult& r){ return r.imbalance_stddev; });
            double avg_inertia = calculate_mean(trial_results, [](const BenchResult& r){ return r.inertia; });

            std::string report = fmt::format(
                "Ratio {:.1f}% Summary:\n"
                "  [Performance]\n"
                "    Avg Fit Time:       {:.2f} ms\n"
                "    Avg Partition Time: {:.2f} ms\n"
                "    Avg Evaluate Time:  {:.2f} ms\n"
                "  [Quality]\n"
                "    Silhouette Score:   {:.4f} (Higher is better)\n"
                "    Load Imbalance:     {:.2f} (StdDev, Lower is better)\n"
                "    Inertia (SSE):      {:.2e}",
                ratio * 100.0f, avg_fit, avg_part, avg_eval, avg_sil, avg_imbal, avg_inertia
            );

            logger.success(report);
        }

    } catch (const std::exception& e) {
        logger.error(fmt::format("Benchmark failed: {}", e.what()));
        return 1;
    }

    return 0;
}