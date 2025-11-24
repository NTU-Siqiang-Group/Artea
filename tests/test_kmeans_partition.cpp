/*
 * @FilePath: /Artea/tests/test_kmeans_partition.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Benchmark comparison between Inter-query vs Nested Parallelism.
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

#include <fmt/format.h>
#include <argparse/argparse.hpp>

#include <artea/cpu/vector_dataset.hpp>
#include <artea/cpu/kmeans_clustering.hpp>
#include <artea/cpu/bruteforce_router.hpp>
#include <artea/cpu/cluster_evaluator.hpp>
#include <artea/cpu/simd_distance.hpp>
#include <artea/definitions.hpp>
#include <artea/logger.hpp>

using namespace artea;
using namespace artea::cpu;

// --- Type Definitions ---
using VertexType = uint32_t;
using ElementType = float;
using DistFuncType = SIMDDistance<ElementType, DistanceMetrics::EUCLIDEAN>;

/**
 * @brief A generic benchmark helper to run K-means with a specific configuration.
 *
 * @tparam KmeansType The specific instantiation of KmeansClustering (with a specific Router).
 */
template <typename KmeansType>
void run_benchmark(
    const std::string& mode_name,
    const VectorDataset<VertexType, ElementType>& dataset,
    uint32_t num_clusters,
    const DistFuncType& dist_func,
    float sampling_ratio,
    int num_trials
) {
    const auto* base_vecs = dataset.get_base_vecs();
    uint32_t num_vecs = base_vecs->get_num_vecs();

    // Evaluator setup
    ClusterEvaluator<VertexType, ElementType, DistFuncType> evaluator(dist_func);

    logger.info(fmt::format(">>> Starting Benchmark: {} <<<", mode_name));

    double total_fit_ms = 0;
    double total_part_ms = 0;

    for (int i = 0; i < num_trials; ++i) {
        // Instantiate KmeansClustering
        KmeansType kmeans(num_vecs, num_clusters, dist_func);

        // 1. Measure Fit Time
        auto t1 = std::chrono::high_resolution_clock::now();
        kmeans.fit(*base_vecs, sampling_ratio);
        auto t2 = std::chrono::high_resolution_clock::now();
        double fit_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        // 2. Measure Partition Time (Also uses Router)
        auto t3 = std::chrono::high_resolution_clock::now();
        auto [reordered, offsets] = kmeans.partition_and_reorder(*base_vecs);
        auto t4 = std::chrono::high_resolution_clock::now();
        double part_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();

        // 3. Evaluate (Optional, just to ensure correctness)
        // Only evaluate on the first run to save time, or evaluate every time if needed.
        double inertia = 0.0;
        if (i == 0) {
            auto metrics = evaluator.evaluate(*base_vecs, kmeans.get_centroids());
            inertia = metrics.inertia;
        }

        total_fit_ms += fit_ms;
        total_part_ms += part_ms;

        logger.info(fmt::format(
            "   Trial {}: Fit={:.2f}ms, Part={:.2f}ms | Inertia={:.2e}",
            i + 1, fit_ms, part_ms, inertia
        ));
    }

    logger.success(fmt::format(
        "[{}] Average: Fit={:.2f}ms, Part={:.2f}ms\n",
        mode_name, total_fit_ms / num_trials, total_part_ms / num_trials
    ));
}

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("test_kmeans_comparison");

    program.add_argument("-c", "--config").default_value(std::string("./datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("-k", "--clusters").default_value(uint32_t(4096)).scan<'u', uint32_t>();
    program.add_argument("-r", "--ratio").help("Sampling ratio").default_value(0.1f).scan<'g', float>();
    program.add_argument("-n", "--trials").help("Number of trials").default_value(3).scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    const std::string config_path = program.get<std::string>("--config");
    const std::string dataset_name = program.get<std::string>("--dataset");
    const uint32_t num_clusters = program.get<uint32_t>("--clusters");
    const float ratio = program.get<float>("--ratio");
    const int trials = program.get<int>("--trials");

    try {
        if (!std::filesystem::exists(config_path)) {
            logger.error("Config file not found.");
            return 1;
        }

        VectorDataset<VertexType, ElementType> dataset(config_path, dataset_name);
        uint32_t dim = dataset.get_base_vecs()->get_vec_dim();

        DistFuncType dist_func(dim);

        // ==============================================================================
        // Configuration 1: Batch Parallel ONLY (intra_query_parallel = false)
        // Recommended for high throughput when processing many vectors.
        // ==============================================================================
        using RouterSeq = BruteforceRouter<VertexType, ElementType, DistFuncType, false>;
        using KmeansSeq = KmeansClustering<VertexType, ElementType, DistFuncType, RouterSeq>;

        run_benchmark<KmeansSeq>(
            "Mode 1: Batch Parallel Only (Default)",
            dataset, num_clusters, dist_func, ratio, trials
        );

        // ==============================================================================
        // Configuration 2: Nested Parallelism (intra_query_parallel = true)
        // Uses TBB to parallelize INSIDE each query calculation as well.
        // Useful if K is very large or vector dimension is huge, but carries overhead.
        // ==============================================================================
        using RouterPar = BruteforceRouter<VertexType, ElementType, DistFuncType, true>;
        using KmeansPar = KmeansClustering<VertexType, ElementType, DistFuncType, RouterPar>;

        run_benchmark<KmeansPar>(
            "Mode 2: Nested Parallelism (Batch + Intra Query Parallel)",
            dataset, num_clusters, dist_func, ratio, trials
        );

    } catch (const std::exception& e) {
        logger.error(fmt::format("Benchmark failed: {}", e.what()));
        return 1;
    }

    return 0;
}