/*
 * @FilePath: /Artea/tests/test_kmeans_partition.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Benchmark for K-means clustering and partitioning using the latest Artea interfaces.
 *               Evaluates Sampling, Training, and Partitioning phases separately.
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <memory>

#include <fmt/format.h>
#include <argparse/argparse.hpp>

#include <artea/cpu/containers/vector_dataset.hpp>
#include <artea/cpu/partitioning/kmeans_clustering.hpp>
#include <artea/cpu/partitioning/bruteforce_router.hpp>
#include <artea/cpu/partitioning/partitioner.hpp>
#include <artea/cpu/partitioning/cluster_evaluator.hpp>
#include <artea/cpu/utils/vector_sampler.hpp>
#include <artea/cpu/utils/simd_distance.hpp>
#include <artea/definitions.hpp>
#include <artea/common/logger.hpp>

using namespace artea;
using namespace artea::cpu;

// --- Type Definitions ---
using VertexType = uint32_t;
using ElementType = float;
// Use Euclidean distance with SIMD optimization
using DistFuncType = SIMDDistance<ElementType, DistanceMetrics::EUCLIDEAN>;
// Router configuration: purely batch parallel, no intra-query parallelism
using RouterType = BruteforceRouter<VertexType, ElementType, DistFuncType, false>;
using SamplerType = VectorSampler<VertexType, ElementType>;

// K-means instantiation
using KmeansType = KmeansClustering<
    VertexType,
    ElementType,
    DistFuncType,
    RouterType,
    SamplerType
>;

// Partitioner instantiation
using PartitionerType = Partitioner<
    VertexType,
    ElementType,
    KmeansType,
    RouterType,
    DistFuncType
>;

/**
 * @brief Performs the full benchmark pipeline: Sampling -> Clustering -> Partitioning -> Evaluation.
 *
 * @param dataset The loaded vector dataset containing base vectors.
 * @param num_clusters The target number of clusters (K).
 * @param dist_func The distance function functor.
 * @param sampling_ratio The ratio of data used for K-means training.
 * @param num_trials The number of times to run the benchmark for averaging.
 */
void run_kmeans_benchmark(
    const VectorDataset<VertexType, ElementType>& dataset,
    uint32_t num_clusters,
    const DistFuncType& dist_func,
    float sampling_ratio,
    int num_trials
) {
    const auto& base_vecs = dataset.get_base_vecs();
    uint32_t num_vecs = base_vecs.get_num_vecs();
    uint32_t dim = base_vecs.get_vec_dim();

    SamplerType sampler;
    ClusterEvaluator<VertexType, ElementType, DistFuncType> evaluator(dist_func);

    logger.info(">>> Starting K-means Partition Benchmark <<<");
    logger.info(fmt::format("Dataset: {} vectors, {} dim", num_vecs, dim));
    logger.info(fmt::format("Config: K={}, Sampling Ratio={:.2f}, Trials={}", num_clusters, sampling_ratio, num_trials));
    logger.info("Router: BruteforceRouter (Batch Parallel Only)");

    double total_sample_ms = 0;
    double total_fit_ms = 0;
    double total_part_ms = 0;

    for (int i = 0; i < num_trials; ++i) {
        logger.debug(fmt::format("--- Trial {}/{} ---", i + 1, num_trials));

        // 1. Measure Sampling Time independently
        // Note: Kmeans::fit calls sampler internally, but we measure it here
        // strictly to evaluate the sampler's performance.
        auto t0 = std::chrono::high_resolution_clock::now();
        {
            auto temp_sample = sampler.sample(base_vecs, sampling_ratio);
            // Prevent optimization
            volatile size_t s = temp_sample.get_num_vecs();
            (void)s;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double sample_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2. Measure Clustering (Fit) Time
        // Instantiating K-means
        KmeansType kmeans(num_vecs, num_clusters, dim, dist_func, sampler);

        auto t2 = std::chrono::high_resolution_clock::now();
        kmeans.fit(base_vecs, sampling_ratio);
        auto t3 = std::chrono::high_resolution_clock::now();
        double fit_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // 3. Measure Partition & Reorder Time
        // Using the new Partitioner class as per interface changes
        PartitionerType partitioner(kmeans);

        auto t4 = std::chrono::high_resolution_clock::now();
        auto partitioned_result = partitioner.partition_and_reorder(base_vecs);
        auto t5 = std::chrono::high_resolution_clock::now();
        double part_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

        // 4. Evaluation (Inertia, Balance, etc.)
        logger.info("Evaluating clustering quality...");
        auto metrics = evaluator.evaluate(base_vecs, kmeans.get_centroids());

        logger.info(fmt::format("   -> Inertia (SSE):       {:.4e}", metrics.inertia));
        logger.info(fmt::format("   -> Silhouette Score:    {:.4f}", metrics.silhouette_score));
        logger.info(fmt::format("   -> Intra-Cluster Dist:  {:.4f}", metrics.intra_cluster_distance_avg));
        logger.info(fmt::format("   -> Inter-Cluster Dist:  {:.4f}", metrics.inter_cluster_distance_avg));
        logger.info(fmt::format("   -> Load Balance StdDev: {:.2f} (lower is better)", metrics.part_sizes_stddev));

        total_sample_ms += sample_ms;
        total_fit_ms += fit_ms;
        total_part_ms += part_ms;

        logger.info(fmt::format(
            "   Trial {} Results: Sampling={:.2f}ms, Fit={:.2f}ms, Partition={:.2f}ms",
            i + 1, sample_ms, fit_ms, part_ms
        ));
    }

    logger.success(fmt::format(
        "Average Results ({} trials):\n"
        "   - Sampling Time:     {:.2f} ms\n"
        "   - Clustering Time:   {:.2f} ms (includes internal sampling)\n"
        "   - Partitioning Time: {:.2f} ms (includes reordering)\n",
        num_trials,
        total_sample_ms / num_trials,
        total_fit_ms / num_trials,
        total_part_ms / num_trials
    ));
}

int main(int argc, char* argv[]) {

    // Argument Parsing
    argparse::ArgumentParser program("test_kmeans_partition");

    program.add_argument("-c", "--config")
        .help("Path to dataset config JSON")
        .default_value(std::string("./datasets.json"));

    program.add_argument("-d", "--dataset")
        .help("Dataset name key in config")
        .default_value(std::string("sift-1m"));

    program.add_argument("-k", "--clusters")
        .help("Number of clusters")
        .default_value(uint32_t(1024))
        .scan<'u', uint32_t>();

    program.add_argument("-r", "--ratio")
        .help("Sampling ratio (0.0 - 1.0)")
        .default_value(0.1f)
        .scan<'g', float>();

    program.add_argument("-n", "--trials")
        .help("Number of benchmark trials")
        .default_value(5)
        .scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const std::string config_path = program.get<std::string>("--config");
    const std::string dataset_name = program.get<std::string>("--dataset");
    const uint32_t num_clusters = program.get<uint32_t>("--clusters");
    const float ratio = program.get<float>("--ratio");
    const int trials = program.get<int>("--trials");

    // Main execution block
    try {
        if (!std::filesystem::exists(config_path)) {
            logger.error(fmt::format("Config file not found: {}", config_path));
            return 1;
        }

        // Load Dataset
        VectorDataset<VertexType, ElementType> dataset(config_path, dataset_name);
        uint32_t dim = dataset.get_base_vecs().get_vec_dim();

        // Initialize SIMD Distance Function
        DistFuncType dist_func(dim);

        // Run Benchmark
        run_kmeans_benchmark(
            dataset,
            num_clusters,
            dist_func,
            ratio,
            trials
        );

    } catch (const std::exception& e) {
        logger.error(fmt::format("Benchmark failed with exception: {}", e.what()));
        return 1;
    }

    return 0;
}