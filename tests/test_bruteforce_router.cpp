// Copyright 2025 Weitang Ye
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
 * @FilePath: /Artea/tests/test_bruteforce_router.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-26 20:43:24
 * @Date: 2025-11-25 12:50:16
 * @Description: Unit test for BruteforceRouter with benchmarks against Faiss and HNSWLib.
 */

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <algorithm>

#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <omp.h>

// Faiss Headers
#include <faiss/IndexFlat.h>
#include <faiss/utils/utils.h>

// HNSWLib Headers
#include <hnswlib/hnswlib.h>

// Artea Headers
#include <artea/cpu/containers/vector_dataset.hpp>
#include <artea/cpu/partitioning/bruteforce_router.hpp>
#include <artea/cpu/utils/simd_distance.hpp>
#include <artea/common/recall_estimator.hpp>
#include <artea/definitions.hpp>
#include <artea/common/logger.hpp>

using namespace artea;
using namespace artea::cpu;

// --- Type Definitions ---
using VertexType = uint32_t;
using ElementType = float;
using DistFuncType = SIMDDistance<ElementType, DistanceMetrics::EUCLIDEAN>;

// Config: No intra-query parallel, pure batch parallel
using ArteaRouter = BruteforceRouter<VertexType, ElementType, DistFuncType, false>;

// Config: Epsilon 1e-5 (Inverse = 100000)
using RecEstimator = RecallEstimator<VertexType, ElementType, DistFuncType, 100000>;

int main(int argc, char* argv[]) {
    argparse::ArgumentParser program("test_bruteforce_router");

    program.add_argument("-c", "--config").default_value(std::string("./datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("-n", "--trials").default_value(3).scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    const std::string config_path = program.get<std::string>("--config");
    const std::string dataset_name = program.get<std::string>("--dataset");
    const int trials = program.get<int>("--trials");

    try {
        // --- 1. Load Dataset ---
        if (!std::filesystem::exists(config_path)) {
            logger.error("Config file not found.");
            return 1;
        }

        VectorDataset<VertexType, ElementType> dataset(config_path, dataset_name);

        const auto& base_vecs = dataset.get_base_vecs();
        const auto& query_vecs = dataset.get_query_vecs();
        const auto& gt_vecs = dataset.get_gt_vecs();

        uint32_t dim = base_vecs.get_vec_dim();
        uint32_t num_base = base_vecs.get_num_vecs();
        uint32_t num_queries = query_vecs.get_num_vecs();

        logger.info(fmt::format("Dataset Loaded: {} base, {} queries, dim={}", num_base, num_queries, dim));
        DistFuncType dist_func(dim);

        // --- 2. Faiss Benchmark ---
        logger.info(">>> Running Faiss IndexFlatL2 <<<");

        faiss::IndexFlatL2 faiss_index(dim);
        faiss_index.add(num_base, base_vecs.get_all());

        // Faiss uses int64_t (long) for labels
        std::vector<faiss::idx_t> faiss_labels(num_queries);
        std::vector<float> faiss_dists(num_queries);

        double faiss_total_ms = 0.0;

        for(int t = 0; t < trials; ++t) {
            auto t1 = std::chrono::high_resolution_clock::now();
            faiss_index.search(num_queries, query_vecs.get_all(), 1, faiss_dists.data(), faiss_labels.data());
            auto t2 = std::chrono::high_resolution_clock::now();
            faiss_total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }

        logger.success(fmt::format("[Faiss] Avg Time: {:.2f} ms | QPS: {:.2f}",
            faiss_total_ms / trials, (num_queries * 1000.0) / (faiss_total_ms / trials)));

        // --- 3. Artea Benchmark ---
        logger.info(">>> Running Artea BruteforceRouter <<<");

        ArteaRouter router(base_vecs, dist_func);
        // Note: Router treats base vectors as 'centroids' here

        std::vector<VertexType> artea_labels;
        double artea_total_ms = 0.0;

        for(int t = 0; t < trials; ++t) {
            auto t1 = std::chrono::high_resolution_clock::now();
            artea_labels = router.batch_query(query_vecs);
            auto t2 = std::chrono::high_resolution_clock::now();
            artea_total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }

        logger.success(fmt::format("[Artea] Avg Time: {:.2f} ms | QPS: {:.2f}",
            artea_total_ms / trials, (num_queries * 1000.0) / (artea_total_ms / trials)));

        // --- 4. HNSWLib Benchmark ---
        logger.info(">>> Running HNSWLib Bruteforce <<<");

        hnswlib::L2Space space(dim);
        hnswlib::BruteforceSearch<float> hnsw_alg(&space, num_base);

        const float* base_ptr = base_vecs.get_all();
        for (size_t i = 0; i < num_base; ++i) {
            hnsw_alg.addPoint((void*)(base_ptr + i * dim), i);
        }

        std::vector<VertexType> hnsw_labels(num_queries);
        double hnsw_total_ms = 0.0;
        const float* query_ptr = query_vecs.get_all();

        for(int t = 0; t < trials; ++t) {
            auto t1 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for num_threads(omp_get_max_threads())
            for (size_t i = 0; i < num_queries; ++i) {
                auto result = hnsw_alg.searchKnn((void*)(query_ptr + i * dim), 1);
                hnsw_labels[i] = static_cast<VertexType>(result.top().second);
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            hnsw_total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }

        logger.success(fmt::format("[HNSWLib] Avg Time: {:.2f} ms | QPS: {:.2f}",
            hnsw_total_ms / trials, (num_queries * 1000.0) / (hnsw_total_ms / trials)));

        // --- 5. Validation using RecallEstimator ---
        RecEstimator estimator(dist_func);

        // 5a. Validate Artea
        logger.info("--- Validating Artea Results ---");
        auto artea_metrics = estimator.calculate_recall_at_1(
            artea_labels, gt_vecs, query_vecs, base_vecs
        );

        // 5b. Validate Faiss (Need conversion to uint32_t)
        logger.info("--- Validating Faiss Results ---");
        std::vector<VertexType> faiss_labels_u32(num_queries);
        for(size_t i=0; i<num_queries; ++i) {
            faiss_labels_u32[i] = static_cast<VertexType>(faiss_labels[i]);
        }

        auto faiss_metrics = estimator.calculate_recall_at_1(
            faiss_labels_u32, gt_vecs, query_vecs, base_vecs
        );

        // 5c. Validate HNSWLib
        logger.info("--- Validating HNSWLib Results ---");
        auto hnsw_metrics = estimator.calculate_recall_at_1(
            hnsw_labels, gt_vecs, query_vecs, base_vecs
        );

        // 5d. Agreement Check
        size_t match_count = 0;
        for(size_t i=0; i<num_queries; ++i) {
            if(artea_labels[i] == faiss_labels_u32[i]) match_count++;
        }

        logger.info(fmt::format("Artea vs Faiss Agreement: {:.2f}%",
            (double)match_count / num_queries * 100.0));

        size_t hnsw_match_count = 0;
        for(size_t i=0; i<num_queries; ++i) {
            if(artea_labels[i] == hnsw_labels[i]) hnsw_match_count++;
        }
        logger.info(fmt::format("Artea vs HNSWLib Agreement: {:.2f}%",
            (double)hnsw_match_count / num_queries * 100.0));

        if (artea_metrics.soft_recall < 0.999) {
            logger.warn("Artea Recall is unexpectedly low!");
        } else {
            logger.success("Artea correctness validated.");
        }

    } catch (const std::exception& e) {
        logger.error(fmt::format("Error: {}", e.what()));
        return 1;
    }

    return 0;
}