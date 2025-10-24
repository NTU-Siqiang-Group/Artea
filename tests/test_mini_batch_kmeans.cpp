/*
 * @FilePath: /Artea/tests/test_mini_batch_kmeans.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Date: 2025-10-24 10:00:00
 * @Description: Test the MiniBatchKmeans class for training and assignment performance.
 */

#include <iostream>
#include <string>
#include <chrono> 
#include <stdexcept>
#include <memory>
#include <format>

#include <artea/cpu/vector_dataset.hpp>
#include <artea/cpu/mini_batch_kmeans.hpp>
#include <artea/logger.hpp>
#include <artea/types.hpp> 

int main() {
    
    // --- Test Configuration ---
    const std::string config_path = "/home/yeweitang/Artea/datasets.json";
    const std::string dataset_name = "sift-1m";
    
    using vecs_num_t = uint32_t;
    using vec_ele_t = float;

    // K-Means Parameters
    constexpr artea::cluster_num_t NUM_CLUSTERS = 256;
    constexpr vecs_num_t BATCH_SIZE = 1024;        // 0.1% of SIMD Dataset
    constexpr artea::iter_t MAX_ITERS = 500;
    constexpr float TOLERANCE = 1e-1;

    // --- Logger ---
    artea::ArteaLogger logger("TestMiniBatchKmeans", artea::LogLevel::INFO);

    logger.info(std::format("Starting MiniBatchKmeans test for dataset: {}", dataset_name));
    logger.info(std::format("Parameters: K={}, BatchSize={}, MaxIters={}, Tolerance={}", NUM_CLUSTERS, BATCH_SIZE, MAX_ITERS, TOLERANCE));

    try {
        // --- 1. Load Dataset ---
        logger.info("Loading dataset...");
        auto dataset = std::make_unique<artea::cpu::VectorDataset<
            vecs_num_t, // vecs_num_t
            vec_ele_t,  // vec_ele_t for base/query
            vecs_num_t  // vec_id_t for ground truth
        >>(config_path, dataset_name);
        
        const auto* base_vecs = dataset->get_base_vecs();
        const auto num_vecs = base_vecs->get_num_vecs();
        const auto vec_dim = base_vecs->get_vec_dim();
        logger.success("Dataset loaded successfully.");

        // --- 2. Initialize K-Means ---
        artea::cpu::MiniBatchKmeans<vecs_num_t, vec_ele_t> kmeans(
            num_vecs, NUM_CLUSTERS, BATCH_SIZE, MAX_ITERS, vec_dim, TOLERANCE
        );

        // --- 3. Test fit() Performance ---
        logger.info("--- Testing fit() ---");
        auto start_fit = std::chrono::high_resolution_clock::now();
        kmeans.fit(base_vecs);
        auto end_fit = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> fit_duration = end_fit - start_fit;
        logger.success(std::format("fit() completed in: {:.2f} ms", fit_duration.count()));

        // --- 4. Test assign_index() Performance (Brute Force) ---
        logger.info("--- Testing assign_index() with BRUTE_FORCE ---");
        auto start_assign_bf = std::chrono::high_resolution_clock::now();
        kmeans.assign_index<artea::cpu::AssignMethod::BRUTE_FORCE>(base_vecs);
        auto end_assign_bf = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> assign_bf_duration = end_assign_bf - start_assign_bf;
        logger.success(std::format("assign_index(BRUTE_FORCE) completed in: {:.2f} ms", assign_bf_duration.count()));

        // --- 5. Test assign_index() Performance (Faiss HNSW) ---
        logger.info("--- Testing assign_index() with FAISS_HNSW ---");
        auto start_assign_hnsw = std::chrono::high_resolution_clock::now();
        kmeans.assign_index<artea::cpu::AssignMethod::FAISS_HNSW>(base_vecs);
        auto end_assign_hnsw = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> assign_hnsw_duration = end_assign_hnsw - start_assign_hnsw;
        logger.success(std::format("assign_index(FAISS_HNSW) completed in: {:.2f} ms", assign_hnsw_duration.count()));


    } catch (const std::exception& e) {
        logger.error(std::string("An exception occurred: ") + e.what());
        return 1;
    }

    logger.success("\nAll MiniBatchKmeans tests completed successfully!");
    return 0;
}