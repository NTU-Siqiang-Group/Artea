/*
 * @FilePath: /Artea/tests/testVectorDataset.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-10-23 19:31:14
 * @Date: 2025-10-23 15:50:42
 * @Description: Test the VectorDataset class
 */

#include <iostream>
#include <string>
#include <chrono> 
#include <stdexcept>

#include <artea/cpu/vector_dataset.hpp>
#include <artea/types.hpp> 

int main() {
    
    const std::string config_path = "/home/yeweitang/Artea/datasets.json";
    const std::string dataset_name = "sift-1m";

    std::cout << "Starting VectorDataset test for dataset: " << dataset_name << std::endl;
    std::cout << "Using config file: " << config_path << std::endl;

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        artea::cpu::VectorDataset<
            uint32_t, // vecs_num_t (e.g., uint32_t)
            float     // vec_ele_t
        > dataset(config_path, dataset_name);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;

        std::cout << "\nSuccessfully loaded the dataset." << std::endl;
        std::cout << "Time taken to load: " << elapsed_ms.count() << " ms" << std::endl;

        // 在这里，你应该能够访问数据集的成员，例如：
        // (注意：为了访问这些成员，你需要在 VectorDataset 类中添加公共的 getter 方法)
        std::cout << "Base vectors: " << dataset.get_base_vecs()->get_num_vecs() << std::endl;
        std::cout << "Query vectors: " << dataset.get_query_vecs()->get_num_vecs() << std::endl;
        std::cout << "Ground truth vectors: " << dataset.get_gt_vecs()->get_num_vecs() << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "\nAn error occurred during the test: " << e.what() << std::endl;
        return 1; // 返回错误码
    } catch (...) {
        std::cerr << "\nAn unknown error occurred." << std::endl;
        return 1; // 返回错误码
    }

    return 0; // 成功
}