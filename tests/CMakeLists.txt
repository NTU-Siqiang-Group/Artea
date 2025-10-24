# CMake最低版本要求
cmake_minimum_required(VERSION 3.16)

# 定义测试项目的名称
project(ArteaTests CXX)

# 添加可执行文件目标
add_executable(test_vector_dataset test_vector_dataset.cpp)
add_executable(test_simd_distance test_simd_distance.cpp)
add_executable(test_mini_batch_kmeans test_mini_batch_kmeans.cpp)

# 链接到 "artea" 库。
# CMake 会自动处理 test_vector_dataset 对 nlohmann_json, argparse, 和 OpenMP 的所有依赖。
target_link_libraries(test_vector_dataset PRIVATE artea)
target_link_libraries(test_simd_distance PRIVATE
    artea
    faiss
)
target_link_libraries(test_mini_batch_kmeans PRIVATE
    artea
    faiss
)
