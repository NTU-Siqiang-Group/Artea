# /*
#  * @FilePath: /yeweitang/Artea/tests/CMakeLists.txt
#  * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
#  * @LastEditTime: 2025-11-06 13:39:42
#  * @Date: 2025-10-23 15:55:05
#  * @Description: 
#  */

cmake_minimum_required(VERSION 3.16)

project(ArteaTests CXX)

add_executable(test_vector_dataset test_vector_dataset.cpp)
add_executable(test_simd_distance test_simd_distance.cpp)
add_executable(test_random_nn test_random_nn.cpp)
add_executable(test_concurrent_graph test_concurrent_graph.cpp)

# Link to Artea library
target_link_libraries(test_vector_dataset PRIVATE artea)
target_link_libraries(test_simd_distance PRIVATE artea faiss)
target_link_libraries(test_random_nn PRIVATE artea)
target_link_libraries(test_concurrent_graph PRIVATE artea)
