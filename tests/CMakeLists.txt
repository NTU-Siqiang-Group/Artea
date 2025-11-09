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
add_executable(test_array test_array.cpp)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling Link-Time Optimization (LTO) for tests")
    set_property(TARGET test_vector_dataset PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET test_simd_distance PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET test_random_nn PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET test_array PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Link to Artea library
target_link_libraries(test_vector_dataset PRIVATE artea)
target_link_libraries(test_simd_distance PRIVATE artea faiss)
target_link_libraries(test_random_nn PRIVATE artea)
target_link_libraries(test_array PRIVATE artea)
