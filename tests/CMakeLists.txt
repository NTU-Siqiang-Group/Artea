cmake_minimum_required(VERSION 3.16)

project(ArteaTests CXX)

file(GLOB TEST_SOURCES "*.cpp")

foreach(sourcefile ${TEST_SOURCES})
    get_filename_component(test_target ${sourcefile} NAME_WE)
    message(STATUS "Configuring test target: ${test_target}")
    add_executable(${test_target} ${sourcefile})
    target_link_libraries(${test_target} PRIVATE artea)
    if(${test_target} STREQUAL "test_simd_distance")
        target_link_libraries(${test_target} PRIVATE faiss)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${test_target} PRIVATE -O3)
        target_link_options(${test_target} PRIVATE -O3)
    endif()
endforeach()