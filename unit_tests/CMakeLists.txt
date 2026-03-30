cmake_minimum_required(VERSION 3.16)

project(ArteaTests CXX)

include(GoogleTest)

find_package(Threads REQUIRED)

file(GLOB TEST_SOURCES "*.cpp")

foreach(sourcefile ${TEST_SOURCES})
    get_filename_component(test_target ${sourcefile} NAME_WE)
    message(STATUS "Configuring test target: ${test_target}")
    add_executable(${test_target} ${sourcefile})
    set_target_properties(${test_target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    target_compile_options(${test_target} PRIVATE -mavx512f -march=native)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${test_target} PRIVATE -O3)
        target_link_options(${test_target} PRIVATE -O3)
    endif()
    target_link_libraries(${test_target} PRIVATE
        artea
        GTest::gtest
        benchmark::benchmark
        Threads::Threads
    )
    add_test(NAME ${test_target} COMMAND ${test_target})
endforeach()