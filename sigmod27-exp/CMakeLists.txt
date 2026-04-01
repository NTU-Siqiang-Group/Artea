cmake_minimum_required(VERSION 3.16)

project(Sigmod27Exp CXX)

include(GoogleTest)

find_package(Threads REQUIRED)

file(GLOB EXP_SOURCES "*.cpp")

foreach(sourcefile ${EXP_SOURCES})
    get_filename_component(exp_target ${sourcefile} NAME_WE)
    message(STATUS "Configuring experiment target: ${exp_target}")
    add_executable(${exp_target} ${sourcefile})
    set_target_properties(${exp_target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    target_compile_options(${exp_target} PRIVATE -mavx512f -march=native)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${exp_target} PRIVATE -O3)
        target_link_options(${exp_target} PRIVATE -O3)
    endif()
    target_link_libraries(${exp_target} PRIVATE
        artea
        GTest::gtest
        benchmark::benchmark
        Threads::Threads
    )
endforeach()
