# Copyright 2025 Weitang Ye
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)

# Renamed project to reflect that these are benchmarks
project(ArteaBench CXX)

file(GLOB BENCH_SOURCES "*.cpp")

foreach(sourcefile ${BENCH_SOURCES})
    get_filename_component(bench_target ${sourcefile} NAME_WE)
    message(STATUS "Configuring benchmark target: ${bench_target}")

    add_executable(${bench_target} ${sourcefile})

    # Enable Interprocedural Optimization (LTO)
    set_target_properties(${bench_target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

    # Add hardware specific flags to the target
    target_compile_options(${bench_target} PRIVATE -mavx512f -march=native)

    # Link against the core library and Google Benchmark
    target_link_libraries(${bench_target} PRIVATE
        artea
        benchmark::benchmark
argparse::argparse
        hnswlib
	    faiss
    )

    # Explicit optimization flags for Release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${bench_target} PRIVATE -O3)
        target_link_options(${bench_target} PRIVATE -O3)
    endif()
endforeach()