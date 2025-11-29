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

project(ArteaTests CXX)

file(GLOB BENCH_SOURCES "*.cpp")

foreach(sourcefile ${BENCH_SOURCES})
    get_filename_component(test_target ${sourcefile} NAME_WE)
    message(STATUS "Configuring test target: ${test_target}")
    add_executable(${test_target} ${sourcefile})
    set_target_properties(${test_target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    add_compile_options(-mavx512f -march=native)
    target_link_libraries(${test_target} PRIVATE artea)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${test_target} PRIVATE -O3)
        target_link_options(${test_target} PRIVATE -O3)
    endif()
endforeach()