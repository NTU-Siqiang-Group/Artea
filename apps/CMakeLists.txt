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

project(ArteaExamples CXX)

# Create bin directory in project root
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

file(GLOB EXAMPLE_SOURCES "*.cpp")

foreach(sourcefile ${EXAMPLE_SOURCES})
    get_filename_component(example_target ${sourcefile} NAME_WE)
    message(STATUS "Configuring example target: ${example_target}")

    add_executable(${example_target} ${sourcefile})

    # Enable Interprocedural Optimization (LTO)
    set_target_properties(${example_target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

    # Add hardware specific flags to the target
    target_compile_options(${example_target} PRIVATE -mavx512f -march=native)

    # Link against the core library
    target_link_libraries(${example_target} PRIVATE artea)

    # Explicit optimization flags for Release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(${example_target} PRIVATE -O3)
        target_link_options(${example_target} PRIVATE -O3)
    endif()

    # Create symlink in project root bin/ directory
    add_custom_command(TARGET ${example_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_BINARY_DIR}/apps/${example_target}
        ${CMAKE_SOURCE_DIR}/bin/${example_target}
        COMMENT "Creating symlink for ${example_target} in project root bin/"
    )
endforeach()
