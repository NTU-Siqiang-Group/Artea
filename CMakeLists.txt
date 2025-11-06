# --- 1. Project basic configuration ---
cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0135 NEW) # Use modern timestamp behavior
project(Artea LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# --- Dependency management ---
include(FetchContent)
find_package(OpenMP REQUIRED)

# --- Find Intel TBB ---
find_package(TBB REQUIRED) # for concurrent_vector and parallel algorithms

# --- Find Intel MKL (Optional) ---
find_package(MKL QUIET) # source the setvars.sh script from Intel oneAPI

FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)

FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG "v2.2"
)

FetchContent_Declare(
  termcolor
  GIT_REPOSITORY https://github.com/ikalnytskyi/termcolor.git
  GIT_TAG "v2.1.0"  
)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG "10.1.1" # Using a slightly older but common version
)

# This command downloads and configures all declared content
FetchContent_MakeAvailable(nlohmann_json argparse termcolor fmt)

# --- 3. Faiss Submodule Configuration ---
# Set options to build a minimal CPU version of Faiss
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable Faiss GPU support")
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Faiss Python wrappers")
set(BUILD_TESTING OFF CACHE BOOL "Disable building Faiss's own tests")
set(FAISS_OPT_LEVEL "native" CACHE STRING "Optimize Faiss for the host CPU")
add_subdirectory(third_party/faiss)

# --- 4. Define core "artea" library ---
add_library(artea INTERFACE)
target_include_directories(artea INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Add compiler flag for AVX512 support to our library
target_compile_options(artea INTERFACE -mavx512f)

target_link_libraries(artea INTERFACE
  nlohmann_json::nlohmann_json
  argparse::argparse
  OpenMP::OpenMP_CXX
  faiss    
  termcolor
  fmt::fmt
  TBB::tbb  # Link Intel TBB
)

# Conditionally add MKL support
if(TARGET MKL::MKL)
    message(STATUS "Found Intel MKL, enabling vectorized features.")
    # Add a definition so C++ code can check for MKL
    target_compile_definitions(artea INTERFACE MKL_ENABLED)
    # Link MKL libraries. MKL::MKL should handle threading/interface layers.
    target_link_libraries(artea INTERFACE MKL::MKL)
else()
    message(STATUS "Intel MKL not found, vectorized features will be disabled.")
endif()

target_compile_definitions(artea INTERFACE FMT_USE_CONSTEVAL=0)

# --- 5. Include subdirectories ---
add_subdirectory(tests)