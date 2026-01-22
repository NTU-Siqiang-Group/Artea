# --- 1. Project basic configuration ---
cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0135 NEW) # Use modern timestamp behavior
project(Artea LANGUAGES C CXX)

# include(CheckIPOSupported)
# check_ipo_supported(RESULT result OUTPUT output)
# if(result)
#     message(STATUS "IPO/LTO is supported. Enabling globally.")
#     set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
# else()
#     message(WARNING "IPO/LTO is not supported: ${output}")
# endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# --- Dependency management ---
include(FetchContent)
find_package(OpenMP REQUIRED)

# source the setvars.sh script from Intel oneAPI

# --- Find Intel TBB ---
find_package(TBB REQUIRED)
message(STATUS "TBB_DIR: ${TBB_DIR}")

# --- Find Intel MKL ---
find_package(MKL REQUIRED)

FetchContent_Declare(
	nlohmann_json
	URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)

FetchContent_Declare(
	argparse
	GIT_REPOSITORY https://github.com/p-ranav/argparse.git
	GIT_TAG "v3.2"
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

FetchContent_Declare(
    range-v3
    GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
    GIT_TAG "0.12.0"
)

FetchContent_Declare(
    hnswlib
    GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
    GIT_TAG "v0.8.0"
)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
)

# --- Google Benchmark Configuration ---
# Disable building tests and examples for Google Benchmark to speed up the process
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "Disable benchmark docs" FORCE)

# --- Faiss Submodule Configuration ---
# Set options to build a minimal CPU version of Faiss
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable Faiss GPU support")
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Faiss Python wrappers")
set(BUILD_TESTING OFF CACHE BOOL "Disable building Faiss's own tests")
set(FAISS_OPT_LEVEL "native" CACHE STRING "Optimize Faiss for the host CPU")
add_subdirectory(third_party/faiss)

# --- Downloads and configures all declared content ---
FetchContent_MakeAvailable(
	nlohmann_json
	argparse
	termcolor
	fmt
	range-v3
	hnswlib
	googletest
	benchmark
)

# --- Define core "artea" library ---
add_library(artea INTERFACE)
target_include_directories(artea INTERFACE
  	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Add compiler flag for AVX512 support to artea library
target_compile_options(artea INTERFACE -mavx512f -march=native)

#
target_compile_definitions(artea INTERFACE
	# avoid error caused by fmt library
	FMT_USE_CONSTEVAL=0
)

target_link_libraries(artea INTERFACE
    nlohmann_json::nlohmann_json
    argparse::argparse
    OpenMP::OpenMP_CXX
    termcolor
    fmt::fmt
    TBB::tbb
    MKL::MKL
	range-v3::range-v3
	hnswlib
	faiss
)

include(CTest)

# --- 5. Include subdirectories ---
add_subdirectory(tests)
add_subdirectory(benchmarks)
