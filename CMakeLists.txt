# --- 1. Project basic configuration ---
cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0135 NEW) # Use modern timestamp behavior
project(Artea LANGUAGES C CXX)



set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Link against GCC-14's libstdc++ at runtime
add_link_options("-Wl,-rpath,/usr/local/gcc-14/lib64")

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -g -fno-omit-frame-pointer")

# --- Dependency management ---
set(FETCHCONTENT_FULLY_DISCONNECTED OFF CACHE BOOL
    "Skip dependency downloads and updates when all sources are already available")
include(FetchContent)
find_package(OpenMP REQUIRED)

# --- Project-managed oneTBB ---
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tbb.cmake)

# --- Project-managed Intel MKL ---
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/mkl.cmake)

# --- Find Boost (for math/statistics) ---
find_package(Boost REQUIRED)

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
	GIT_TAG "12.1.0"
)

FetchContent_Declare(
    range-v3
    GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
    GIT_TAG "0.12.0"
)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG "v1.17.0"
)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG "v1.9.5"
)

# FetchContent_Declare(
#     indicators
#     GIT_REPOSITORY https://github.com/p-ranav/indicators.git
#     GIT_TAG "v2.3"
# )

# FetchContent_Declare(
#     tabulate
#     GIT_REPOSITORY https://github.com/p-ranav/tabulate.git
#     GIT_TAG "v1.5"
# )

# --- Google Benchmark Configuration ---
# Disable building tests and examples for Google Benchmark to speed up the process
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "Disable benchmark docs" FORCE)

# --- Dependency test configuration ---
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

# --- Downloads and configures all declared content ---
FetchContent_MakeAvailable(
	nlohmann_json
	argparse
	termcolor
	fmt
	range-v3
	googletest
	benchmark
)

# --- Define core "artea" library ---
add_library(artea INTERFACE)
target_include_directories(artea INTERFACE
  	"${CMAKE_CURRENT_SOURCE_DIR}/include"
)

# Add compiler flags for SIMD and architecture-specific optimization.
# `-mprefer-vector-width=512` overrides GCC's default 256-bit auto-vector
# choice: without it GCC autovectorises into ymm registers even under
# `-mavx512f -march=native` (a Skylake-X frequency-throttling precaution that
# no longer applies to Ice Lake / Sapphire Rapids). The hand-written _mm512
# kernels are unaffected; this widens the autovectorised periphery (queue
# maintenance, traversal helpers) to full zmm width, matching the
# bench-artea build flags in artea-benchmark/cmake/common.cmake.
target_compile_options(artea INTERFACE
    -march=native -mavx512f -mfma -funroll-loops -mprefer-vector-width=512)

# --- Compile Definitions (Macros) ---
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
	Boost::boost
)

if (NOT ARTEA_SKIP_SUBDIRS)
    include(CTest)

    # --- 5. Include subdirectories ---
    add_subdirectory(unit_tests)
    add_subdirectory(micro_benchmarks)
    add_subdirectory(apps)
endif()
