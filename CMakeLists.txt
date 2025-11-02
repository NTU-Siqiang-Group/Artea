# --- 1. Project basic configuration ---
cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0135 NEW) # Use modern timestamp behavior
project(Artea LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 2. Dependency management (FetchContent & FindPackage) ---
include(FetchContent)
find_package(OpenMP REQUIRED)

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
  eigen # We give it a name 'eigen'
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG "5.0.0"
)



# This command downloads and configures all declared content
FetchContent_MakeAvailable(nlohmann_json argparse termcolor eigen)

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
)

# --- 5. Include subdirectories ---
add_subdirectory(tests)