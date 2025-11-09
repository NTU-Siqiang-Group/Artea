/*
 * @FilePath: /Artea/tests/test_array.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-08 18:51:41
 * @Date: 2025-11-07 11:37:32
 * @Description: 
 */

#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <fmt/format.h> 

#include "artea/logger.hpp"
#include "artea/cpu/array.hpp"

namespace {
    artea::ArteaLogger logger("Benchmark", artea::LogLevel::INFO);
}

/**
 * @brief A simple timer class that uses the global logger to log its result.
 */
class Timer {
public:
    /**
     * @brief Constructs the timer and starts the clock.
     * @param task_name The name of the task being timed.
     */
    Timer(std::string task_name)
        : _name(std::move(task_name)),
          _start_time(std::chrono::high_resolution_clock::now()) {}

    /**
     * @brief Destructor that stops the clock and logs the elapsed time via the global logger.
     */
    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - _start_time).count();
        logger.info(fmt::format("Task '{}' took: {} ms", _name, duration));
    }

private:
    std::string _name;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
};

/**
 * @brief Generates a vector of random integers.
 * @param count The number of random integers to generate.
 * @return A std::vector<int> filled with random data.
 */
std::vector<int> generate_random_data(size_t count) {
    std::vector<int> data(count);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 1'000'000);
    std::generate(data.begin(), data.end(), [&]() { return dist(rng); });
    return data;
}

/**
 * @brief Runs the benchmark for std::sort.
 */
void benchmark_sort() {
    const size_t NUM_ELEMENTS = 10'000'000;
    logger.info(fmt::format("--- Starting std::sort Benchmark ({} elements) ---", NUM_ELEMENTS));

    // 1. Generate master data to ensure both containers sort the exact same values.
    auto master_data = generate_random_data(NUM_ELEMENTS);

    // 2. Benchmark artea::cpu::Array
    {
        artea::cpu::Array<int> my_array(NUM_ELEMENTS);
        std::copy(master_data.begin(), master_data.end(), my_array.begin());
        Timer timer("std::sort on artea::cpu::Array");
        std::sort(my_array.begin(), my_array.end());
    }

    // 3. Benchmark std::vector
    {
        std::vector<int> my_vector(NUM_ELEMENTS);
        std::copy(master_data.begin(), master_data.end(), my_vector.begin());
        Timer timer("std::sort on std::vector");
        std::sort(my_vector.begin(), my_vector.end());
    }
}

/**
 * @brief Runs the benchmark for std::merge.
 */
void benchmark_merge() {
    const size_t SIZE1 = 5'000'000;
    const size_t SIZE2 = 5'000'000;
    logger.info(fmt::format("--- Starting std::merge Benchmark ({} + {} elements) ---", SIZE1, SIZE2));

    // 1. Generate and sort master data for merging.
    auto master_data1 = generate_random_data(SIZE1);
    auto master_data2 = generate_random_data(SIZE2);
    std::sort(master_data1.begin(), master_data1.end());
    std::sort(master_data2.begin(), master_data2.end());

    // 2. Benchmark artea::cpu::Array
    {
        artea::cpu::Array<int> source1(SIZE1);
        std::copy(master_data1.begin(), master_data1.end(), source1.begin());
        
        artea::cpu::Array<int> source2(SIZE2);
        std::copy(master_data2.begin(), master_data2.end(), source2.begin());
        
        artea::cpu::Array<int> destination(SIZE1 + SIZE2);

        Timer timer("std::merge on artea::cpu::Array");
        std::merge(source1.cbegin(), source1.cend(), 
                   source2.cbegin(), source2.cend(), 
                   destination.begin());
    }

    // 3. Benchmark std::vector
    {
        std::vector<int> source1(SIZE1);
        std::copy(master_data1.begin(), master_data1.end(), source1.begin());

        std::vector<int> source2(SIZE2);
        std::copy(master_data2.begin(), master_data2.end(), source2.begin());

        std::vector<int> destination(SIZE1 + SIZE2);

        Timer timer("std::merge on std::vector");
        std::merge(source1.cbegin(), source1.cend(), 
                   source2.cbegin(), source2.cend(), 
                   destination.begin());
    }
}

#ifdef TBB_ENABLED

// --- Conversion Benchmark ---
void benchmark_conversion() {
    const size_t NUM_ELEMENTS = 10'000'000;
    logger.info(fmt::format("--- Starting Conversion Benchmark ({} elements) ---", NUM_ELEMENTS));

    // 1. Create and populate the source tbb::concurrent_vector
    logger.info("Creating source tbb::concurrent_vector...");
    auto source_data = generate_random_data(NUM_ELEMENTS);
    tbb::concurrent_vector<int> tbb_vec(source_data.begin(), source_data.end());
    logger.info("Source vector created.");

    // 2. Benchmark conversion to artea::cpu::Array
    {
        Timer timer("tbb::concurrent_vector -> artea::cpu::Array");
        artea::cpu::Array<int> my_array = artea::cpu::Array<int>::from(tbb_vec);
        if (my_array.empty()) { throw; }
    }

    // 3. Benchmark conversion to std::vector
    {
        Timer timer("tbb::concurrent_vector -> std::vector");
        std::vector<int> my_vector(tbb_vec.begin(), tbb_vec.end());
        if (my_vector.empty()) { throw; }
    }
}

#endif

int main() {
    logger.info("Starting performance comparison between artea::cpu::Array and std::vector.");
    
    benchmark_sort();
    benchmark_merge();

    #ifdef TBB_ENABLED
    benchmark_conversion();
    #endif

    logger.success("--- Benchmark Finished ---");
    logger.info("Note: Performance should be very similar as both containers provide");
    logger.info("random access iterators and manage a contiguous block of memory.");

    return 0;
}