/*
 * @FilePath: /Artea/tests/test_array.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-15 14:25:41
 * @Date: 2025-11-07 11:37:32
 * @Description:
 */

#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>

#include <fmt/format.h>
#include <tbb/concurrent_vector.h>

#include <artea/common/logger.hpp>
#include <artea/cpu/containers/array.hpp>

class Timer {
public:
    Timer(std::string task_name)
        : _name(std::move(task_name)),
          _start_time(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - _start_time).count();
        artea::logger.info(fmt::format("Task '{}' took: {} ms", _name, duration));
    }

private:
    std::string _name;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;
};

std::vector<int> generate_random_data(size_t count) {
    std::vector<int> data(count);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 1'000'000);
    std::generate(data.begin(), data.end(), [&]() { return dist(rng); });
    return data;
}

void benchmark_sort() {
    const size_t NUM_ELEMENTS = 10'000'000;
    artea::logger.info(fmt::format("--- Starting std::sort Benchmark ({} elements) ---", NUM_ELEMENTS));

    auto master_data = generate_random_data(NUM_ELEMENTS);

    // Benchmark artea::cpu::Array
    {
        artea::cpu::Array<int> my_array;
        my_array.resize(NUM_ELEMENTS);
        std::copy(master_data.begin(), master_data.end(), my_array.begin());

        Timer timer("std::sort on artea::cpu::Array");
        std::sort(my_array.begin(), my_array.end());
    }

    // Benchmark std::vector
    {
        std::vector<int> my_vector(NUM_ELEMENTS);
        std::copy(master_data.begin(), master_data.end(), my_vector.begin());

        Timer timer("std::sort on std::vector");
        std::sort(my_vector.begin(), my_vector.end());
    }
}

void benchmark_merge() {
    const size_t SIZE1 = 5'000'000;
    const size_t SIZE2 = 5'000'000;
    artea::logger.info(fmt::format("--- Starting std::merge Benchmark ({} + {} elements) ---", SIZE1, SIZE2));

    auto master_data1 = generate_random_data(SIZE1);
    auto master_data2 = generate_random_data(SIZE2);
    std::sort(master_data1.begin(), master_data1.end());
    std::sort(master_data2.begin(), master_data2.end());

    // Benchmark artea::cpu::Array
    {
        artea::cpu::Array<int> source1;
        source1.resize(SIZE1);
        std::copy(master_data1.begin(), master_data1.end(), source1.begin());

        artea::cpu::Array<int> source2;
        source2.resize(SIZE2);
        std::copy(master_data2.begin(), master_data2.end(), source2.begin());

        artea::cpu::Array<int> destination;
        destination.resize(SIZE1 + SIZE2);

        Timer timer("std::merge on artea::cpu::Array");
        std::merge(source1.cbegin(), source1.cend(),
                   source2.cbegin(), source2.cend(),
                   destination.begin());
    }

    // Benchmark std::vector
    {
        std::vector<int> source1(master_data1); // Simpler copy
        std::vector<int> source2(master_data2); // Simpler copy
        std::vector<int> destination(SIZE1 + SIZE2);

        Timer timer("std::merge on std::vector");
        std::merge(source1.cbegin(), source1.cend(),
                   source2.cbegin(), source2.cend(),
                   destination.begin());
    }
}

void benchmark_conversion() {
    const size_t NUM_ELEMENTS = 10'000'000;
    artea::logger.info(fmt::format("--- Starting Conversion Benchmark ({} elements) ---", NUM_ELEMENTS));

    artea::logger.info("Creating source tbb::concurrent_vector...");
    auto source_data = generate_random_data(NUM_ELEMENTS);
    tbb::concurrent_vector<int> tbb_vec(source_data.begin(), source_data.end());
    artea::logger.info("Source vector created.");

    // Benchmark conversion to artea::cpu::Array
    {
        artea::cpu::Array<int> my_array;
        Timer timer("tbb::concurrent_vector -> artea::cpu::Array");
        my_array.from(tbb_vec);
        if (my_array.size() != NUM_ELEMENTS) { throw std::logic_error("Conversion failed!"); }
    }

    // Benchmark conversion to std::vector
    {
        Timer timer("tbb::concurrent_vector -> std::vector");
        std::vector<int> my_vector(tbb_vec.begin(), tbb_vec.end());
        if (my_vector.size() != NUM_ELEMENTS) { throw std::logic_error("Conversion failed!"); }
    }
}

int main() {
    artea::logger.info("Starting performance comparison between artea::cpu::Array and std::vector.");

    benchmark_sort();
    benchmark_merge();
    benchmark_conversion();

    artea::logger.success("--- Benchmark Finished ---");
    artea::logger.info("Note: Performance should be very similar as both containers provide");
    artea::logger.info("random access iterators and manage a contiguous block of memory.");

    return 0;
}