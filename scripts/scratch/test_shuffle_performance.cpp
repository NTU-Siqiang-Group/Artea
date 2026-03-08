// Test std::shuffle performance with 1 million IDs

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>

int main() {
    const size_t num_ids = 1'000'000;

    std::cout << "Testing std::shuffle performance with " << num_ids << " IDs\n";
    std::cout << std::string(60, '=') << "\n\n";

    // Initialize vector with sequential IDs
    std::vector<uint32_t> ids(num_ids);
    for (size_t i = 0; i < num_ids; ++i) {
        ids[i] = static_cast<uint32_t>(i);
    }

    // Test with different random engines
    std::random_device rd;

    // Test 1: std::mt19937
    {
        std::mt19937 gen(rd());
        auto start = std::chrono::high_resolution_clock::now();
        std::shuffle(ids.begin(), ids.end(), gen);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "std::mt19937:\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(3)
                  << duration.count() / 1000.0 << " ms\n";
        std::cout << "  First 10 IDs: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << ids[i] << " ";
        }
        std::cout << "\n\n";
    }

    // Reset IDs
    for (size_t i = 0; i < num_ids; ++i) {
        ids[i] = static_cast<uint32_t>(i);
    }

    // Test 2: std::mt19937_64
    {
        std::mt19937_64 gen(rd());
        auto start = std::chrono::high_resolution_clock::now();
        std::shuffle(ids.begin(), ids.end(), gen);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "std::mt19937_64:\n";
        std::cout << "  Time: " << std::fixed << std::setprecision(3)
                  << duration.count() / 1000.0 << " ms\n";
        std::cout << "  First 10 IDs: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << ids[i] << " ";
        }
        std::cout << "\n\n";
    }

    // Reset IDs
    for (size_t i = 0; i < num_ids; ++i) {
        ids[i] = static_cast<uint32_t>(i);
    }

    // Test 3: Multiple shuffles
    {
        std::mt19937 gen(rd());
        const int num_runs = 10;

        auto start = std::chrono::high_resolution_clock::now();
        for (int run = 0; run < num_runs; ++run) {
            std::shuffle(ids.begin(), ids.end(), gen);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Multiple shuffles (" << num_runs << " runs):\n";
        std::cout << "  Total time: " << std::fixed << std::setprecision(3)
                  << duration.count() / 1000.0 << " ms\n";
        std::cout << "  Average per shuffle: " << std::fixed << std::setprecision(3)
                  << duration.count() / 1000.0 / num_runs << " ms\n";
        std::cout << "  First 10 IDs: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << ids[i] << " ";
        }
        std::cout << "\n\n";
    }

    // Test 4: Different sizes
    std::cout << "Testing different sizes:\n";
    std::vector<size_t> sizes = {10'000, 100'000, 1'000'000, 10'000'000};

    for (size_t size : sizes) {
        std::vector<uint32_t> test_ids(size);
        for (size_t i = 0; i < size; ++i) {
            test_ids[i] = static_cast<uint32_t>(i);
        }

        std::mt19937 gen(rd());
        auto start = std::chrono::high_resolution_clock::now();
        std::shuffle(test_ids.begin(), test_ids.end(), gen);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  " << std::setw(10) << size << " IDs: "
                  << std::fixed << std::setprecision(3)
                  << std::setw(10) << duration.count() / 1000.0 << " ms\n";
    }

    return 0;
}
