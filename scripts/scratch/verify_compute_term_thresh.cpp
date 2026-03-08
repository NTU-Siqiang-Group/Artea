// Copyright 2026 Weitang Ye
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * @FilePath: /Artea/scripts/scratch/verify_compute_term_thresh.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Verify compute_term_thresh values match documentation
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <boost/math/distributions/normal.hpp>

// Compute term_thresh using the same formula as LBGreedyVG
uint32_t compute_term_thresh(float coverage_ratio, float confidence, uint32_t batch_size) {
    double p = 1.0 - static_cast<double>(coverage_ratio);

    boost::math::normal_distribution<double> normal(0.0, 1.0);
    double z_value = boost::math::quantile(normal, static_cast<double>(confidence));

    double n = static_cast<double>(batch_size);
    double mean = n * p;
    double std_dev = std::sqrt(n * p * (1.0 - p));
    double term_thresh_double = mean - z_value * std_dev;

    return static_cast<uint32_t>(std::max(1.0, std::floor(term_thresh_double)));
}

int main() {
    std::cout << "=" << std::string(80, '=') << std::endl;
    std::cout << "VERIFY compute_term_thresh DOCUMENTATION VALUES" << std::endl;
    std::cout << "=" << std::string(80, '=') << std::endl;
    std::cout << std::endl;

    // Test cases from documentation (95% coverage, 96% confidence)
    float coverage_ratio = 0.95f;
    float confidence = 0.96f;

    struct TestCase {
        uint32_t batch_size;
        double expected_mean;
        double expected_std_dev;
        uint32_t expected_term_thresh;
    };

    TestCase test_cases[] = {
        {512, 25.6, 4.93, 16},
        {1024, 51.2, 6.97, 38},
        {2048, 102.4, 9.86, 85}
    };

    std::cout << "Testing with coverage_ratio=" << coverage_ratio
              << ", confidence=" << confidence << std::endl;
    std::cout << std::endl;

    // Compute z-value
    boost::math::normal_distribution<double> normal(0.0, 1.0);
    double z_value = boost::math::quantile(normal, static_cast<double>(confidence));
    std::cout << "Z-value for " << confidence * 100 << "% confidence: "
              << std::fixed << std::setprecision(4) << z_value << std::endl;
    std::cout << std::endl;

    // Compute uncovered rate
    double p = 1.0 - static_cast<double>(coverage_ratio);
    std::cout << "Uncovered rate (p): " << p << std::endl;
    std::cout << std::endl;

    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(12) << "Batch Size"
              << std::setw(15) << "Mean (μ)"
              << std::setw(15) << "Std Dev (σ)"
              << std::setw(20) << "term_thresh"
              << std::setw(10) << "Match"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    bool all_passed = true;

    for (const auto& test : test_cases) {
        uint32_t batch_size = test.batch_size;

        // Compute actual values
        double n = static_cast<double>(batch_size);
        double mean = n * p;
        double std_dev = std::sqrt(n * p * (1.0 - p));
        uint32_t term_thresh = compute_term_thresh(coverage_ratio, confidence, batch_size);

        // Check if values match documentation (with small tolerance for floating point)
        bool mean_match = std::abs(mean - test.expected_mean) < 0.1;
        bool std_dev_match = std::abs(std_dev - test.expected_std_dev) < 0.1;
        bool term_thresh_match = (term_thresh == test.expected_term_thresh);
        bool all_match = mean_match && std_dev_match && term_thresh_match;

        std::cout << std::left << std::setw(12) << batch_size
                  << std::setw(15) << std::fixed << std::setprecision(2) << mean
                  << std::setw(15) << std::fixed << std::setprecision(2) << std_dev
                  << std::setw(20) << term_thresh
                  << std::setw(10) << (all_match ? "✓" : "✗")
                  << std::endl;

        if (!all_match) {
            std::cout << "  Expected: mean=" << test.expected_mean
                      << ", std_dev=" << test.expected_std_dev
                      << ", term_thresh=" << test.expected_term_thresh << std::endl;
            all_passed = false;
        }
    }

    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::endl;

    std::cout << "=" << std::string(80, '=') << std::endl;
    std::cout << "All tests passed: " << (all_passed ? "✓ YES" : "✗ NO") << std::endl;
    std::cout << "=" << std::string(80, '=') << std::endl;

    return all_passed ? 0 : 1;
}
