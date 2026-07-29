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
 * @FilePath: /Artea/tests/test_four_ary_heap.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: GoogleTest suite for FourAryHeap correctness verification.
 */

#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <functional>
#include <filesystem>
#include <memory>
#include <gtest/gtest.h>
#include <argparse/argparse.hpp>
#include <fmt/format.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>

using namespace artea;
using namespace artea::cpu;

// --- Type Definitions ---
// The four-ary heap is metric/dim-agnostic, but the metric + padded dimension
// are now compile-time traits resolved from the user-specified dataset. We pull
// candidate_entry_t (and friends) out of the dispatched <Metric, Dim> router_traits
// so the heap is exercised under exactly the dataset's compile-time pair.
template <DistanceMetricsT Metric, vec_dim_t Dim>
using candidate_entry_for = typename router_traits_t<Metric, Dim>::candidate_entry_t;

template <DistanceMetricsT Metric, vec_dim_t Dim, typename Compare>
using four_ary_heap_for = typename router_traits_t<Metric, Dim>::template four_ary_heap_t<Compare>;

template <DistanceMetricsT Metric, vec_dim_t Dim>
using max_heap_for = four_ary_heap_for<Metric, Dim, std::less<candidate_entry_for<Metric, Dim>>>;

template <DistanceMetricsT Metric, vec_dim_t Dim>
using min_heap_for = four_ary_heap_for<Metric, Dim, std::greater<candidate_entry_for<Metric, Dim>>>;

// --- Global Configuration ---
struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    std::string metric;
    uint32_t num_operations;  // Number of push/pop operations
    uint32_t seed;            // Random seed for reproducibility
} g_config;

// --- DataProvider: loads the dataset purely to supply the padded dim; with
// the --metric input it fixes the compile-time (metric, padded-dim) pair the
// heap should be instantiated under. ---
class DataProvider {
public:
    static DataProvider& instance() {
        static DataProvider inst;
        return inst;
    }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }
        ARTEA_INFO(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset_ = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);

        const auto& base_vecs = dataset_->get_base_vecs();
        dataset_info_ = DatasetInfra{parse_metric(g_config.metric), base_vecs.get_vec_dim()};
    }

    DatasetInfra get_dataset_info() const { return dataset_info_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    DatasetInfra dataset_info_{};
};

// --- Helper Functions ---

/**
 * @brief Generate random candidate entries for testing.
 */
template <DistanceMetricsT Metric, vec_dim_t Dim>
std::vector<candidate_entry_for<Metric, Dim>> generate_random_entries(uint32_t count, uint32_t seed) {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using vertex_id_t = typename router_traits_t<Metric, Dim>::vertex_id_t;
    using distance_t = typename router_traits_t<Metric, Dim>::distance_t;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<vertex_id_t> id_dist(0, 1000000);
    std::uniform_real_distribution<distance_t> dist_dist(0.0f, 1000.0f);

    std::vector<candidate_entry_t> entries;
    entries.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        vertex_id_t id = id_dist(rng);
        distance_t dist = dist_dist(rng);
        entries.push_back(candidate_entry_t::make_entry(id, dist));
    }

    return entries;
}

// --- Test Fixture ---

class FourAryHeapTest : public ::testing::Test {
protected:
    void SetUp() override {
        ARTEA_INFO("----------------------------------------------------------");
    }

    void TearDown() override {
        ARTEA_INFO("----------------------------------------------------------");
    }
};

// --- Tests for Max-Heap (std::less) ---

template <DistanceMetricsT Metric, vec_dim_t Dim>
void run_MaxHeapBasicOperations() {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using max_heap_t = max_heap_for<Metric, Dim>;
    using distance_t = typename router_traits_t<Metric, Dim>::distance_t;

    ARTEA_INFO(" -> Testing Max-Heap Basic Operations...");

    max_heap_t heap(candidate_entry_t::make_min_entry());

    // Test empty
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0);

    // Push many elements with various distances (50 elements for thorough testing)
    std::vector<distance_t> distances = {
        10.0f, 50.0f, 20.0f, 80.0f, 15.0f, 90.0f, 5.0f, 70.0f,
        30.0f, 60.0f, 25.0f, 85.0f, 40.0f, 95.0f, 35.0f, 75.0f,
        45.0f, 65.0f, 55.0f, 100.0f, 12.0f, 48.0f, 22.0f, 78.0f,
        18.0f, 88.0f, 8.0f, 68.0f, 32.0f, 62.0f, 28.0f, 82.0f,
        42.0f, 92.0f, 38.0f, 72.0f, 47.0f, 67.0f, 57.0f, 97.0f,
        13.0f, 53.0f, 23.0f, 83.0f, 17.0f, 87.0f, 7.0f, 73.0f,
        33.0f, 63.0f
    };

    for (size_t i = 0; i < distances.size(); ++i) {
        heap.push(candidate_entry_t::make_entry(i, distances[i]));
    }

    EXPECT_FALSE(heap.empty());
    EXPECT_EQ(heap.size(), distances.size());

    // Max-heap should have largest element on top
    EXPECT_FLOAT_EQ(heap.top().get_distance(), 100.0f);

    // Pop all elements and verify they come out in descending order
    std::vector<distance_t> popped_distances;
    while (!heap.empty()) {
        popped_distances.push_back(heap.top().get_distance());
        heap.pop();
    }

    // Verify descending order
    for (size_t i = 1; i < popped_distances.size(); ++i) {
        EXPECT_GE(popped_distances[i-1], popped_distances[i])
            << "Max-heap order violated at position " << i;
    }

    EXPECT_TRUE(heap.empty());

    ARTEA_SUCCESS("Max-Heap Basic Operations passed.");
}

TEST_F(FourAryHeapTest, MaxHeapBasicOperations) {
    infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) { run_MaxHeapBasicOperations<Metric, Dim>(); });
}

template <DistanceMetricsT Metric, vec_dim_t Dim>
void run_MaxHeapVsStdPriorityQueue() {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using max_heap_t = max_heap_for<Metric, Dim>;

    ARTEA_INFO(fmt::format(" -> Testing Max-Heap vs std::priority_queue with {} operations...",
                            g_config.num_operations));

    auto entries = generate_random_entries<Metric, Dim>(g_config.num_operations, g_config.seed);

    // Initialize both heaps
    max_heap_t four_ary_heap(candidate_entry_t::make_min_entry());
    std::priority_queue<candidate_entry_t, std::vector<candidate_entry_t>, std::less<candidate_entry_t>> std_heap;

    // Push all entries
    for (const auto& entry : entries) {
        four_ary_heap.push(entry);
        std_heap.push(entry);
    }

    EXPECT_EQ(four_ary_heap.size(), std_heap.size());

    // Pop all and compare distances (IDs may differ for equal distances due to heap structure)
    uint32_t compared = 0;
    while (!four_ary_heap.empty() && !std_heap.empty()) {
        const auto& four_ary_top = four_ary_heap.top();
        const auto& std_top = std_heap.top();

        // Only compare distances - heap order for equal distances is implementation-defined
        EXPECT_FLOAT_EQ(four_ary_top.get_distance(), std_top.get_distance())
            << "Distance mismatch at comparison " << compared;

        four_ary_heap.pop();
        std_heap.pop();
        compared++;
    }

    EXPECT_TRUE(four_ary_heap.empty());
    EXPECT_TRUE(std_heap.empty());
    EXPECT_EQ(compared, g_config.num_operations);

    ARTEA_SUCCESS(fmt::format("Max-Heap matched std::priority_queue for {} operations.", compared));
}

TEST_F(FourAryHeapTest, MaxHeapVsStdPriorityQueue) {
    infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) { run_MaxHeapVsStdPriorityQueue<Metric, Dim>(); });
}

template <DistanceMetricsT Metric, vec_dim_t Dim>
void run_MaxHeapInitialize() {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using max_heap_t = max_heap_for<Metric, Dim>;

    ARTEA_INFO(" -> Testing Max-Heap Initialize (Floyd's construction)...");

    auto entries = generate_random_entries<Metric, Dim>(g_config.num_operations, g_config.seed);

    // Initialize FourAryHeap using Floyd's construction
    max_heap_t four_ary_heap(candidate_entry_t::make_min_entry());
    four_ary_heap.initialize(entries);

    // Build std::priority_queue by pushing all elements
    std::priority_queue<candidate_entry_t, std::vector<candidate_entry_t>, std::less<candidate_entry_t>> std_heap;
    for (const auto& entry : entries) {
        std_heap.push(entry);
    }

    EXPECT_EQ(four_ary_heap.size(), std_heap.size());

    // Pop all and compare
    uint32_t compared = 0;
    while (!four_ary_heap.empty() && !std_heap.empty()) {
        EXPECT_FLOAT_EQ(four_ary_heap.top().get_distance(), std_heap.top().get_distance())
            << "Mismatch at comparison " << compared;

        four_ary_heap.pop();
        std_heap.pop();
        compared++;
    }

    EXPECT_EQ(compared, g_config.num_operations);

    ARTEA_SUCCESS("Max-Heap Initialize passed.");
}

TEST_F(FourAryHeapTest, MaxHeapInitialize) {
    infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) { run_MaxHeapInitialize<Metric, Dim>(); });
}

// --- Tests for Min-Heap (std::greater) ---

template <DistanceMetricsT Metric, vec_dim_t Dim>
void run_MinHeapBasicOperations() {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using min_heap_t = min_heap_for<Metric, Dim>;
    using distance_t = typename router_traits_t<Metric, Dim>::distance_t;

    ARTEA_INFO(" -> Testing Min-Heap Basic Operations...");

    min_heap_t heap(candidate_entry_t::make_invalid_entry());

    // Test empty
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0);

    // Push many elements with various distances (50 elements for thorough testing)
    std::vector<distance_t> distances = {
        10.0f, 50.0f, 20.0f, 80.0f, 15.0f, 90.0f, 5.0f, 70.0f,
        30.0f, 60.0f, 25.0f, 85.0f, 40.0f, 95.0f, 35.0f, 75.0f,
        45.0f, 65.0f, 55.0f, 100.0f, 12.0f, 48.0f, 22.0f, 78.0f,
        18.0f, 88.0f, 8.0f, 68.0f, 32.0f, 62.0f, 28.0f, 82.0f,
        42.0f, 92.0f, 38.0f, 72.0f, 47.0f, 67.0f, 57.0f, 97.0f,
        13.0f, 53.0f, 23.0f, 83.0f, 17.0f, 87.0f, 7.0f, 73.0f,
        33.0f, 63.0f
    };

    for (size_t i = 0; i < distances.size(); ++i) {
        heap.push(candidate_entry_t::make_entry(i, distances[i]));
    }

    EXPECT_FALSE(heap.empty());
    EXPECT_EQ(heap.size(), distances.size());

    // Min-heap should have smallest element on top
    EXPECT_FLOAT_EQ(heap.top().get_distance(), 5.0f);

    // Pop all elements and verify they come out in ascending order
    std::vector<distance_t> popped_distances;
    while (!heap.empty()) {
        popped_distances.push_back(heap.top().get_distance());
        heap.pop();
    }

    // Verify ascending order
    for (size_t i = 1; i < popped_distances.size(); ++i) {
        EXPECT_LE(popped_distances[i-1], popped_distances[i])
            << "Min-heap order violated at position " << i;
    }

    EXPECT_TRUE(heap.empty());

    ARTEA_SUCCESS("Min-Heap Basic Operations passed.");
}

TEST_F(FourAryHeapTest, MinHeapBasicOperations) {
    infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) { run_MinHeapBasicOperations<Metric, Dim>(); });
}

template <DistanceMetricsT Metric, vec_dim_t Dim>
void run_MinHeapVsStdPriorityQueue() {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using min_heap_t = min_heap_for<Metric, Dim>;

    ARTEA_INFO(fmt::format(" -> Testing Min-Heap vs std::priority_queue with {} operations...",
                            g_config.num_operations));

    auto entries = generate_random_entries<Metric, Dim>(g_config.num_operations, g_config.seed);

    // Initialize both heaps
    min_heap_t four_ary_heap(candidate_entry_t::make_invalid_entry());
    std::priority_queue<candidate_entry_t, std::vector<candidate_entry_t>, std::greater<candidate_entry_t>> std_heap;

    // Push all entries
    for (const auto& entry : entries) {
        four_ary_heap.push(entry);
        std_heap.push(entry);
    }

    EXPECT_EQ(four_ary_heap.size(), std_heap.size());

    // Pop all and compare distances (IDs may differ for equal distances due to heap structure)
    uint32_t compared = 0;
    while (!four_ary_heap.empty() && !std_heap.empty()) {
        const auto& four_ary_top = four_ary_heap.top();
        const auto& std_top = std_heap.top();

        // Only compare distances - heap order for equal distances is implementation-defined
        EXPECT_FLOAT_EQ(four_ary_top.get_distance(), std_top.get_distance())
            << "Distance mismatch at comparison " << compared;

        four_ary_heap.pop();
        std_heap.pop();
        compared++;
    }

    EXPECT_TRUE(four_ary_heap.empty());
    EXPECT_TRUE(std_heap.empty());
    EXPECT_EQ(compared, g_config.num_operations);

    ARTEA_SUCCESS(fmt::format("Min-Heap matched std::priority_queue for {} operations.", compared));
}

TEST_F(FourAryHeapTest, MinHeapVsStdPriorityQueue) {
    infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) { run_MinHeapVsStdPriorityQueue<Metric, Dim>(); });
}

template <DistanceMetricsT Metric, vec_dim_t Dim>
void run_MinHeapInitialize() {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using min_heap_t = min_heap_for<Metric, Dim>;

    ARTEA_INFO(" -> Testing Min-Heap Initialize (Floyd's construction)...");

    auto entries = generate_random_entries<Metric, Dim>(g_config.num_operations, g_config.seed);

    // Initialize FourAryHeap using Floyd's construction
    min_heap_t four_ary_heap(candidate_entry_t::make_invalid_entry());
    four_ary_heap.initialize(entries);

    // Build std::priority_queue by pushing all elements
    std::priority_queue<candidate_entry_t, std::vector<candidate_entry_t>, std::greater<candidate_entry_t>> std_heap;
    for (const auto& entry : entries) {
        std_heap.push(entry);
    }

    EXPECT_EQ(four_ary_heap.size(), std_heap.size());

    // Pop all and compare
    uint32_t compared = 0;
    while (!four_ary_heap.empty() && !std_heap.empty()) {
        EXPECT_FLOAT_EQ(four_ary_heap.top().get_distance(), std_heap.top().get_distance())
            << "Mismatch at comparison " << compared;

        four_ary_heap.pop();
        std_heap.pop();
        compared++;
    }

    EXPECT_EQ(compared, g_config.num_operations);

    ARTEA_SUCCESS("Min-Heap Initialize passed.");
}

TEST_F(FourAryHeapTest, MinHeapInitialize) {
    infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) { run_MinHeapInitialize<Metric, Dim>(); });
}

// --- Edge Cases ---

template <DistanceMetricsT Metric, vec_dim_t Dim>
void run_EdgeCaseSingleElement() {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using max_heap_t = max_heap_for<Metric, Dim>;

    ARTEA_INFO(" -> Testing Edge Case: Single Element...");

    max_heap_t heap(candidate_entry_t::make_min_entry());

    auto entry = candidate_entry_t::make_entry(42, 3.14f);
    heap.push(entry);

    EXPECT_EQ(heap.size(), 1);
    EXPECT_FLOAT_EQ(heap.top().get_distance(), 3.14f);
    EXPECT_EQ(heap.top().get_vid(), 42);

    heap.pop();
    EXPECT_TRUE(heap.empty());

    ARTEA_SUCCESS("Single Element test passed.");
}

TEST_F(FourAryHeapTest, EdgeCaseSingleElement) {
    infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) { run_EdgeCaseSingleElement<Metric, Dim>(); });
}

template <DistanceMetricsT Metric, vec_dim_t Dim>
void run_EdgeCaseDuplicateDistances() {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using max_heap_t = max_heap_for<Metric, Dim>;
    using vertex_id_t = typename router_traits_t<Metric, Dim>::vertex_id_t;

    ARTEA_INFO(" -> Testing Edge Case: Duplicate Distances...");

    max_heap_t heap(candidate_entry_t::make_min_entry());

    // Push entries with same distance but different IDs (30+ entries for thorough testing)
    // 20 entries with distance 10.0
    for (vertex_id_t i = 1; i <= 20; ++i) {
        heap.push(candidate_entry_t::make_entry(i, 10.0f));
    }
    // 10 entries with distance 20.0
    for (vertex_id_t i = 21; i <= 30; ++i) {
        heap.push(candidate_entry_t::make_entry(i, 20.0f));
    }
    // 5 entries with distance 30.0
    for (vertex_id_t i = 31; i <= 35; ++i) {
        heap.push(candidate_entry_t::make_entry(i, 30.0f));
    }

    EXPECT_EQ(heap.size(), 35);
    EXPECT_FLOAT_EQ(heap.top().get_distance(), 30.0f);

    // Pop all 30.0 entries
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(heap.top().get_distance(), 30.0f);
        heap.pop();
    }

    // Now top should be 20.0
    EXPECT_FLOAT_EQ(heap.top().get_distance(), 20.0f);

    // Pop all 20.0 entries
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(heap.top().get_distance(), 20.0f);
        heap.pop();
    }

    // Now top should be 10.0
    EXPECT_FLOAT_EQ(heap.top().get_distance(), 10.0f);

    // All remaining should have distance 10.0
    std::vector<vertex_id_t> ids;
    while (!heap.empty()) {
        EXPECT_FLOAT_EQ(heap.top().get_distance(), 10.0f);
        ids.push_back(heap.top().get_vid());
        heap.pop();
    }

    // Should have collected 20 IDs
    EXPECT_EQ(ids.size(), 20);

    ARTEA_SUCCESS("Duplicate Distances test passed.");
}

TEST_F(FourAryHeapTest, EdgeCaseDuplicateDistances) {
    infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) { run_EdgeCaseDuplicateDistances<Metric, Dim>(); });
}

template <DistanceMetricsT Metric, vec_dim_t Dim>
void run_EdgeCaseClearAndReuse() {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using max_heap_t = max_heap_for<Metric, Dim>;
    using vertex_id_t = typename router_traits_t<Metric, Dim>::vertex_id_t;
    using distance_t = typename router_traits_t<Metric, Dim>::distance_t;

    ARTEA_INFO(" -> Testing Edge Case: Clear and Reuse...");

    max_heap_t heap(candidate_entry_t::make_min_entry());

    // First batch (30 elements)
    for (vertex_id_t i = 1; i <= 30; ++i) {
        heap.push(candidate_entry_t::make_entry(i, static_cast<distance_t>(i * 10.0f)));
    }
    EXPECT_EQ(heap.size(), 30);
    EXPECT_FLOAT_EQ(heap.top().get_distance(), 300.0f);

    // Clear
    heap.clear();
    EXPECT_TRUE(heap.empty());
    EXPECT_EQ(heap.size(), 0);

    // Second batch (40 elements)
    for (vertex_id_t i = 31; i <= 70; ++i) {
        heap.push(candidate_entry_t::make_entry(i, static_cast<distance_t>(i * 5.0f)));
    }
    EXPECT_EQ(heap.size(), 40);
    EXPECT_FLOAT_EQ(heap.top().get_distance(), 350.0f);

    // Verify all elements are from second batch
    std::vector<distance_t> distances;
    while (!heap.empty()) {
        distances.push_back(heap.top().get_distance());
        heap.pop();
    }

    // Should have 40 elements in descending order
    EXPECT_EQ(distances.size(), 40);
    for (size_t i = 1; i < distances.size(); ++i) {
        EXPECT_GE(distances[i-1], distances[i]);
    }

    ARTEA_SUCCESS("Clear and Reuse test passed.");
}

TEST_F(FourAryHeapTest, EdgeCaseClearAndReuse) {
    infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) { run_EdgeCaseClearAndReuse<Metric, Dim>(); });
}

template <DistanceMetricsT Metric, vec_dim_t Dim>
void run_StressTestMixedOperations() {
    using candidate_entry_t = candidate_entry_for<Metric, Dim>;
    using max_heap_t = max_heap_for<Metric, Dim>;
    using vertex_id_t = typename router_traits_t<Metric, Dim>::vertex_id_t;
    using distance_t = typename router_traits_t<Metric, Dim>::distance_t;

    ARTEA_INFO(fmt::format(" -> Stress Test: Mixed Push/Pop Operations with {} ops...",
                            g_config.num_operations));

    std::mt19937 rng(g_config.seed);
    std::uniform_int_distribution<vertex_id_t> id_dist(0, 1000000);
    std::uniform_real_distribution<distance_t> dist_dist(0.0f, 1000.0f);
    std::uniform_int_distribution<int> op_dist(0, 1); // 0 = push, 1 = pop

    max_heap_t four_ary_heap(candidate_entry_t::make_min_entry());
    std::priority_queue<candidate_entry_t, std::vector<candidate_entry_t>, std::less<candidate_entry_t>> std_heap;

    for (uint32_t i = 0; i < g_config.num_operations; ++i) {
        int op = op_dist(rng);

        if (op == 0 || four_ary_heap.empty()) {
            // Push operation
            auto entry = candidate_entry_t::make_entry(id_dist(rng), dist_dist(rng));
            four_ary_heap.push(entry);
            std_heap.push(entry);
        } else {
            // Pop operation
            ASSERT_FALSE(four_ary_heap.empty());
            ASSERT_FALSE(std_heap.empty());

            EXPECT_FLOAT_EQ(four_ary_heap.top().get_distance(), std_heap.top().get_distance());

            four_ary_heap.pop();
            std_heap.pop();
        }

        EXPECT_EQ(four_ary_heap.size(), std_heap.size());
    }

    ARTEA_SUCCESS(fmt::format("Stress Test passed with {} operations.", g_config.num_operations));
}

TEST_F(FourAryHeapTest, StressTestMixedOperations) {
    infra_dispatch(DataProvider::instance().get_dataset_info(), ARTEA_METRIC_LAMBDA(void) { run_StressTestMixedOperations<Metric, Dim>(); });
}

// --- Main ---

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize Argument Parser
    argparse::ArgumentParser program("test_four_ary_heap");

    program.add_argument("-c", "--config")
        .default_value(artea::default_dataset_config_path());
    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"));
    program.add_argument("--metric")
        .default_value(std::string("euclidean"))
        .help("Distance metric: 'euclidean', 'inner_product', or 'cosine'");

    program.add_argument("-n", "--num_operations")
        .help("Number of operations for stress tests")
        .scan<'u', uint32_t>()
        .default_value(uint32_t{10000});

    program.add_argument("-s", "--seed")
        .help("Random seed for reproducibility")
        .scan<'u', uint32_t>()
        .default_value(uint32_t{42});

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Populate Global Config
    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.metric = program.get<std::string>("--metric");
    g_config.num_operations = program.get<uint32_t>("--num_operations");
    g_config.seed = program.get<uint32_t>("--seed");

    ARTEA_INFO("==========================================================");
    ARTEA_INFO("      Starting FourAryHeap Correctness Suite");
    ARTEA_INFO(fmt::format("      Config: Dataset={}, Operations={}, Seed={}",
                           g_config.dataset_name, g_config.num_operations, g_config.seed));
    ARTEA_INFO("==========================================================");

    DataProvider::instance().init();

    return RUN_ALL_TESTS();
}
