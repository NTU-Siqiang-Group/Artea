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

#include <gtest/gtest.h>
#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fmt/format.h>

using namespace artea;
using namespace artea::cpu;

// ivf_partitions_t / ivf_construct_policy_t are ComputerTraits-dependent
// (<Metric, Dim>), but the partitioning logic is metric/dimension agnostic — this
// is a synthetic test with no vectors. We still resolve a concrete <Metric, Dim> at
// runtime from the user-selected --metric / --dim inputs so the compile-time
// axes stay consistent with the rest of the project rather than being
// hand-pinned.
struct TestConfig {
    std::string dataset_name;
    std::string metric;
    uint32_t    dim;
} g_config;

// Builds the dispatch axes from the --metric / --dim inputs (this test is
// synthetic: no dataset is loaded, so the padded dim is taken from the CLI).
static DatasetInfra dataset_info() {
    return DatasetInfra{parse_metric(g_config.metric), static_cast<vec_dim_t>(g_config.dim)};
}

class IVFPartitionsTest : public ::testing::Test {
protected:

    /**
     * @brief Serial reference implementation for correctness verification.
     * @param part_ids Partition assignments for each vertex
     * @param num_partitions Total number of partitions
     * @return Pair of (partition_offsets, partition_vids)
     */
    static auto serial_from_partition_ids(
        const std::vector<part_id_t>& part_ids,
        const part_num_t num_partitions
    ) -> std::pair<std::vector<vertex_num_t>, std::vector<vertex_id_t>> {
        const vertex_num_t num_vertices = static_cast<vertex_num_t>(part_ids.size());

        // Count vertices per partition
        std::vector<vertex_num_t> partition_counts(num_partitions, 0);
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            partition_counts[part_ids[i]]++;
        }

        // Build offsets (prefix sum)
        std::vector<vertex_num_t> partition_offsets(num_partitions + 1, 0);
        for (part_num_t p = 0; p < num_partitions; ++p) {
            partition_offsets[p + 1] = partition_offsets[p] + partition_counts[p];
        }

        // Allocate and populate vertex IDs
        std::vector<vertex_id_t> partition_vids(num_vertices);
        std::vector<vertex_num_t> write_positions = partition_offsets;
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            part_id_t p = part_ids[i];
            partition_vids[write_positions[p]++] = static_cast<vertex_id_t>(i);
        }

        return {std::move(partition_offsets), std::move(partition_vids)};
    }

    /**
     * @brief Verify that two partition results are equivalent (order within partition may differ).
     */
    static auto verify_partitions_equal(
        const std::vector<vertex_num_t>& offsets1,
        const std::vector<vertex_id_t>& vids1,
        const std::vector<vertex_num_t>& offsets2,
        const std::vector<vertex_id_t>& vids2
    ) -> bool {
        if (offsets1.size() != offsets2.size()) return false;
        if (vids1.size() != vids2.size()) return false;

        const part_num_t num_partitions = static_cast<part_num_t>(offsets1.size() - 1);

        // Check offsets match
        for (part_num_t p = 0; p <= num_partitions; ++p) {
            if (offsets1[p] != offsets2[p]) return false;
        }

        // Check each partition contains the same vertex IDs (order may differ)
        for (part_num_t p = 0; p < num_partitions; ++p) {
            vertex_num_t start = offsets1[p];
            vertex_num_t end = offsets1[p + 1];

            std::vector<vertex_id_t> partition1(vids1.begin() + start, vids1.begin() + end);
            std::vector<vertex_id_t> partition2(vids2.begin() + start, vids2.begin() + end);

            std::sort(partition1.begin(), partition1.end());
            std::sort(partition2.begin(), partition2.end());

            if (partition1 != partition2) return false;
        }

        return true;
    }
};

// Test 1: Small dataset correctness
TEST_F(IVFPartitionsTest, SmallDatasetCorrectness) {
    const vertex_num_t num_vertices = 100;
    const part_num_t num_partitions = 10;

    // Generate random partition assignments
    std::vector<part_id_t> part_ids(num_vertices);
    std::mt19937 rng(42);
    std::uniform_int_distribution<part_id_t> dist(0, num_partitions - 1);
    for (vertex_num_t i = 0; i < num_vertices; ++i) {
        part_ids[i] = dist(rng);
    }

    // Serial reference
    auto [serial_offsets, serial_vids] = serial_from_partition_ids(part_ids, num_partitions);

    infra_dispatch(dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        // Parallel implementation
        ivf_partitions_t<Metric, Dim> ivf_partitions;
        ivf_partitions.from_partition_ids(part_ids, num_partitions);

        // Extract results
        std::vector<vertex_num_t> parallel_offsets(
            ivf_partitions.get_partition_offsets().begin(),
            ivf_partitions.get_partition_offsets().end()
        );
        std::vector<vertex_id_t> parallel_vids(
            ivf_partitions.get_partition_vids().begin(),
            ivf_partitions.get_partition_vids().end()
        );

        // Verify
        EXPECT_TRUE(verify_partitions_equal(serial_offsets, serial_vids, parallel_offsets, parallel_vids));
    });
}

// Test 2: Large dataset correctness
TEST_F(IVFPartitionsTest, LargeDatasetCorrectness) {
    const vertex_num_t num_vertices = 1000000;
    const part_num_t num_partitions = 1000;

    // Generate random partition assignments
    std::vector<part_id_t> part_ids(num_vertices);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<part_id_t> dist(0, num_partitions - 1);
    for (vertex_num_t i = 0; i < num_vertices; ++i) {
        part_ids[i] = dist(rng);
    }

    // Serial reference
    auto [serial_offsets, serial_vids] = serial_from_partition_ids(part_ids, num_partitions);

    infra_dispatch(dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        // Parallel implementation
        ivf_partitions_t<Metric, Dim> ivf_partitions;
        ivf_partitions.from_partition_ids(part_ids, num_partitions);

        // Extract results
        std::vector<vertex_num_t> parallel_offsets(
            ivf_partitions.get_partition_offsets().begin(),
            ivf_partitions.get_partition_offsets().end()
        );
        std::vector<vertex_id_t> parallel_vids(
            ivf_partitions.get_partition_vids().begin(),
            ivf_partitions.get_partition_vids().end()
        );

        // Verify
        EXPECT_TRUE(verify_partitions_equal(serial_offsets, serial_vids, parallel_offsets, parallel_vids));
    });
}

// Test 3: Performance comparison
TEST_F(IVFPartitionsTest, PerformanceComparison) {
    const vertex_num_t num_vertices = 1000000;   // 1M vertices
    const part_num_t num_partitions = 20000;     // 20K partitions

    // Generate random partition assignments
    std::vector<part_id_t> part_ids(num_vertices);
    std::mt19937 rng(99999);
    std::uniform_int_distribution<part_id_t> dist(0, num_partitions - 1);
    for (vertex_num_t i = 0; i < num_vertices; ++i) {
        part_ids[i] = dist(rng);
    }

    auto [serial_duration, parallel_duration] = infra_dispatch(dataset_info(), ARTEA_METRIC_LAMBDA(std::pair<std::chrono::milliseconds, std::chrono::milliseconds>) {
        // Measure serial time
        ivf_partitions_t<Metric, Dim> ivf_partitions_serial;
        auto serial_start = std::chrono::high_resolution_clock::now();
        ivf_partitions_serial.template from_partition_ids<ivf_construct_policy_t<Metric, Dim>::serial>(part_ids, num_partitions);
        auto serial_end = std::chrono::high_resolution_clock::now();
        auto serial_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serial_end - serial_start);

        // Measure parallel time
        ivf_partitions_t<Metric, Dim> ivf_partitions;
        auto parallel_start = std::chrono::high_resolution_clock::now();
        ivf_partitions.template from_partition_ids<ivf_construct_policy_t<Metric, Dim>::parallel>(part_ids, num_partitions);
        auto parallel_end = std::chrono::high_resolution_clock::now();
        auto parallel_duration = std::chrono::duration_cast<std::chrono::milliseconds>(parallel_end - parallel_start);

        // Extract results from serial
        std::vector<vertex_num_t> serial_offsets(
            ivf_partitions_serial.get_partition_offsets().begin(),
            ivf_partitions_serial.get_partition_offsets().end()
        );
        std::vector<vertex_id_t> serial_vids(
            ivf_partitions_serial.get_partition_vids().begin(),
            ivf_partitions_serial.get_partition_vids().end()
        );

        // Extract results from parallel
        std::vector<vertex_num_t> parallel_offsets(
            ivf_partitions.get_partition_offsets().begin(),
            ivf_partitions.get_partition_offsets().end()
        );
        std::vector<vertex_id_t> parallel_vids(
            ivf_partitions.get_partition_vids().begin(),
            ivf_partitions.get_partition_vids().end()
        );

        // Verify correctness
        EXPECT_TRUE(verify_partitions_equal(serial_offsets, serial_vids, parallel_offsets, parallel_vids));
        return {serial_duration, parallel_duration};
    });

    // Print performance results
    double speedup = static_cast<double>(serial_duration.count()) / parallel_duration.count();
    fmt::print("\n=== Performance Comparison ===\n");
    fmt::print("Dataset: {} vertices, {} partitions\n", num_vertices, num_partitions);
    fmt::print("Serial time:   {} ms\n", serial_duration.count());
    fmt::print("Parallel time: {} ms\n", parallel_duration.count());
    fmt::print("Speedup:       {:.2f}x\n", speedup);
    fmt::print("==============================\n\n");

    // Expect at least 1.5x speedup on multi-core systems
    // (actual speedup depends on thread count, data distribution, cache effects)
    EXPECT_GT(speedup, 1.5);
}

// Test 4: Edge case - single partition
TEST_F(IVFPartitionsTest, SinglePartition) {
    const vertex_num_t num_vertices = 1000;
    const part_num_t num_partitions = 1;

    std::vector<part_id_t> part_ids(num_vertices, 0);  // All in partition 0

    auto [serial_offsets, serial_vids] = serial_from_partition_ids(part_ids, num_partitions);

    infra_dispatch(dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        ivf_partitions_t<Metric, Dim> ivf_partitions;
        ivf_partitions.from_partition_ids(part_ids, num_partitions);

        std::vector<vertex_num_t> parallel_offsets(
            ivf_partitions.get_partition_offsets().begin(),
            ivf_partitions.get_partition_offsets().end()
        );
        std::vector<vertex_id_t> parallel_vids(
            ivf_partitions.get_partition_vids().begin(),
            ivf_partitions.get_partition_vids().end()
        );

        EXPECT_TRUE(verify_partitions_equal(serial_offsets, serial_vids, parallel_offsets, parallel_vids));
    });
}

// Test 5: Edge case - empty partitions
TEST_F(IVFPartitionsTest, EmptyPartitions) {
    const vertex_num_t num_vertices = 100;
    const part_num_t num_partitions = 20;

    // Only use partitions 0, 5, 10, 15 (others are empty)
    std::vector<part_id_t> part_ids(num_vertices);
    std::mt19937 rng(777);
    std::vector<part_id_t> used_partitions = {0, 5, 10, 15};
    std::uniform_int_distribution<size_t> dist(0, used_partitions.size() - 1);
    for (vertex_num_t i = 0; i < num_vertices; ++i) {
        part_ids[i] = used_partitions[dist(rng)];
    }

    auto [serial_offsets, serial_vids] = serial_from_partition_ids(part_ids, num_partitions);

    infra_dispatch(dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        ivf_partitions_t<Metric, Dim> ivf_partitions;
        ivf_partitions.from_partition_ids(part_ids, num_partitions);

        std::vector<vertex_num_t> parallel_offsets(
            ivf_partitions.get_partition_offsets().begin(),
            ivf_partitions.get_partition_offsets().end()
        );
        std::vector<vertex_id_t> parallel_vids(
            ivf_partitions.get_partition_vids().begin(),
            ivf_partitions.get_partition_vids().end()
        );

        EXPECT_TRUE(verify_partitions_equal(serial_offsets, serial_vids, parallel_offsets, parallel_vids));

        // Verify empty partitions have zero size
        for (part_id_t p : {1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19}) {
            EXPECT_EQ(ivf_partitions.get_partition_size(p), 0);
        }
    });
}

// Test 6: API functionality
TEST_F(IVFPartitionsTest, APIFunctionality) {
    const vertex_num_t num_vertices = 50;
    const part_num_t num_partitions = 5;

    std::vector<part_id_t> part_ids(num_vertices);
    for (vertex_num_t i = 0; i < num_vertices; ++i) {
        part_ids[i] = i % num_partitions;  // Round-robin assignment
    }

    infra_dispatch(dataset_info(), ARTEA_METRIC_LAMBDA(void) {
        ivf_partitions_t<Metric, Dim> ivf_partitions;
        ivf_partitions.from_partition_ids(part_ids, num_partitions);

        // Test get_num_partitions
        EXPECT_EQ(ivf_partitions.get_num_partitions(), num_partitions);

        // Test get_partition_size
        for (part_num_t p = 0; p < num_partitions; ++p) {
            EXPECT_EQ(ivf_partitions.get_partition_size(p), num_vertices / num_partitions);
        }

        // Test get_partition_vids
        for (part_num_t p = 0; p < num_partitions; ++p) {
            auto vids = ivf_partitions.get_partition_vids(p);
            EXPECT_EQ(vids.size(), num_vertices / num_partitions);

            // Verify all vertices in this partition have correct partition ID
            for (vertex_id_t vid : vids) {
                EXPECT_EQ(part_ids[vid], p);
            }
        }
    });
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_ivf_partitions");
    program.add_argument("-d", "--dataset")
        .default_value(std::string("sift-1m"))
        .help("Dataset name (kept for CLI consistency; the synthetic partition "
              "logic loads no vectors).");
    program.add_argument("--metric")
        .default_value(std::string("euclidean"))
        .help("Distance metric: 'euclidean', 'inner_product', or 'cosine'");
    program.add_argument("--dim")
        .default_value(128u).scan<'u', uint32_t>()
        .help("SIMD-padded dimension used to pick the compile-time <Metric, Dim> "
              "pair (no dataset is loaded; default 128 = sift-1m's padded dim).");

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.metric = program.get<std::string>("--metric");
    g_config.dim = program.get<uint32_t>("--dim");

    return RUN_ALL_TESTS();
}
