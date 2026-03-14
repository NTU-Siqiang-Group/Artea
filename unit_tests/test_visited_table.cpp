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
 * @FilePath: /Artea/tests/test_visited_table.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: GoogleTest suite for ThreadLocalBitmap, VersionTagTable,
 *               and VisitedTablePool correctness verification.
 */

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <atomic>
#include <gtest/gtest.h>
#include <argparse/argparse.hpp>
#include <fmt/format.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/containers/thread_local_bitmap.hpp>
#include <artea/cpu/containers/version_tag_table.hpp>

using namespace artea;
using namespace artea::cpu;

// --- Type Definitions ---
using vec_num_t = uint32_t;
using vec_ele_t = float;
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using index_traits_t = IndexTraits<base_traits_t>;
using router_traits_t = RouterTraits<computer_traits_t, index_traits_t, false>;

// --- Global Configuration ---
struct TestConfig {
    uint32_t num_elements;
    uint32_t seed;
} g_config;

// ============================================================================
// Test Fixture
// ============================================================================

class VisitedTableTest : public ::testing::Test {
protected:
    void SetUp() override {
        logger.info("----------------------------------------------------------");
    }
    void TearDown() override {
        logger.info("----------------------------------------------------------");
    }
};

// ============================================================================
// 1. Single-Thread: ThreadLocalBitmap
// ============================================================================

TEST_F(VisitedTableTest, Bitmap_SetAndTest) {
    logger.info(" -> [ThreadLocalBitmap] Set and Test");

    const size_t N = g_config.num_elements;
    ThreadLocalBitmap bitmap(N);

    // Initially all bits should be unset
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FALSE(bitmap.test(i)) << "Bit " << i << " should be unset initially";
    }

    // Set every other bit
    for (size_t i = 0; i < N; i += 2) {
        bitmap.set(i);
    }

    // Verify pattern
    for (size_t i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            EXPECT_TRUE(bitmap.test(i)) << "Bit " << i << " should be set";
        } else {
            EXPECT_FALSE(bitmap.test(i)) << "Bit " << i << " should be unset";
        }
    }

    logger.success(" [ThreadLocalBitmap] Set and Test passed.");
}

TEST_F(VisitedTableTest, Bitmap_Clear) {
    logger.info(" -> [ThreadLocalBitmap] Clear");

    const size_t N = g_config.num_elements;
    ThreadLocalBitmap bitmap(N);

    // Set all bits
    for (size_t i = 0; i < N; ++i) {
        bitmap.set(i);
    }

    // Clear
    bitmap.clear();

    // All bits should be unset
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FALSE(bitmap.test(i)) << "Bit " << i << " should be unset after clear";
    }

    logger.success(" [ThreadLocalBitmap] Clear passed.");
}

TEST_F(VisitedTableTest, Bitmap_RandomPattern) {
    logger.info(" -> [ThreadLocalBitmap] Random Pattern");

    const size_t N = g_config.num_elements;
    ThreadLocalBitmap bitmap(N);

    // Generate random indices to set
    std::mt19937 rng(g_config.seed);
    std::vector<bool> expected(N, false);
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    const size_t num_sets = N / 2;
    for (size_t i = 0; i < num_sets; ++i) {
        size_t idx = dist(rng);
        bitmap.set(idx);
        expected[idx] = true;
    }

    // Verify
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(bitmap.test(i), expected[i])
            << "Mismatch at index " << i;
    }

    logger.success(" [ThreadLocalBitmap] Random Pattern passed.");
}

TEST_F(VisitedTableTest, Bitmap_RepeatedClearAndReuse) {
    logger.info(" -> [ThreadLocalBitmap] Repeated Clear and Reuse");

    const size_t N = g_config.num_elements;
    ThreadLocalBitmap bitmap(N);

    for (int round = 0; round < 10; ++round) {
        // Set some bits
        for (size_t i = round; i < N; i += 10) {
            bitmap.set(i);
        }

        // Clear
        bitmap.clear();

        // Verify all unset
        for (size_t i = 0; i < N; ++i) {
            EXPECT_FALSE(bitmap.test(i))
                << "Round " << round << ": bit " << i << " should be unset after clear";
        }
    }

    logger.success(" [ThreadLocalBitmap] Repeated Clear and Reuse passed.");
}

// ============================================================================
// 2. Single-Thread: VersionTagTable
// ============================================================================

TEST_F(VisitedTableTest, VersionTag_SetAndTest) {
    logger.info(" -> [VersionTagTable] Set and Test");

    const size_t N = g_config.num_elements;
    VersionTagTable table(N);

    // Initially all should be unvisited
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FALSE(table.test(i)) << "Element " << i << " should be unvisited initially";
    }

    // Set every other element
    for (size_t i = 0; i < N; i += 2) {
        table.set(i);
    }

    // Verify pattern
    for (size_t i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            EXPECT_TRUE(table.test(i)) << "Element " << i << " should be visited";
        } else {
            EXPECT_FALSE(table.test(i)) << "Element " << i << " should be unvisited";
        }
    }

    logger.success(" [VersionTagTable] Set and Test passed.");
}

TEST_F(VisitedTableTest, VersionTag_Clear) {
    logger.info(" -> [VersionTagTable] Clear (version bump)");

    const size_t N = g_config.num_elements;
    VersionTagTable table(N);

    // Set all elements
    for (size_t i = 0; i < N; ++i) {
        table.set(i);
    }

    // Clear (should just bump version, O(1))
    table.clear();

    // All should be unvisited now
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FALSE(table.test(i)) << "Element " << i << " should be unvisited after clear";
    }

    logger.success(" [VersionTagTable] Clear passed.");
}

TEST_F(VisitedTableTest, VersionTag_RepeatedClearAndReuse) {
    logger.info(" -> [VersionTagTable] Repeated Clear and Reuse");

    const size_t N = g_config.num_elements;
    VersionTagTable table(N);

    for (int round = 0; round < 100; ++round) {
        // Set some elements
        for (size_t i = round % N; i < N; i += 10) {
            table.set(i);
        }

        // Clear
        table.clear();

        // Verify all unvisited
        for (size_t i = 0; i < N; ++i) {
            EXPECT_FALSE(table.test(i))
                << "Round " << round << ": element " << i << " should be unvisited after clear";
        }
    }

    logger.success(" [VersionTagTable] Repeated Clear and Reuse passed.");
}

TEST_F(VisitedTableTest, VersionTag_CrossVersionIsolation) {
    logger.info(" -> [VersionTagTable] Cross-Version Isolation");

    const size_t N = g_config.num_elements;
    VersionTagTable table(N);

    // Version 1: set even indices
    for (size_t i = 0; i < N; i += 2) {
        table.set(i);
    }

    // Bump to version 2
    table.clear();

    // Version 2: set odd indices
    for (size_t i = 1; i < N; i += 2) {
        table.set(i);
    }

    // Even indices (set in version 1) must NOT be visible in version 2
    for (size_t i = 0; i < N; ++i) {
        if (i % 2 == 0) {
            EXPECT_FALSE(table.test(i))
                << "Element " << i << " (set in previous version) should not be visible";
        } else {
            EXPECT_TRUE(table.test(i))
                << "Element " << i << " (set in current version) should be visible";
        }
    }

    logger.success(" [VersionTagTable] Cross-Version Isolation passed.");
}

TEST_F(VisitedTableTest, VersionTag_WrapAround) {
    logger.info(" -> [VersionTagTable] Version Wrap-Around");

    const size_t N = 1024;
    VersionTagTable table(N);

    // Force wrap-around: clear 65535 times (uint16_t max)
    // After construction, version = 1. We need to reach 0 to trigger memset.
    // That means calling clear() until version wraps from 65535 -> 0 -> reset to 1.
    const uint32_t wrap_count = 65535;
    for (uint32_t i = 0; i < wrap_count; ++i) {
        // Set a few elements each round to ensure they get properly cleared
        table.set(0);
        table.set(N / 2);
        table.set(N - 1);
        table.clear();
    }

    // After wrap-around, table should still function correctly
    EXPECT_FALSE(table.test(0));
    EXPECT_FALSE(table.test(N / 2));
    EXPECT_FALSE(table.test(N - 1));

    // Set and verify
    table.set(42);
    EXPECT_TRUE(table.test(42));
    EXPECT_FALSE(table.test(43));

    // One more clear should work
    table.clear();
    EXPECT_FALSE(table.test(42));

    logger.success(" [VersionTagTable] Version Wrap-Around passed.");
}

TEST_F(VisitedTableTest, VersionTag_RandomPattern) {
    logger.info(" -> [VersionTagTable] Random Pattern");

    const size_t N = g_config.num_elements;
    VersionTagTable table(N);

    std::mt19937 rng(g_config.seed);
    std::vector<bool> expected(N, false);
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    const size_t num_sets = N / 2;
    for (size_t i = 0; i < num_sets; ++i) {
        size_t idx = dist(rng);
        table.set(idx);
        expected[idx] = true;
    }

    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(table.test(i), expected[i])
            << "Mismatch at index " << i;
    }

    logger.success(" [VersionTagTable] Random Pattern passed.");
}

// ============================================================================
// 3. Multi-Thread: VisitedTablePool with ThreadLocalBitmap
// ============================================================================

TEST_F(VisitedTableTest, Pool_Bitmap_ThreadIsolation) {
    logger.info(" -> [VisitedTablePool<ThreadLocalBitmap>] Thread Isolation");

    const vec_num_t N = g_config.num_elements;
    VisitedTablePool<router_traits_t, ThreadLocalBitmap> pool(N);
    pool.warmup();

    const int num_threads = tbb_max_num_threads();
    std::atomic<int> failures{0};

    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_threads, 1),
        [&](const tbb::blocked_range<int>& r) {
            auto& visited = pool.acquire();

            // Each thread sets a unique stripe of indices
            int tid = r.begin();
            for (vec_num_t i = static_cast<vec_num_t>(tid); i < N;
                 i += static_cast<vec_num_t>(num_threads)) {
                visited.set(i);
            }

            // Verify own stripe is set
            for (vec_num_t i = static_cast<vec_num_t>(tid); i < N;
                 i += static_cast<vec_num_t>(num_threads)) {
                if (!visited.test(i)) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }

            // Clear for next use
            visited.clear();

            // Verify all cleared
            for (vec_num_t i = 0; i < N; ++i) {
                if (visited.test(i)) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    );

    EXPECT_EQ(failures.load(), 0) << "Thread isolation or clear failures detected";

    logger.success(" [VisitedTablePool<ThreadLocalBitmap>] Thread Isolation passed.");
}

TEST_F(VisitedTableTest, Pool_Bitmap_RepeatedAcquire) {
    logger.info(" -> [VisitedTablePool<ThreadLocalBitmap>] Repeated Acquire returns same table");

    const vec_num_t N = g_config.num_elements;
    VisitedTablePool<router_traits_t, ThreadLocalBitmap> pool(N);
    pool.warmup();

    std::atomic<int> failures{0};

    tbb::parallel_for(
        tbb::blocked_range<int>(0, tbb_max_num_threads(), 1),
        [&](const tbb::blocked_range<int>&) {
            auto& table1 = pool.acquire();
            auto& table2 = pool.acquire();

            // Same thread must get the same table instance
            if (&table1 != &table2) {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    );

    EXPECT_EQ(failures.load(), 0) << "acquire() returned different tables for the same thread";

    logger.success(" [VisitedTablePool<ThreadLocalBitmap>] Repeated Acquire passed.");
}

TEST_F(VisitedTableTest, Pool_Bitmap_BatchQuerySimulation) {
    logger.info(" -> [VisitedTablePool<ThreadLocalBitmap>] Batch Query Simulation");

    const vec_num_t N = g_config.num_elements;
    const vec_num_t num_queries = 256;
    VisitedTablePool<router_traits_t, ThreadLocalBitmap> pool(N);
    pool.warmup();

    std::atomic<int> failures{0};
    std::mt19937 seed_rng(g_config.seed);
    std::vector<uint32_t> seeds(num_queries);
    for (auto& s : seeds) s = seed_rng();

    tbb::parallel_for(
        tbb::blocked_range<vec_num_t>(0, num_queries),
        [&](const tbb::blocked_range<vec_num_t>& r) {
            auto& visited = pool.acquire();
            for (vec_num_t q = r.begin(); q != r.end(); ++q) {
                visited.clear();

                // Simulate beam search: visit random subset
                std::mt19937 rng(seeds[q]);
                std::uniform_int_distribution<vec_num_t> dist(0, N - 1);
                const vec_num_t visit_count = std::min(N, vec_num_t{64});
                std::vector<vec_num_t> visited_ids;
                visited_ids.reserve(visit_count);

                for (vec_num_t i = 0; i < visit_count; ++i) {
                    vec_num_t id = dist(rng);
                    if (!visited.test(id)) {
                        visited.set(id);
                        visited_ids.push_back(id);
                    }
                }

                // Verify all visited IDs are still marked
                for (vec_num_t id : visited_ids) {
                    if (!visited.test(id)) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    );

    EXPECT_EQ(failures.load(), 0) << "Batch query simulation failures detected";

    logger.success(" [VisitedTablePool<ThreadLocalBitmap>] Batch Query Simulation passed.");
}

// ============================================================================
// 4. Multi-Thread: VisitedTablePool with VersionTagTable
// ============================================================================

TEST_F(VisitedTableTest, Pool_VersionTag_ThreadIsolation) {
    logger.info(" -> [VisitedTablePool<VersionTagTable>] Thread Isolation");

    const vec_num_t N = g_config.num_elements;
    VisitedTablePool<router_traits_t, VersionTagTable> pool(N);
    pool.warmup();

    const int num_threads = tbb_max_num_threads();
    std::atomic<int> failures{0};

    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_threads, 1),
        [&](const tbb::blocked_range<int>& r) {
            auto& visited = pool.acquire();

            int tid = r.begin();
            for (vec_num_t i = static_cast<vec_num_t>(tid); i < N;
                 i += static_cast<vec_num_t>(num_threads)) {
                visited.set(i);
            }

            for (vec_num_t i = static_cast<vec_num_t>(tid); i < N;
                 i += static_cast<vec_num_t>(num_threads)) {
                if (!visited.test(i)) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }

            // Clear (O(1) version bump)
            visited.clear();

            for (vec_num_t i = 0; i < N; ++i) {
                if (visited.test(i)) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    );

    EXPECT_EQ(failures.load(), 0) << "Thread isolation or clear failures detected";

    logger.success(" [VisitedTablePool<VersionTagTable>] Thread Isolation passed.");
}

TEST_F(VisitedTableTest, Pool_VersionTag_RepeatedAcquire) {
    logger.info(" -> [VisitedTablePool<VersionTagTable>] Repeated Acquire returns same table");

    const vec_num_t N = g_config.num_elements;
    VisitedTablePool<router_traits_t, VersionTagTable> pool(N);
    pool.warmup();

    std::atomic<int> failures{0};

    tbb::parallel_for(
        tbb::blocked_range<int>(0, tbb_max_num_threads(), 1),
        [&](const tbb::blocked_range<int>&) {
            auto& table1 = pool.acquire();
            auto& table2 = pool.acquire();

            if (&table1 != &table2) {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        }
    );

    EXPECT_EQ(failures.load(), 0) << "acquire() returned different tables for the same thread";

    logger.success(" [VisitedTablePool<VersionTagTable>] Repeated Acquire passed.");
}

TEST_F(VisitedTableTest, Pool_VersionTag_BatchQuerySimulation) {
    logger.info(" -> [VisitedTablePool<VersionTagTable>] Batch Query Simulation");

    const vec_num_t N = g_config.num_elements;
    const vec_num_t num_queries = 256;
    VisitedTablePool<router_traits_t, VersionTagTable> pool(N);
    pool.warmup();

    std::atomic<int> failures{0};
    std::mt19937 seed_rng(g_config.seed);
    std::vector<uint32_t> seeds(num_queries);
    for (auto& s : seeds) s = seed_rng();

    tbb::parallel_for(
        tbb::blocked_range<vec_num_t>(0, num_queries),
        [&](const tbb::blocked_range<vec_num_t>& r) {
            auto& visited = pool.acquire();
            for (vec_num_t q = r.begin(); q != r.end(); ++q) {
                visited.clear();

                std::mt19937 rng(seeds[q]);
                std::uniform_int_distribution<vec_num_t> dist(0, N - 1);
                const vec_num_t visit_count = std::min(N, vec_num_t{64});
                std::vector<vec_num_t> visited_ids;
                visited_ids.reserve(visit_count);

                for (vec_num_t i = 0; i < visit_count; ++i) {
                    vec_num_t id = dist(rng);
                    if (!visited.test(id)) {
                        visited.set(id);
                        visited_ids.push_back(id);
                    }
                }

                for (vec_num_t id : visited_ids) {
                    if (!visited.test(id)) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    );

    EXPECT_EQ(failures.load(), 0) << "Batch query simulation failures detected";

    logger.success(" [VisitedTablePool<VersionTagTable>] Batch Query Simulation passed.");
}

TEST_F(VisitedTableTest, Pool_VersionTag_ManyClears) {
    logger.info(" -> [VisitedTablePool<VersionTagTable>] Many Clears (version stress)");

    const vec_num_t N = 1024;
    const vec_num_t num_queries = 1000;
    VisitedTablePool<router_traits_t, VersionTagTable> pool(N);
    pool.warmup();

    std::atomic<int> failures{0};

    tbb::parallel_for(
        tbb::blocked_range<vec_num_t>(0, num_queries),
        [&](const tbb::blocked_range<vec_num_t>& r) {
            auto& visited = pool.acquire();
            for (vec_num_t q = r.begin(); q != r.end(); ++q) {
                visited.clear();

                // Set a deterministic pattern based on query id
                vec_num_t base = q % N;
                visited.set(base);
                visited.set((base + 100) % N);
                visited.set((base + 500) % N);

                if (!visited.test(base) ||
                    !visited.test((base + 100) % N) ||
                    !visited.test((base + 500) % N)) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }

                // Unrelated indices should not be visited
                vec_num_t unrelated = (base + 1) % N;
                if (unrelated != base && unrelated != (base + 100) % N &&
                    unrelated != (base + 500) % N) {
                    if (visited.test(unrelated)) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
    );

    EXPECT_EQ(failures.load(), 0) << "Many-clears stress test failures detected";

    logger.success(" [VisitedTablePool<VersionTagTable>] Many Clears passed.");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_visited_table");

    program.add_argument("-n", "--num_elements")
        .help("Number of elements for visited tables")
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

    g_config.num_elements = program.get<uint32_t>("--num_elements");
    g_config.seed = program.get<uint32_t>("--seed");

    logger.info("==========================================================");
    logger.info("      Starting VisitedTable Correctness Suite");
    logger.info(fmt::format("      Config: Elements={}, Seed={}",
                           g_config.num_elements, g_config.seed));
    logger.info("==========================================================");

    return RUN_ALL_TESTS();
}
