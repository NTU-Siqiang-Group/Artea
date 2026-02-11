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
 * @FilePath: /Artea/tests/test_candidate_queue.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: GoogleTest suite for StdCandidateQueue, LinearCandidateQueue,
 *               and FHCandidateQueue correctness verification.
 */

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <gtest/gtest.h>
#include <argparse/argparse.hpp>
#include <fmt/format.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/router/std_candidate_queue.hpp>
#include <artea/cpu/router/linear_candidate_queue.hpp>
#include <artea/cpu/router/fh_candidate_queue.hpp>

using namespace artea;
using namespace artea::cpu;

// --- Type Definitions ---
using vec_num_t = uint32_t;
using vec_ele_t = float;
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using index_traits_t = IndexTraits<base_traits_t>;
using router_traits_t = RouterTraits<computer_traits_t, index_traits_t, false>;
using candidate_entry_t = typename router_traits_t::candidate_entry_t;
using vertex_id_t = typename router_traits_t::vertex_id_t;
using distance_t = typename router_traits_t::distance_t;
using container_t = cache_aligned_container_t<candidate_entry_t>;

// Queue types under test
using std_queue_t = StdCandidateQueue<router_traits_t>;
using linear_queue_t = LinearCandidateQueue<router_traits_t>;
using fh_queue_t = FHCandidateQueue<router_traits_t>;

// --- Global Configuration ---
struct TestConfig {
    uint32_t scale;   // Scale factor for test sizes
    uint32_t seed;    // Random seed for reproducibility
} g_config;

// --- Helper: create an unexplored entry ---
static candidate_entry_t make_entry(vertex_id_t id, distance_t dist) {
    return candidate_entry_t::make_entry(id, dist);
}

// --- Helper: drain all unexplored entries from a queue ---
template <typename QueueT>
static std::vector<distance_t> drain_unexplored(QueueT& q) {
    constexpr distance_t max_dist = router_traits_t::max_distance;
    constexpr vertex_id_t invalid_id = router_traits_t::invalid_vertex_id;
    std::vector<distance_t> result;
    while (true) {
        auto [id, dist] = q.pop_best_unexplored();
        if (dist == max_dist && id == invalid_id) {
            break;
        }
        result.push_back(dist);
    }
    return result;
}

// ============================================================================
// Test Fixture
// ============================================================================

class CandidateQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        logger.info("----------------------------------------------------------");
    }
    void TearDown() override {
        logger.info("----------------------------------------------------------");
    }
};

// ============================================================================
// 1. Initialize Function Tests
// ============================================================================

TEST_F(CandidateQueueTest, Initialize_StdQueue) {
    logger.info(" -> [StdQueue] Initialize Function");

    const std::size_t K = 32 * g_config.scale;
    std_queue_t q(K);

    // Create initial candidates with shuffled distances
    std::vector<candidate_entry_t> init_candidates;
    init_candidates.reserve(K);

    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        init_candidates.push_back(make_entry(static_cast<vertex_id_t>(i), distances[i]));
    }

    // Initialize queue
    q.initialize(init_candidates);

    // Verify queue size
    EXPECT_EQ(q.get_result_size(), K);
    EXPECT_FALSE(q.empty());

    // Verify candidates are sorted by draining
    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), K);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "StdQueue initialize: order violated at position " << i;
    }

    logger.success(" [StdQueue] Initialize Function passed.");
}

TEST_F(CandidateQueueTest, Initialize_LinearQueue) {
    logger.info(" -> [LinearQueue] Initialize Function");

    const std::size_t K = 32 * g_config.scale;
    linear_queue_t q(K);

    // Create initial candidates with shuffled distances
    using stateful_entry_t = typename router_traits_t::stateful_candidate_entry_t;
    cache_aligned_container_t<stateful_entry_t> init_candidates;
    init_candidates.reserve(K);

    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        init_candidates.emplace_back(static_cast<vertex_id_t>(i), distances[i]);
    }

    // Initialize queue
    q.initialize(init_candidates);

    // Verify queue size
    EXPECT_EQ(q.get_result_size(), K);
    EXPECT_FALSE(q.empty());

    // Verify candidates are sorted
    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), K);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "LinearQueue initialize: order violated at position " << i;
    }

    logger.success(" [LinearQueue] Initialize Function passed.");
}

TEST_F(CandidateQueueTest, Initialize_FHQueue) {
    logger.info(" -> [FHQueue] Initialize Function");

    const std::size_t K = 32 * g_config.scale;
    fh_queue_t q(K);

    // Create initial candidates with shuffled distances
    std::vector<candidate_entry_t> init_candidates;
    init_candidates.reserve(K);

    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        init_candidates.push_back(make_entry(static_cast<vertex_id_t>(i), distances[i]));
    }

    // Initialize queue
    q.initialize(init_candidates);

    // Verify queue size
    EXPECT_EQ(q.get_result_size(), K);
    EXPECT_FALSE(q.empty());

    // Verify candidates are sorted
    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), K);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "FHQueue initialize: order violated at position " << i;
    }

    logger.success(" [FHQueue] Initialize Function passed.");
}

// ============================================================================
// 2. Random Initialize Function Tests
// ============================================================================

TEST_F(CandidateQueueTest, RandomInitialize_StdQueue) {
    logger.info(" -> [StdQueue] Random Initialize Function");

    const std::size_t K = 32 * g_config.scale;
    const std::size_t num_vecs = 1000;
    const std::size_t dim = 128;

    // Create test data
    using vector_array_t = typename router_traits_t::vector_array_t;
    using dist_func_t = typename router_traits_t::dist_func_t;
    using random_seq_t = typename router_traits_t::random_seq_t;

    vector_array_t base_vecs(num_vecs, dim);
    std::mt19937 rng(g_config.seed);
    std::uniform_real_distribution<vec_ele_t> dist(0.0f, 1.0f);

    // Fill with random data
    for (std::size_t i = 0; i < num_vecs; ++i) {
        vec_ele_t* vec = base_vecs.get(i);
        for (std::size_t j = 0; j < dim; ++j) {
            vec[j] = dist(rng);
        }
    }

    // Create query vector
    std::vector<vec_ele_t> query_vec(dim);
    for (std::size_t j = 0; j < dim; ++j) {
        query_vec[j] = dist(rng);
    }

    // Create distance function and random sequence
    dist_func_t dist_func(dim);
    random_seq_t random_seq(num_vecs);

    // Initialize queue
    std_queue_t q(K);
    q.random_initialize(random_seq, dist_func, query_vec.data(), base_vecs);

    // Verify queue size
    EXPECT_EQ(q.get_result_size(), K);
    EXPECT_FALSE(q.empty());

    // Verify candidates are sorted
    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), K);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "StdQueue random_initialize: order violated at position " << i;
    }

    logger.success(" [StdQueue] Random Initialize Function passed.");
}

TEST_F(CandidateQueueTest, RandomInitialize_LinearQueue) {
    logger.info(" -> [LinearQueue] Random Initialize Function");

    const std::size_t K = 32 * g_config.scale;
    const std::size_t num_vecs = 1000;
    const std::size_t dim = 128;

    // Create test data
    using vector_array_t = typename router_traits_t::vector_array_t;
    using dist_func_t = typename router_traits_t::dist_func_t;
    using random_seq_t = typename router_traits_t::random_seq_t;

    vector_array_t base_vecs(num_vecs, dim);
    std::mt19937 rng(g_config.seed);
    std::uniform_real_distribution<vec_ele_t> dist(0.0f, 1.0f);

    for (std::size_t i = 0; i < num_vecs; ++i) {
        vec_ele_t* vec = base_vecs.get(i);
        for (std::size_t j = 0; j < dim; ++j) {
            vec[j] = dist(rng);
        }
    }

    std::vector<vec_ele_t> query_vec(dim);
    for (std::size_t j = 0; j < dim; ++j) {
        query_vec[j] = dist(rng);
    }

    dist_func_t dist_func(dim);
    random_seq_t random_seq(num_vecs);

    linear_queue_t q(K);
    q.random_initialize(random_seq, dist_func, query_vec.data(), base_vecs);

    EXPECT_EQ(q.get_result_size(), K);
    EXPECT_FALSE(q.empty());

    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), K);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "LinearQueue random_initialize: order violated at position " << i;
    }

    logger.success(" [LinearQueue] Random Initialize Function passed.");
}

TEST_F(CandidateQueueTest, RandomInitialize_FHQueue) {
    logger.info(" -> [FHQueue] Random Initialize Function");

    const std::size_t K = 32 * g_config.scale;
    const std::size_t num_vecs = 1000;
    const std::size_t dim = 128;

    // Create test data
    using vector_array_t = typename router_traits_t::vector_array_t;
    using dist_func_t = typename router_traits_t::dist_func_t;
    using random_seq_t = typename router_traits_t::random_seq_t;

    vector_array_t base_vecs(num_vecs, dim);
    std::mt19937 rng(g_config.seed);
    std::uniform_real_distribution<vec_ele_t> dist(0.0f, 1.0f);

    for (std::size_t i = 0; i < num_vecs; ++i) {
        vec_ele_t* vec = base_vecs.get(i);
        for (std::size_t j = 0; j < dim; ++j) {
            vec[j] = dist(rng);
        }
    }

    std::vector<vec_ele_t> query_vec(dim);
    for (std::size_t j = 0; j < dim; ++j) {
        query_vec[j] = dist(rng);
    }

    dist_func_t dist_func(dim);
    random_seq_t random_seq(num_vecs);

    fh_queue_t q(K);
    q.random_initialize(random_seq, dist_func, query_vec.data(), base_vecs);

    EXPECT_EQ(q.get_result_size(), K);
    EXPECT_FALSE(q.empty());

    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), K);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "FHQueue random_initialize: order violated at position " << i;
    }

    logger.success(" [FHQueue] Random Initialize Function passed.");
}

// ============================================================================
// 3. Basic Sort: unordered in, ordered out
// ============================================================================

TEST_F(CandidateQueueTest, BasicSort_StdQueue) {
    logger.info(" -> [StdQueue] Basic Sort: unordered in, ordered out");

    const std::size_t N = 64 * g_config.scale;
    std_queue_t q(N);

    // Generate shuffled distances [1.0, 2.0, ..., N]
    std::vector<distance_t> distances(N);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < N; ++i) {
        q.try_push(static_cast<vertex_id_t>(i), distances[i]);
    }

    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), N);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "StdQueue order violated at position " << i;
    }

    // Final call should return invalid
    auto [sentinel_id, sentinel_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(sentinel_dist, router_traits_t::max_distance);

    logger.success(" [StdQueue] Basic Sort passed.");
}

TEST_F(CandidateQueueTest, BasicSort_LinearQueue) {
    logger.info(" -> [LinearQueue] Basic Sort: unordered in, ordered out");

    const std::size_t N = 64 * g_config.scale;
    linear_queue_t q(N);

    std::vector<distance_t> distances(N);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < N; ++i) {
        q.try_push(static_cast<vertex_id_t>(i), distances[i]);
    }

    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), N);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "LinearQueue order violated at position " << i;
    }

    auto [sentinel_id, sentinel_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(sentinel_dist, router_traits_t::max_distance);

    logger.success(" [LinearQueue] Basic Sort passed.");
}

TEST_F(CandidateQueueTest, BasicSort_FHQueue) {
    logger.info(" -> [FHQueue] Basic Sort: unordered in, ordered out");

    const std::size_t N = 64 * g_config.scale;
    fh_queue_t q(N);

    std::vector<distance_t> distances(N);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < N; ++i) {
        q.try_push(static_cast<vertex_id_t>(i), distances[i]);
    }

    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), N);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "FHQueue order violated at position " << i;
    }

    auto [sentinel_id, sentinel_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(sentinel_dist, router_traits_t::max_distance);

    logger.success(" [FHQueue] Basic Sort passed.");
}

// ============================================================================
// 2. Capacity & Eviction
// ============================================================================

TEST_F(CandidateQueueTest, CapacityEviction_StdQueue) {
    logger.info(" -> [StdQueue] Capacity & Eviction");

    const std::size_t K = 32 * g_config.scale;
    std_queue_t q(K);

    // Fill with [10, 20, 30, ..., K*10]
    for (std::size_t i = 1; i <= K; ++i) {
        EXPECT_TRUE(q.try_push(static_cast<vertex_id_t>(i),
                                           static_cast<distance_t>(i * 10)));
    }
    EXPECT_EQ(q.get_result_size(), K);

    // Insert worse element -> should be rejected
    bool accepted_worse = q.try_push(999, static_cast<distance_t>(K * 10 + 10));
    EXPECT_FALSE(accepted_worse);
    EXPECT_EQ(q.get_result_size(), K);

    // Insert better element (between first and second) -> should be accepted
    distance_t better_dist = 15.0f;
    bool accepted_better = q.try_push(998, better_dist);
    EXPECT_TRUE(accepted_better);
    EXPECT_EQ(q.get_result_size(), K);  // still K, worst was evicted

    logger.success(" [StdQueue] Capacity & Eviction passed.");
}

TEST_F(CandidateQueueTest, CapacityEviction_LinearQueue) {
    logger.info(" -> [LinearQueue] Capacity & Eviction");

    const std::size_t K = 32 * g_config.scale;
    linear_queue_t q(K);

    for (std::size_t i = 1; i <= K; ++i) {
        EXPECT_TRUE(q.try_push(static_cast<vertex_id_t>(i),
                                           static_cast<distance_t>(i * 10)));
    }
    EXPECT_EQ(q.get_result_size(), K);

    // Insert worse element -> should be rejected
    bool accepted_worse = q.try_push(999, static_cast<distance_t>(K * 10 + 10));
    EXPECT_FALSE(accepted_worse);
    EXPECT_EQ(q.get_result_size(), K);

    // Insert better element -> should be accepted, worst evicted
    distance_t better_dist = 15.0f;
    bool accepted_better = q.try_push(998, better_dist);
    EXPECT_TRUE(accepted_better);
    EXPECT_EQ(q.get_result_size(), K);

    // Verify the queue contains the correct best-K elements
    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), K);
    // 15.0 should be present, K*10 should NOT be present
    EXPECT_FLOAT_EQ(out[1], 15.0f);  // second smallest (after 10.0)

    logger.success(" [LinearQueue] Capacity & Eviction passed.");
}

TEST_F(CandidateQueueTest, CapacityEviction_FHQueue) {
    logger.info(" -> [FHQueue] Capacity & Eviction");

    const std::size_t K = 32 * g_config.scale;
    fh_queue_t q(K);

    for (std::size_t i = 1; i <= K; ++i) {
        EXPECT_TRUE(q.try_push(static_cast<vertex_id_t>(i),
                                           static_cast<distance_t>(i * 10)));
    }
    EXPECT_EQ(q.get_result_size(), K);

    bool accepted_worse = q.try_push(999, static_cast<distance_t>(K * 10 + 10));
    EXPECT_FALSE(accepted_worse);
    EXPECT_EQ(q.get_result_size(), K);

    distance_t better_dist = 15.0f;
    bool accepted_better = q.try_push(998, better_dist);
    EXPECT_TRUE(accepted_better);
    EXPECT_EQ(q.get_result_size(), K);

    logger.success(" [FHQueue] Capacity & Eviction passed.");
}

// ============================================================================
// 3. Cursor Regression (Lazy Deletion correctness)
//    After exploring some entries, inserting a better entry must be returned
//    next by pop_best_unexplored(), not skipped.
// ============================================================================

TEST_F(CandidateQueueTest, CursorRegression_StdQueue) {
    logger.info(" -> [StdQueue] Cursor Regression");

    const std::size_t K = 32 * g_config.scale;
    std_queue_t q(K);

    // Insert [10, 20, 30, ..., K*10]
    for (std::size_t i = 1; i <= K; ++i) {
        q.try_push(static_cast<vertex_id_t>(i),
                               static_cast<distance_t>(i * 10));
    }

    // Explore the first entry (distance = 10)
    auto [first_id, first_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(first_dist, 10.0f);

    // Insert a new entry with distance 15 (better than next unexplored = 20)
    bool ok = q.try_push(900, 15.0f);
    EXPECT_TRUE(ok);

    // The next unexplored MUST be 15, not 20
    auto [second_id, second_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(second_dist, 15.0f)
        << "Cursor regression bug: expected 15.0 but got " << second_dist;

    auto [third_id, third_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(third_dist, 20.0f);

    logger.success(" [StdQueue] Cursor Regression passed.");
}

TEST_F(CandidateQueueTest, CursorRegression_LinearQueue) {
    logger.info(" -> [LinearQueue] Cursor Regression (core test)");

    const std::size_t K = 32 * g_config.scale;
    linear_queue_t q(K);

    for (std::size_t i = 1; i <= K; ++i) {
        q.try_push(static_cast<vertex_id_t>(i),
                               static_cast<distance_t>(i * 10));
    }

    // Explore the first entry (distance = 10)
    auto [first_id, first_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(first_dist, 10.0f);

    // Insert 15 — better than next unexplored (20), cursor must regress
    bool ok = q.try_push(900, 15.0f);
    EXPECT_TRUE(ok);

    auto [second_id, second_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(second_dist, 15.0f)
        << "Cursor regression bug: expected 15.0 but got " << second_dist;

    auto [third_id, third_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(third_dist, 20.0f);

    logger.success(" [LinearQueue] Cursor Regression passed.");
}

TEST_F(CandidateQueueTest, CursorRegression_FHQueue) {
    logger.info(" -> [FHQueue] Cursor Regression");

    const std::size_t K = 32 * g_config.scale;
    fh_queue_t q(K);

    for (std::size_t i = 1; i <= K; ++i) {
        q.try_push(static_cast<vertex_id_t>(i),
                               static_cast<distance_t>(i * 10));
    }

    auto [first_id, first_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(first_dist, 10.0f);

    bool ok = q.try_push(900, 15.0f);
    EXPECT_TRUE(ok);

    auto [second_id, second_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(second_dist, 15.0f)
        << "Cursor regression bug: expected 15.0 but got " << second_dist;

    auto [third_id, third_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(third_dist, 20.0f);

    logger.success(" [FHQueue] Cursor Regression passed.");
}

// Deep cursor regression: explore many entries, then insert something early
TEST_F(CandidateQueueTest, CursorRegressionDeep_LinearQueue) {
    logger.info(" -> [LinearQueue] Deep Cursor Regression");

    const std::size_t K = 64 * g_config.scale;
    linear_queue_t q(K);

    for (std::size_t i = 1; i <= K; ++i) {
        q.try_push(static_cast<vertex_id_t>(i),
                               static_cast<distance_t>(i * 10));
    }

    // Explore the first half
    const std::size_t half = K / 2;
    for (std::size_t i = 0; i < half; ++i) {
        auto [id, dist] = q.pop_best_unexplored();
        EXPECT_FLOAT_EQ(dist, static_cast<distance_t>((i + 1) * 10));
    }

    // Now cursor points to entry with distance (half+1)*10
    // Insert something with distance 5.0 — way before the cursor
    bool ok = q.try_push(9999, 5.0f);
    EXPECT_TRUE(ok);

    // Must get 5.0 next, not (half+1)*10
    auto [next_id, next_dist] = q.pop_best_unexplored();
    EXPECT_FLOAT_EQ(next_dist, 5.0f)
        << "Deep cursor regression bug: expected 5.0 but got " << next_dist;

    logger.success(" [LinearQueue] Deep Cursor Regression passed.");
}

// ============================================================================
// 4. Threshold Logic: fast rejection for equal and worse distances
// ============================================================================

TEST_F(CandidateQueueTest, ThresholdLogic_StdQueue) {
    logger.info(" -> [StdQueue] Threshold Logic");

    const std::size_t K = 32 * g_config.scale;
    std_queue_t q(K);

    // Fill with [10, 20, ..., K*10]. Worst = K*10.
    for (std::size_t i = 1; i <= K; ++i) {
        q.try_push(static_cast<vertex_id_t>(i),
                               static_cast<distance_t>(i * 10));
    }

    distance_t worst = static_cast<distance_t>(K * 10);

    // Equal to worst -> rejected
    EXPECT_FALSE(q.try_push(800, worst));
    // Worse than worst -> rejected
    EXPECT_FALSE(q.try_push(801, worst + 1.0f));
    // Slightly worse -> rejected
    EXPECT_FALSE(q.try_push(802, worst + 100.0f));

    // Verify queue unchanged
    EXPECT_EQ(q.get_result_size(), K);

    logger.success(" [StdQueue] Threshold Logic passed.");
}

TEST_F(CandidateQueueTest, ThresholdLogic_LinearQueue) {
    logger.info(" -> [LinearQueue] Threshold Logic");

    const std::size_t K = 32 * g_config.scale;
    linear_queue_t q(K);

    for (std::size_t i = 1; i <= K; ++i) {
        q.try_push(static_cast<vertex_id_t>(i),
                               static_cast<distance_t>(i * 10));
    }

    distance_t worst = static_cast<distance_t>(K * 10);

    EXPECT_FALSE(q.try_push(800, worst));
    EXPECT_FALSE(q.try_push(801, worst + 1.0f));
    EXPECT_FALSE(q.try_push(802, worst + 100.0f));

    EXPECT_EQ(q.get_result_size(), K);

    logger.success(" [LinearQueue] Threshold Logic passed.");
}

TEST_F(CandidateQueueTest, ThresholdLogic_FHQueue) {
    logger.info(" -> [FHQueue] Threshold Logic");

    const std::size_t K = 32 * g_config.scale;
    fh_queue_t q(K);

    for (std::size_t i = 1; i <= K; ++i) {
        q.try_push(static_cast<vertex_id_t>(i),
                               static_cast<distance_t>(i * 10));
    }

    distance_t worst = static_cast<distance_t>(K * 10);

    EXPECT_FALSE(q.try_push(800, worst));
    EXPECT_FALSE(q.try_push(801, worst + 1.0f));
    EXPECT_FALSE(q.try_push(802, worst + 100.0f));

    EXPECT_EQ(q.get_result_size(), K);

    logger.success(" [FHQueue] Threshold Logic passed.");
}

// ============================================================================
// 5. Extract Results/IDs: verify output order and correctness
// ============================================================================

TEST_F(CandidateQueueTest, ExtractResults_StdQueue) {
    logger.info(" -> [StdQueue] Extract Results Order");

    const std::size_t K = 32 * g_config.scale;
    std_queue_t q(K);

    // Insert shuffled distances [1.0, 2.0, ..., K]
    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        q.try_push(static_cast<vertex_id_t>(i), distances[i]);
    }

    // Extract results
    auto results = q.extract_results(K);
    ASSERT_EQ(results.size(), K);

    // Verify ascending order by distance
    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i - 1].second, results[i].second)
            << "StdQueue extract_results: order violated at position " << i;
    }

    // Verify distances are [1.0, 2.0, ..., K]
    for (std::size_t i = 0; i < K; ++i) {
        EXPECT_FLOAT_EQ(results[i].second, static_cast<distance_t>(i + 1))
            << "StdQueue extract_results: incorrect distance at position " << i;
    }

    logger.success(" [StdQueue] Extract Results Order passed.");
}

TEST_F(CandidateQueueTest, ExtractResults_LinearQueue) {
    logger.info(" -> [LinearQueue] Extract Results Order");

    const std::size_t K = 32 * g_config.scale;
    linear_queue_t q(K);

    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        q.try_push(static_cast<vertex_id_t>(i), distances[i]);
    }

    auto results = q.extract_results(K);
    ASSERT_EQ(results.size(), K);

    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i - 1].second, results[i].second)
            << "LinearQueue extract_results: order violated at position " << i;
    }

    for (std::size_t i = 0; i < K; ++i) {
        EXPECT_FLOAT_EQ(results[i].second, static_cast<distance_t>(i + 1))
            << "LinearQueue extract_results: incorrect distance at position " << i;
    }

    logger.success(" [LinearQueue] Extract Results Order passed.");
}

TEST_F(CandidateQueueTest, ExtractResults_FHQueue) {
    logger.info(" -> [FHQueue] Extract Results Order");

    const std::size_t K = 32 * g_config.scale;
    fh_queue_t q(K);

    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        q.try_push(static_cast<vertex_id_t>(i), distances[i]);
    }

    auto results = q.extract_results(K);
    ASSERT_EQ(results.size(), K);

    for (std::size_t i = 1; i < results.size(); ++i) {
        EXPECT_LE(results[i - 1].second, results[i].second)
            << "FHQueue extract_results: order violated at position " << i;
    }

    for (std::size_t i = 0; i < K; ++i) {
        EXPECT_FLOAT_EQ(results[i].second, static_cast<distance_t>(i + 1))
            << "FHQueue extract_results: incorrect distance at position " << i;
    }

    logger.success(" [FHQueue] Extract Results Order passed.");
}

TEST_F(CandidateQueueTest, ExtractResultIds_StdQueue) {
    logger.info(" -> [StdQueue] Extract Result IDs Order");

    const std::size_t K = 32 * g_config.scale;
    std_queue_t q(K);

    // Insert with known ID-distance mapping: ID=i, distance=i+1
    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        // Use distance as ID for easy verification
        vertex_id_t id = static_cast<vertex_id_t>(distances[i]);
        q.try_push(id, distances[i]);
    }

    auto result_ids = q.extract_result_ids(K);
    ASSERT_EQ(result_ids.size(), K);

    // Verify IDs are in ascending order (since ID == distance)
    for (std::size_t i = 1; i < result_ids.size(); ++i) {
        EXPECT_LE(result_ids[i - 1], result_ids[i])
            << "StdQueue extract_result_ids: order violated at position " << i;
    }

    logger.success(" [StdQueue] Extract Result IDs Order passed.");
}

TEST_F(CandidateQueueTest, ExtractResultIds_LinearQueue) {
    logger.info(" -> [LinearQueue] Extract Result IDs Order");

    const std::size_t K = 32 * g_config.scale;
    linear_queue_t q(K);

    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        vertex_id_t id = static_cast<vertex_id_t>(distances[i]);
        q.try_push(id, distances[i]);
    }

    auto result_ids = q.extract_result_ids(K);
    ASSERT_EQ(result_ids.size(), K);

    for (std::size_t i = 1; i < result_ids.size(); ++i) {
        EXPECT_LE(result_ids[i - 1], result_ids[i])
            << "LinearQueue extract_result_ids: order violated at position " << i;
    }

    logger.success(" [LinearQueue] Extract Result IDs Order passed.");
}

TEST_F(CandidateQueueTest, ExtractResultIds_FHQueue) {
    logger.info(" -> [FHQueue] Extract Result IDs Order");

    const std::size_t K = 32 * g_config.scale;
    fh_queue_t q(K);

    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        vertex_id_t id = static_cast<vertex_id_t>(distances[i]);
        q.try_push(id, distances[i]);
    }

    auto result_ids = q.extract_result_ids(K);
    ASSERT_EQ(result_ids.size(), K);

    for (std::size_t i = 1; i < result_ids.size(); ++i) {
        EXPECT_LE(result_ids[i - 1], result_ids[i])
            << "FHQueue extract_result_ids: order violated at position " << i;
    }

    logger.success(" [FHQueue] Extract Result IDs Order passed.");
}

// Test consistency between extract_results and extract_result_ids
TEST_F(CandidateQueueTest, ExtractConsistency_StdQueue) {
    logger.info(" -> [StdQueue] Extract Results/IDs Consistency");

    const std::size_t K = 32 * g_config.scale;

    // Create two identical queues
    std_queue_t q1(K), q2(K);

    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        q1.try_push(static_cast<vertex_id_t>(i), distances[i]);
        q2.try_push(static_cast<vertex_id_t>(i), distances[i]);
    }

    auto results = q1.extract_results(K);
    auto result_ids = q2.extract_result_ids(K);

    ASSERT_EQ(results.size(), result_ids.size());

    // Verify IDs match
    for (std::size_t i = 0; i < K; ++i) {
        EXPECT_EQ(results[i].first, result_ids[i])
            << "StdQueue: ID mismatch at position " << i;
    }

    logger.success(" [StdQueue] Extract Results/IDs Consistency passed.");
}

TEST_F(CandidateQueueTest, ExtractConsistency_LinearQueue) {
    logger.info(" -> [LinearQueue] Extract Results/IDs Consistency");

    const std::size_t K = 32 * g_config.scale;

    linear_queue_t q1(K), q2(K);

    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        q1.try_push(static_cast<vertex_id_t>(i), distances[i]);
        q2.try_push(static_cast<vertex_id_t>(i), distances[i]);
    }

    auto results = q1.extract_results(K);
    auto result_ids = q2.extract_result_ids(K);

    ASSERT_EQ(results.size(), result_ids.size());

    for (std::size_t i = 0; i < K; ++i) {
        EXPECT_EQ(results[i].first, result_ids[i])
            << "LinearQueue: ID mismatch at position " << i;
    }

    logger.success(" [LinearQueue] Extract Results/IDs Consistency passed.");
}

TEST_F(CandidateQueueTest, ExtractConsistency_FHQueue) {
    logger.info(" -> [FHQueue] Extract Results/IDs Consistency");

    const std::size_t K = 32 * g_config.scale;

    fh_queue_t q1(K), q2(K);

    std::vector<distance_t> distances(K);
    std::iota(distances.begin(), distances.end(), 1.0f);
    std::mt19937 rng(g_config.seed);
    std::shuffle(distances.begin(), distances.end(), rng);

    for (std::size_t i = 0; i < K; ++i) {
        q1.try_push(static_cast<vertex_id_t>(i), distances[i]);
        q2.try_push(static_cast<vertex_id_t>(i), distances[i]);
    }

    auto results = q1.extract_results(K);
    auto result_ids = q2.extract_result_ids(K);

    ASSERT_EQ(results.size(), result_ids.size());

    for (std::size_t i = 0; i < K; ++i) {
        EXPECT_EQ(results[i].first, result_ids[i])
            << "FHQueue: ID mismatch at position " << i;
    }

    logger.success(" [FHQueue] Extract Results/IDs Consistency passed.");
}

// ============================================================================
// 6. Large-scale randomized stress test
//    Push N random entries into capacity-K queue, then drain and verify order.
// ============================================================================

TEST_F(CandidateQueueTest, StressTest_StdQueue) {
    logger.info(fmt::format(" -> [StdQueue] Stress Test (scale={})", g_config.scale));

    const std::size_t K = 64 * g_config.scale;
    const std::size_t N = 256 * g_config.scale;
    std_queue_t q(K);

    std::mt19937 rng(g_config.seed);
    std::uniform_real_distribution<distance_t> dist(0.0f, 10000.0f);

    for (std::size_t i = 0; i < N; ++i) {
        q.try_push(static_cast<vertex_id_t>(i), dist(rng));
    }

    auto out = drain_unexplored(q);
    EXPECT_GT(out.size(), 0u);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "StdQueue stress: order violated at " << i;
    }

    logger.success(" [StdQueue] Stress Test passed.");
}

TEST_F(CandidateQueueTest, StressTest_LinearQueue) {
    logger.info(fmt::format(" -> [LinearQueue] Stress Test (scale={})", g_config.scale));

    const std::size_t K = 64 * g_config.scale;
    const std::size_t N = 256 * g_config.scale;
    linear_queue_t q(K);

    std::mt19937 rng(g_config.seed);
    std::uniform_real_distribution<distance_t> dist(0.0f, 10000.0f);

    for (std::size_t i = 0; i < N; ++i) {
        q.try_push(static_cast<vertex_id_t>(i), dist(rng));
    }

    auto out = drain_unexplored(q);
    ASSERT_EQ(out.size(), K);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "LinearQueue stress: order violated at " << i;
    }

    logger.success(" [LinearQueue] Stress Test passed.");
}

TEST_F(CandidateQueueTest, StressTest_FHQueue) {
    logger.info(fmt::format(" -> [FHQueue] Stress Test (scale={})", g_config.scale));

    const std::size_t K = 64 * g_config.scale;
    const std::size_t N = 256 * g_config.scale;
    fh_queue_t q(K);

    std::mt19937 rng(g_config.seed);
    std::uniform_real_distribution<distance_t> dist(0.0f, 10000.0f);

    for (std::size_t i = 0; i < N; ++i) {
        q.try_push(static_cast<vertex_id_t>(i), dist(rng));
    }

    auto out = drain_unexplored(q);
    EXPECT_GT(out.size(), 0u);
    for (std::size_t i = 1; i < out.size(); ++i) {
        EXPECT_LE(out[i - 1], out[i])
            << "FHQueue stress: order violated at " << i;
    }

    logger.success(" [FHQueue] Stress Test passed.");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);

    argparse::ArgumentParser program("test_candidate_queue");

    program.add_argument("-k", "--scale")
        .help("Scale factor for test sizes (default 1)")
        .scan<'u', uint32_t>()
        .default_value(uint32_t{100});

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

    g_config.scale = program.get<uint32_t>("--scale");
    g_config.seed = program.get<uint32_t>("--seed");

    logger.info("==========================================================");
    logger.info("      Starting CandidateQueue Correctness Suite");
    logger.info(fmt::format("      Config: Scale={}, Seed={}",
                           g_config.scale, g_config.seed));
    logger.info("==========================================================");

    return RUN_ALL_TESTS();
}
