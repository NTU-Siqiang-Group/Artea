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
 * @FilePath: /Artea/unit_tests/test_level_group_arena.cpp
 * @Description: Synthetic GoogleTest coverage for LevelGroupArena metadata,
 *               explicit reservation, slot offsets and concurrent claims.
 *               No dataset or contiguous-storage accessor is required.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <artea/cpu/framework/type_traits/base_traits.hpp>
#include <artea/cpu/framework/type_traits/index_traits.hpp>
#include <artea/cpu/index/dynamic_structure/level_group_arena.hpp>

using namespace artea;
using namespace artea::cpu;

namespace {

// The arena is metric/dimension independent; use the production neighbor type.
using base_traits_t  = BaseTraits<uint32_t, float>;
using index_traits_t = IndexTraits<base_traits_t>;
using arena_t        = dynamic::LevelGroupArena<index_traits_t>;
using vertex_num_t   = index_traits_t::vertex_num_t;

// ============================================================================
// Helpers
// ============================================================================

void expect_valid_offsets(const arena_t& arena,
                          const std::vector<std::size_t>& offsets) {
    const std::size_t width    = arena.slot_nbrs_count();
    const std::size_t capacity = arena.slot_capacity();
    const std::size_t claimed  = arena.num_claimed_slots();

    ASSERT_GT(width, 0u);
    EXPECT_GE(claimed, offsets.size());
    EXPECT_LE(claimed, capacity);
    for (const auto offset : offsets) {
        ASSERT_EQ(offset % width, 0u) << "Unaligned offset: " << offset;
        ASSERT_LT(offset / width, claimed) << "Offset was not reserved";
        ASSERT_LT(offset / width, capacity) << "Slot exceeds storage capacity";
    }

    auto sorted = offsets;
    std::sort(sorted.begin(), sorted.end());
    EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
        << "The same slot was handed out more than once";
}

// ============================================================================
// Construction and explicit reservation
// ============================================================================

TEST(LevelGroupArenaTest, ConstructionMetadata) {
    const arena_t arena(/*highest_level_id=*/3, /*slot_nbrs_count=*/200);

    EXPECT_EQ(arena.highest_level_id(), 3u);
    EXPECT_EQ(arena.slot_nbrs_count(), 200u);
    EXPECT_EQ(arena.slot_capacity(), 0u);
    EXPECT_EQ(arena.num_claimed_slots(), 0u);
}

TEST(LevelGroupArenaTest, ReserveBeforeClaimsCoversRequestedCapacity) {
    arena_t arena(/*highest_level_id=*/0, /*slot_nbrs_count=*/80);

    for (const vertex_num_t requested : {0u, 1u, 7u, 8u, 9u, 65u, 513u}) {
        SCOPED_TRACE(fmt::format("requested={}", requested));
        const auto previous_capacity = arena.slot_capacity();
        arena.ensure_slot_capacity(requested);

        // Segmented storage may round up; exact capacity is not an API promise.
        EXPECT_GE(arena.slot_capacity(), requested);
        EXPECT_GE(arena.slot_capacity(), previous_capacity);
        EXPECT_EQ(arena.num_claimed_slots(), 0u);
    }
}

TEST(LevelGroupArenaTest, RepeatedSmallerReservationPreservesClaims) {
    arena_t arena(/*highest_level_id=*/1, /*slot_nbrs_count=*/120);
    arena.ensure_slot_capacity(32);
    const auto capacity = arena.slot_capacity();
    const auto first = arena.claim_slot();
    const auto claimed = arena.num_claimed_slots();

    for (const vertex_num_t requested : {0u, 8u, 32u}) {
        arena.ensure_slot_capacity(requested);
        EXPECT_GE(arena.slot_capacity(), capacity);
        EXPECT_EQ(arena.num_claimed_slots(), claimed);
    }

    const auto second = arena.claim_slot();
    EXPECT_EQ(second, first + arena.slot_nbrs_count());
    ASSERT_NO_FATAL_FAILURE(expect_valid_offsets(arena, {first, second}));
}

// ============================================================================
// Single-thread claims and thread-local reservations
// ============================================================================

class LevelGroupArenaSlotWidthTest : public ::testing::TestWithParam<vertex_num_t> {};

TEST_P(LevelGroupArenaSlotWidthTest, ClaimsUseNeighborOffsetsAcrossChunks) {
    const vertex_num_t width = GetParam();
    constexpr vertex_num_t claim_count = 257;
    arena_t arena(/*highest_level_id=*/2, width);
    arena.ensure_slot_capacity(claim_count + arena_t::slot_chunk_size);

    std::vector<std::size_t> offsets;
    offsets.reserve(claim_count);
    for (std::size_t i = 0; i < claim_count; ++i) {
        offsets.push_back(arena.claim_slot());
        ASSERT_EQ(offsets.back(), i * width)
            << "Offsets must count neighbors, not bytes or slots";
    }

    ASSERT_NO_FATAL_FAILURE(expect_valid_offsets(arena, offsets));
    EXPECT_LT(arena.num_claimed_slots() - offsets.size(), arena_t::slot_chunk_size);
}

INSTANTIATE_TEST_SUITE_P(
    SlotWidths, LevelGroupArenaSlotWidthTest, ::testing::Values(1u, 7u, 80u, 200u));

TEST(LevelGroupArenaTest, ClaimedCountIncludesUnusedThreadLocalSlots) {
    arena_t arena(/*highest_level_id=*/0, /*slot_nbrs_count=*/7);
    const auto chunk_size = arena_t::slot_chunk_size;
    arena.ensure_slot_capacity(3 * chunk_size);
    std::vector<std::size_t> offsets;

    offsets.push_back(arena.claim_slot());
    const auto first_reservation = arena.num_claimed_slots();
    EXPECT_EQ(first_reservation, chunk_size);
    EXPECT_GT(first_reservation, offsets.size());

    for (vertex_num_t i = 1; i < chunk_size; ++i) {
        offsets.push_back(arena.claim_slot());
        EXPECT_EQ(arena.num_claimed_slots(), first_reservation)
            << "Consuming a cached slot must not reserve another batch";
    }

    offsets.push_back(arena.claim_slot());
    EXPECT_EQ(arena.num_claimed_slots(), 2 * chunk_size);
    EXPECT_GT(arena.num_claimed_slots(), offsets.size());
    ASSERT_NO_FATAL_FAILURE(expect_valid_offsets(arena, offsets));
}

TEST(LevelGroupArenaTest, SeparateArenasKeepIndependentReservations) {
    arena_t lower(/*highest_level_id=*/0, /*slot_nbrs_count=*/80);
    arena_t upper(/*highest_level_id=*/3, /*slot_nbrs_count=*/200);
    lower.ensure_slot_capacity(64);
    upper.ensure_slot_capacity(64);
    std::vector<std::size_t> lower_offsets;
    std::vector<std::size_t> upper_offsets;

    for (std::size_t i = 0; i < 17; ++i) {
        const auto upper_claimed = upper.num_claimed_slots();
        lower_offsets.push_back(lower.claim_slot());
        EXPECT_EQ(upper.num_claimed_slots(), upper_claimed);
        if (i < 3) {
            const auto lower_claimed = lower.num_claimed_slots();
            upper_offsets.push_back(upper.claim_slot());
            EXPECT_EQ(lower.num_claimed_slots(), lower_claimed);
        }
    }

    EXPECT_EQ(lower_offsets.front(), 0u);
    EXPECT_EQ(upper_offsets.front(), 0u);
    ASSERT_NO_FATAL_FAILURE(expect_valid_offsets(lower, lower_offsets));
    ASSERT_NO_FATAL_FAILURE(expect_valid_offsets(upper, upper_offsets));
    EXPECT_GT(lower.num_claimed_slots(), upper.num_claimed_slots());
    EXPECT_EQ(lower.highest_level_id(), 0u);
    EXPECT_EQ(upper.highest_level_id(), 3u);
}

// ============================================================================
// Concurrent claims: validate after joining, without assuming scheduler order
// ============================================================================

class LevelGroupArenaParallelTest : public ::testing::TestWithParam<int> {};

TEST_P(LevelGroupArenaParallelTest, ClaimsAreUniqueAndWithinReservedCapacity) {
    const int num_threads = GetParam();
    tbb::global_control control(
        tbb::global_control::max_allowed_parallelism, num_threads);
    tbb::task_arena workers(num_threads);

    constexpr vertex_num_t claim_count = 4099;
    arena_t arena(/*highest_level_id=*/2, /*slot_nbrs_count=*/7);
    // Enough even if every claim came from a different thread; under 2 MiB.
    // Capacity must not depend on which workers TBB happens to schedule.
    arena.ensure_slot_capacity(claim_count * arena_t::slot_chunk_size);
    std::vector<std::size_t> offsets(claim_count);
    tbb::enumerable_thread_specific<std::size_t> claims_per_thread(std::size_t{0});

    workers.execute([&] {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, claim_count, 1),
            [&](const tbb::blocked_range<std::size_t>& range) {
                auto& local_count = claims_per_thread.local();
                for (std::size_t i = range.begin(); i != range.end(); ++i) {
                    offsets[i] = arena.claim_slot();
                    ++local_count;
                }
            });
    });

    std::size_t delivered = 0;
    for (const auto count : claims_per_thread) {
        EXPECT_GT(count, 0u);
        delivered += count;
    }
    ASSERT_EQ(delivered, offsets.size());
    ASSERT_NO_FATAL_FAILURE(expect_valid_offsets(arena, offsets));
    EXPECT_EQ(arena.num_claimed_slots() % arena_t::slot_chunk_size, 0u);
    EXPECT_LE(arena.num_claimed_slots() - delivered,
              claims_per_thread.size() * (arena_t::slot_chunk_size - 1))
        << "Each participating thread can retain at most one partial batch";

    RecordProperty("thread_limit", num_threads);
    RecordProperty("workers_observed", static_cast<int>(claims_per_thread.size()));
    ARTEA_INFO(fmt::format(
        "[LevelGroupArena] thread_limit={}, workers={}, delivered={}, reserved={}",
        num_threads, claims_per_thread.size(), delivered, arena.num_claimed_slots()));
}

INSTANTIATE_TEST_SUITE_P(
    ThreadCounts, LevelGroupArenaParallelTest, ::testing::Values(1, 4, 8));

}  // namespace

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
