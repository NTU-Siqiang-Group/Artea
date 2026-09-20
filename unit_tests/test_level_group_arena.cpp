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
 * @Description: Synthetic coverage for segmented arena growth, allocation
 *               failures, concurrent access and graph compaction.
 */

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
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
#include <artea/cpu/index/compact_structure/hierarchical_graph.hpp>
#include <artea/cpu/index/compactor/hierarchical_graph_compactor.hpp>
#include <artea/cpu/index/dynamic_structure/hierarchical_graph.hpp>
#include <artea/cpu/index/dynamic_structure/level_group_arena.hpp>

// Specialize allocation only for this test's private neighbor type. Production
// allocators and traits remain unchanged, including the block-table allocator.
namespace arena_allocation_test {
struct Neighbor { std::uint64_t value = 0; };
using Block = artea::cpu::cache_aligned_container_t<Neighbor>;
inline thread_local int data_fail_after = -1;
inline thread_local int table_fail_after = -1;
inline std::atomic<std::size_t> live_data_blocks{0};
inline std::atomic<std::size_t> data_allocations{0};

void check_failure(int& remaining) {
    if (remaining == 0) throw std::bad_alloc();
    if (remaining > 0) --remaining;
}

struct FailureScope {
    ~FailureScope() { data_fail_after = table_fail_after = -1; }
};
}  // namespace arena_allocation_test

template <>
auto artea::cpu::AlignedAllocator<arena_allocation_test::Neighbor, 64>::allocate(
    std::size_t count) -> arena_allocation_test::Neighbor* {
    using namespace arena_allocation_test;
    check_failure(data_fail_after);
    auto* memory = artea::cpu::AlignedAllocator<std::byte, 64>{}.allocate(
        count * sizeof(arena_allocation_test::Neighbor));
    ++live_data_blocks;
    ++data_allocations;
    return reinterpret_cast<arena_allocation_test::Neighbor*>(memory);
}

template <>
void artea::cpu::AlignedAllocator<arena_allocation_test::Neighbor, 64>::deallocate(
    arena_allocation_test::Neighbor* memory, std::size_t count) noexcept {
    --arena_allocation_test::live_data_blocks;
    artea::cpu::AlignedAllocator<std::byte, 64>{}.deallocate(
        reinterpret_cast<std::byte*>(memory), count * sizeof(arena_allocation_test::Neighbor));
}

template <>
auto tbb::cache_aligned_allocator<arena_allocation_test::Block>::allocate(
    std::size_t count) -> arena_allocation_test::Block* {
    arena_allocation_test::check_failure(arena_allocation_test::table_fail_after);
    return reinterpret_cast<arena_allocation_test::Block*>(
        tbb::cache_aligned_allocator<std::byte>{}.allocate(count * sizeof(arena_allocation_test::Block)));
}

using namespace artea;
using namespace artea::cpu;

namespace {

// The arena is metric/dimension independent; use the production neighbor type.
using base_traits_t  = BaseTraits<uint32_t, float>;
using index_traits_t = IndexTraits<base_traits_t>;
using arena_t        = dynamic::LevelGroupArena<index_traits_t>;
using vertex_num_t   = index_traits_t::vertex_num_t;
using nbr_t          = index_traits_t::nbr_t;

static_assert(std::is_same_v<decltype(std::declval<arena_t&>().slot_ptr(0)), nbr_t*>);
static_assert(std::is_same_v<decltype(std::declval<const arena_t&>().slot_ptr(0)), const nbr_t*>);

// ============================================================================
// Helpers
// ============================================================================

void expect_valid_offsets(const arena_t& arena, const std::vector<std::size_t>& offsets) {
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

INSTANTIATE_TEST_SUITE_P(SlotWidths, LevelGroupArenaSlotWidthTest, ::testing::Values(1u, 7u, 80u, 200u));

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

INSTANTIATE_TEST_SUITE_P(ThreadCounts, LevelGroupArenaParallelTest, ::testing::Values(1, 4, 8));

// ============================================================================
// Segmented storage and arithmetic boundaries
// ============================================================================

TEST(LevelGroupArenaTest, BlockRoundingAndMemoryAccounting) {
    arena_t arena(0, 7);
    constexpr vertex_num_t block_slots_count = index_traits_t::slots_per_block;
    for (const vertex_num_t requested :
         {0u, 1u, block_slots_count - 1, block_slots_count, block_slots_count + 1,
          8 * block_slots_count + 1, 16 * block_slots_count + 1}) {
        arena.ensure_slot_capacity(requested);
        const std::size_t blocks = (requested + arena_t::slots_per_block - 1) / arena_t::slots_per_block;
        const auto usage = arena.memory_usage();
        EXPECT_EQ(arena.slot_capacity(), blocks * arena_t::slots_per_block);
        EXPECT_EQ(usage.block_count, blocks);
        EXPECT_EQ(usage.data_bytes, blocks * arena_t::slots_per_block * 7 * sizeof(nbr_t));
        EXPECT_GE(usage.block_table_capacity, blocks);
        EXPECT_EQ(usage.block_table_bytes,
                  usage.block_table_capacity * sizeof(cache_aligned_container_t<nbr_t>));
        EXPECT_EQ(arena.num_claimed_slots(), 0u);
    }
}

TEST_P(LevelGroupArenaSlotWidthTest, WholeSlotsKeepTheirAddressesAndContentsAcrossGrowth) {
    const auto width = GetParam();
    arena_t arena(2, width);
    std::vector<std::size_t> offsets;
    std::vector<nbr_t*> addresses;
    constexpr std::size_t claim_count = 2 * arena_t::slots_per_block + 3;

    for (std::size_t i = 0; i < claim_count; ++i) {
        const auto offset = arena.claim_slot();
        auto* slot = arena.slot_ptr(offset);
        for (vertex_num_t entry = 0; entry < width; ++entry) {
            slot[entry] = nbr_t(static_cast<vertex_num_t>(i), static_cast<float>(entry));
        }
        offsets.push_back(offset);
        addresses.push_back(slot);
    }
    const auto claimed = arena.num_claimed_slots();
    EXPECT_EQ(arena.memory_usage().block_count, 3u);
    // Grow past the block table's initial inline segment table as well.
    arena.ensure_slot_capacity(17 * arena_t::slots_per_block);
    EXPECT_EQ(arena.num_claimed_slots(), claimed);
    ASSERT_NO_FATAL_FAILURE(expect_valid_offsets(arena, offsets));

    const auto& view = std::as_const(arena);
    for (std::size_t i = 0; i < claim_count; ++i) {
        const auto* slot = view.slot_ptr(offsets[i]);
        ASSERT_EQ(slot, addresses[i]);
        for (vertex_num_t entry = 0; entry < width; ++entry) {
            ASSERT_EQ(slot[entry].get_vid(), i);
            ASSERT_EQ(slot[entry].get_distance(), static_cast<float>(entry));
        }
        if (i % arena_t::slots_per_block != 0) {
            EXPECT_EQ(slot, addresses[i - 1] + width);
        }
    }
}

TEST(LevelGroupArenaTest, RejectsZeroWidthAndInvalidOffsets) {
    EXPECT_THROW((arena_t(0, 0)), std::invalid_argument);
    arena_t arena(0, 7);
    EXPECT_THROW(arena.slot_ptr(0), std::out_of_range);
    arena.ensure_slot_capacity(1);
    EXPECT_THROW(arena.slot_ptr(1), std::out_of_range);
    EXPECT_THROW(arena.slot_ptr(arena.slot_capacity() * 7u), std::out_of_range);
    EXPECT_THROW(arena.slot_ptr(std::numeric_limits<std::size_t>::max()), std::out_of_range);
}

// Keep the production neighbor layout; vary only the arena's capacity type.
struct SmallCapacityTraits : index_traits_t { using vertex_num_t = std::uint16_t; };
struct WideCapacityTraits : index_traits_t { using vertex_num_t = std::size_t; };

TEST(LevelGroupArenaTest, FinalPartialBlockAndChunkReachTheTypeLimit) {
    dynamic::LevelGroupArena<SmallCapacityTraits> arena(0, 3);
    constexpr std::size_t limit = std::numeric_limits<std::uint16_t>::max();
    constexpr std::size_t block_slots_count = SmallCapacityTraits::slots_per_block;
    EXPECT_EQ(arena.slot_capacity(), 0u);
    for (std::size_t i = 0; i < limit; ++i) {
        const auto offset = arena.claim_slot();
        ASSERT_EQ(offset, i * 3);
        auto* slot = arena.slot_ptr(offset);
        slot[0] = nbr_t(static_cast<vertex_num_t>(i), 1.0f);
        slot[2] = nbr_t(static_cast<vertex_num_t>(i), 2.0f);
    }
    EXPECT_EQ(arena.slot_capacity(), limit);
    EXPECT_EQ(arena.memory_usage().block_count, (limit + block_slots_count - 1) / block_slots_count);
    EXPECT_EQ(arena.memory_usage().data_bytes, limit * 3 * sizeof(nbr_t));
    EXPECT_EQ(arena.num_claimed_slots(), limit);
    EXPECT_EQ(arena.slot_ptr((limit - 1) * 3)[2].get_vid(), limit - 1);
    EXPECT_THROW(arena.claim_slot(), std::runtime_error);
    EXPECT_EQ(arena.num_claimed_slots(), limit);
}

TEST(LevelGroupArenaTest, RejectsBlockAndTotalSizeOverflowBeforeAllocation) {
    using wide_arena_t = dynamic::LevelGroupArena<WideCapacityTraits>;
    const auto limit = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW((wide_arena_t(0, limit)), std::length_error);
    wide_arena_t arena(0, 2);
    EXPECT_THROW(arena.ensure_slot_capacity(limit), std::length_error);
    EXPECT_EQ(arena.slot_capacity(), 0u);
    EXPECT_EQ(arena.memory_usage().block_count, 0u);
    EXPECT_NO_THROW(arena.ensure_slot_capacity(1));
}

// ============================================================================
// Concurrent publication and access
// ============================================================================

TEST_P(LevelGroupArenaParallelTest, LazyClaimsGrowBeyondEmptyAndInsufficientReservations) {
    const int num_threads = GetParam();
    constexpr std::size_t claims_per_thread = 2 * arena_t::slots_per_block + 3;
    for (const std::size_t prefix : {std::size_t{0}, arena_t::slots_per_block - arena_t::slot_chunk_size}) {
        SCOPED_TRACE(fmt::format("threads={}, prefix={}", num_threads, prefix));
        arena_t arena(1, 7);
        arena.ensure_slot_capacity(prefix);
        std::vector<std::size_t> offsets(prefix + num_threads * claims_per_thread);
        for (std::size_t i = 0; i < prefix; ++i) offsets[i] = arena.claim_slot();

        // Real threads and a common start exercise first growth and the last
        // available batch without relying on how TBB schedules small tasks.
        std::barrier start(num_threads);
        std::vector<std::jthread> workers;
        for (int worker = 0; worker < num_threads; ++worker) {
            workers.emplace_back([&, worker] {
                start.arrive_and_wait();
                for (std::size_t i = 0; i < claims_per_thread; ++i) {
                    const auto index = prefix + worker * claims_per_thread + i;
                    offsets[index] = arena.claim_slot();
                    std::fill_n(arena.slot_ptr(offsets[index]), 7,
                                nbr_t(static_cast<vertex_num_t>(index), 42.0f));
                }
            });
        }
        for (auto& worker : workers) worker.join();

        ASSERT_NO_FATAL_FAILURE(expect_valid_offsets(arena, offsets));
        const auto claimed = arena.num_claimed_slots();
        EXPECT_LE(claimed - offsets.size(), num_threads * (arena_t::slot_chunk_size - 1));
        EXPECT_LT(arena.slot_capacity() - claimed, arena_t::slots_per_block);
        for (std::size_t i = prefix; i < offsets.size(); ++i) {
            const auto* slot = arena.slot_ptr(offsets[i]);
            for (std::size_t entry = 0; entry < 7; ++entry) {
                ASSERT_EQ(slot[entry].get_vid(), i);
                ASSERT_EQ(slot[entry].get_distance(), 42.0f);
            }
        }
    }
}

TEST(LevelGroupArenaTest, PartlyConsumedThreadLocalChunksStayReservedAcrossGrowth) {
    arena_t arena(0, 1);
    std::barrier sync(2);
    std::vector<std::size_t> worker_offsets(arena_t::slot_chunk_size);
    std::jthread worker([&] {
        worker_offsets[0] = arena.claim_slot();
        sync.arrive_and_wait();
        sync.arrive_and_wait();
        for (std::size_t i = 1; i < worker_offsets.size(); ++i) worker_offsets[i] = arena.claim_slot();
    });
    sync.arrive_and_wait();
    std::vector<std::size_t> offsets;
    for (std::size_t i = 0; i < 2 * arena_t::slots_per_block; ++i) offsets.push_back(arena.claim_slot());
    sync.arrive_and_wait();
    worker.join();

    for (std::size_t i = 0; i < worker_offsets.size(); ++i) EXPECT_EQ(worker_offsets[i], i);
    offsets.insert(offsets.end(), worker_offsets.begin(), worker_offsets.end());
    ASSERT_NO_FATAL_FAILURE(expect_valid_offsets(arena, offsets));
    EXPECT_EQ(arena.num_claimed_slots(), offsets.size());
}

TEST_P(LevelGroupArenaParallelTest, ConcurrentGrowthDoesNotDuplicateBlocksOrSlots) {
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, GetParam());
    tbb::task_arena workers(GetParam());
    arena_t arena(1, 7);
    constexpr std::size_t requests = 128;
    std::vector<std::size_t> offsets(requests);

    for (const bool same_capacity : {true, false}) {
        workers.execute([&] {
            tbb::parallel_for(tbb::blocked_range<std::size_t>(0, requests, 1),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    for (auto i = range.begin(); i != range.end(); ++i) {
                        const auto blocks = same_capacity ? 1u : 1u + i % 17u;
                        arena.ensure_slot_capacity(blocks * arena_t::slots_per_block);
                        offsets[i] = arena.claim_slot();
                        std::fill_n(arena.slot_ptr(offsets[i]), 7,
                                    nbr_t(static_cast<vertex_num_t>(i), 42.0f));
                    }
                });
        });
        EXPECT_EQ(arena.memory_usage().block_count, same_capacity ? 1u : 17u);
        ASSERT_NO_FATAL_FAILURE(expect_valid_offsets(arena, offsets));
        for (std::size_t i = 0; i < requests; ++i) {
            const auto* slot = std::as_const(arena).slot_ptr(offsets[i]);
            for (std::size_t entry = 0; entry < 7; ++entry) {
                ASSERT_EQ(slot[entry].get_vid(), i);
                ASSERT_EQ(slot[entry].get_distance(), 42.0f);
            }
        }
    }
}

TEST(LevelGroupArenaTest, ReadersKeepOldPointersWhileBlocksArePublished) {
    arena_t arena(0, 7);
    const auto offset = arena.claim_slot();
    auto* held = arena.slot_ptr(offset);
    std::fill_n(held, 7, nbr_t(23, 5.0f));
    std::atomic<bool> reader_ready{false};
    std::atomic<bool> done{false};
    std::exception_ptr failure;

    // An explicit thread avoids depending on TBB scheduling two tasks at once.
    std::jthread grower([&] {
        reader_ready.wait(false);
        try {
            for (std::size_t i = arena_t::slot_chunk_size; i < 65 * arena_t::slots_per_block; ++i) {
                arena.claim_slot();
            }
        } catch (...) {
            failure = std::current_exception();
        }
        done.store(true, std::memory_order_release);
    });
    reader_ready.store(true);
    reader_ready.notify_one();
    do {
        const auto& view = std::as_const(arena);
        EXPECT_EQ(view.slot_ptr(offset), held);
        EXPECT_EQ(held[0].get_vid(), 23u);
        EXPECT_EQ(held[6].get_distance(), 5.0f);
        // Acquiring capacity must make even the last complete slot accessible.
        const std::size_t last = (view.slot_capacity() - 1) * 7u;
        EXPECT_TRUE(view.slot_ptr(last)[6].is_invalid());
    } while (!done.load(std::memory_order_acquire));
    grower.join();
    EXPECT_EQ(failure, nullptr);
    EXPECT_EQ(arena.slot_capacity(), 65 * arena_t::slots_per_block);
}

// ============================================================================
// Fail allocations locally, including a real concurrent_vector tail failure
// ============================================================================

struct FaultTraits : index_traits_t {
    using nbr_t = arena_allocation_test::Neighbor;
};
struct TinyBlockFaultTraits : FaultTraits { static constexpr std::size_t slots_per_block = 3; };
using fault_arena_t = dynamic::LevelGroupArena<FaultTraits>;

TEST(LevelGroupArenaTest, ConcurrentFirstClaimsAllocateOneDataBlock) {
    using namespace arena_allocation_test;
    ASSERT_EQ(live_data_blocks, 0u);
    const auto allocations = data_allocations.load();
    {
        fault_arena_t arena(0, 1);
        std::barrier start(8);
        std::vector<std::jthread> workers;
        for (std::size_t worker = 0; worker < 8; ++worker) {
            workers.emplace_back([&] {
                start.arrive_and_wait();
                arena.slot_ptr(arena.claim_slot())->value = 123;
            });
        }
        for (auto& worker : workers) worker.join();
        EXPECT_EQ(data_allocations.load() - allocations, 1u);
        EXPECT_EQ(live_data_blocks, 1u);
        EXPECT_EQ(arena.memory_usage().block_count, 1u);
        EXPECT_EQ(arena.num_claimed_slots(), 8 * fault_arena_t::slot_chunk_size);
    }
    EXPECT_EQ(live_data_blocks, 0u);
}

TEST(LevelGroupArenaTest, FailedClaimPreservesCountersAndCanRetryDataAllocation) {
    using namespace arena_allocation_test;
    ASSERT_EQ(live_data_blocks, 0u);
    for (const bool has_prefix : {false, true}) {
        FailureScope cleanup;
        fault_arena_t arena(0, 1);
        const auto prefix = has_prefix ? fault_arena_t::slots_per_block : 0;
        for (std::size_t i = 0; i < prefix; ++i) ASSERT_EQ(arena.claim_slot(), i);
        data_fail_after = 0;
        for (int attempt = 0; attempt < 2; ++attempt) {
            EXPECT_THROW(arena.claim_slot(), std::bad_alloc);
            EXPECT_EQ(arena.num_claimed_slots(), prefix);
            EXPECT_EQ(arena.slot_capacity(), prefix);
        }
        data_fail_after = -1;
        EXPECT_EQ(arena.claim_slot(), prefix);
        data_fail_after = 0;
        // Retry filled a valid TLS batch; cached claims need no allocations.
        for (std::size_t i = 1; i < fault_arena_t::slot_chunk_size; ++i) {
            EXPECT_EQ(arena.claim_slot(), prefix + i);
        }
        EXPECT_EQ(arena.num_claimed_slots(), prefix + fault_arena_t::slot_chunk_size);
    }
    EXPECT_EQ(live_data_blocks, 0u);
}

TEST(LevelGroupArenaTest, FailedClaimKeepsPartiallyPublishedBlocksForRetry) {
    using namespace arena_allocation_test;
    FailureScope cleanup;
    {
        dynamic::LevelGroupArena<TinyBlockFaultTraits> arena(0, 1);
        data_fail_after = 1; // An 8-slot batch needs three blocks; fail on the second.
        EXPECT_THROW(arena.claim_slot(), std::bad_alloc);
        EXPECT_EQ(arena.slot_capacity(), 3u);
        EXPECT_EQ(arena.num_claimed_slots(), 0u);
        auto* held = arena.slot_ptr(0);
        held->value = 789;
        EXPECT_EQ(live_data_blocks, 1u);

        data_fail_after = -1;
        EXPECT_EQ(arena.claim_slot(), 0u);
        EXPECT_EQ(arena.slot_capacity(), 9u);
        EXPECT_EQ(arena.num_claimed_slots(), 8u);
        EXPECT_EQ(arena.slot_ptr(0), held);
        EXPECT_EQ(held->value, 789u);
        EXPECT_EQ(live_data_blocks, 3u);
    }
    EXPECT_EQ(live_data_blocks, 0u);
}

TEST(LevelGroupArenaTest, DataAllocationFailureKeepsPublishedPrefixAndAllowsRetry) {
    using namespace arena_allocation_test;
    FailureScope cleanup;
    ASSERT_EQ(live_data_blocks, 0u);
    {
        fault_arena_t arena(0, 1);
        arena.ensure_slot_capacity(1);
        const auto offset = arena.claim_slot();
        auto* held = arena.slot_ptr(offset);
        held->value = 123;
        const auto claimed = arena.num_claimed_slots();

        data_fail_after = 1;  // Publish one more block, then fail locally.
        EXPECT_THROW(arena.ensure_slot_capacity(3 * fault_arena_t::slots_per_block), std::bad_alloc);
        EXPECT_EQ(arena.slot_capacity(), 2 * fault_arena_t::slots_per_block);
        EXPECT_EQ(arena.num_claimed_slots(), claimed);
        EXPECT_EQ(arena.slot_ptr(offset), held);
        EXPECT_EQ(held->value, 123u);
        EXPECT_EQ(live_data_blocks, 2u);
        EXPECT_THROW(arena.slot_ptr(arena.slot_capacity()), std::out_of_range);

        data_fail_after = -1;
        EXPECT_NO_THROW(arena.ensure_slot_capacity(3 * fault_arena_t::slots_per_block));
        EXPECT_EQ(arena.slot_ptr(offset), held);
        EXPECT_EQ(held->value, 123u);
        EXPECT_EQ(live_data_blocks, 3u);
    }
    EXPECT_EQ(live_data_blocks, 0u);
}

TEST(LevelGroupArenaTest, BlockTableFailureRejectsFurtherGrowthAndReclaimsStorage) {
    using namespace arena_allocation_test;
    FailureScope cleanup;
    ASSERT_EQ(live_data_blocks, 0u);
    for (const bool has_prefix : {false, true}) {
        {
            fault_arena_t arena(0, 1);
            arena_allocation_test::Neighbor* held = nullptr;
            std::size_t offset = 0;
            if (has_prefix) {
                arena.ensure_slot_capacity(1);
                // Fill the existing table allocation so the next append allocates.
                arena.ensure_slot_capacity(arena.memory_usage().block_table_capacity
                                           * fault_arena_t::slots_per_block);
                offset = arena.claim_slot();
                held = arena.slot_ptr(offset);
                held->value = 456;
            }
            const auto capacity = arena.slot_capacity();
            for (std::size_t i = has_prefix ? 1 : 0; i < capacity; ++i) {
                ASSERT_EQ(arena.claim_slot(), i);
            }
            const auto claimed = arena.num_claimed_slots();
            const auto blocks = live_data_blocks.load();
            table_fail_after = 0;
            EXPECT_THROW(arena.claim_slot(), std::bad_alloc);
            EXPECT_EQ(arena.slot_capacity(), capacity);
            EXPECT_EQ(arena.num_claimed_slots(), claimed);
            EXPECT_EQ(live_data_blocks, blocks); // Unpublished local block was freed.
            EXPECT_THROW(arena.slot_ptr(capacity), std::out_of_range);

            table_fail_after = -1;
            const auto allocations = data_allocations.load();
            EXPECT_THROW(arena.ensure_slot_capacity(capacity + 1), std::bad_alloc);
            EXPECT_THROW(arena.claim_slot(), std::bad_alloc);
            EXPECT_EQ(arena.num_claimed_slots(), claimed);
            EXPECT_EQ(data_allocations, allocations); // Reject before any retry allocation.
            EXPECT_NO_THROW(arena.ensure_slot_capacity(capacity));
            if (has_prefix) {
                EXPECT_EQ(arena.slot_ptr(offset), held);
                EXPECT_EQ(held->value, 456u);
            }
        }
        EXPECT_EQ(live_data_blocks, 0u);
    }
}

TEST(LevelGroupArenaTest, GraphAllocatesNeighborStorageOnlyForAssignedGroups) {
    using graph_t = dynamic::HierarchicalGraph<index_traits_t>;
    constexpr vertex_num_t expected_vertices = 100'000;
    graph_t graph(/*max_level=*/12, /*upper_neighbors=*/40, /*bottom_neighbors=*/80, expected_vertices);
    EXPECT_EQ(graph.arena_memory_usage().data_bytes, 0u);
    EXPECT_EQ(graph.arena_memory_usage().block_count, 0u);
    EXPECT_EQ(graph.arena_memory_usage().block_table_bytes, 0u);
    graph.add_vertices(expected_vertices);
    EXPECT_EQ(graph.arena_memory_usage().data_bytes, 0u);

    graph.assign_layer(0, 0);
    graph.assign_layer(1, 3);
    EXPECT_EQ(graph.arena_memory_usage().block_count, 2u);
    constexpr std::size_t block_slots_count = index_traits_t::slots_per_block;
    EXPECT_EQ(graph.arena_memory_usage().data_bytes, block_slots_count * (80 + 200) * sizeof(nbr_t));
    for (vertex_num_t vid = 2; vid <= block_slots_count + 1; ++vid) graph.assign_layer(vid, 0);
    const auto usage = graph.arena_memory_usage();
    EXPECT_EQ(usage.block_count, 3u);
    EXPECT_EQ(usage.data_bytes, block_slots_count * (2 * 80 + 200) * sizeof(nbr_t));
    EXPECT_GE(usage.block_table_capacity, usage.block_count);
    EXPECT_EQ(usage.block_table_bytes, usage.block_table_capacity * sizeof(cache_aligned_container_t<nbr_t>));
    for (vertex_num_t level = 0; level <= 12; ++level) {
        const auto blocks = level == 0 ? 2u : (level == 3 ? 1u : 0u);
        EXPECT_EQ(graph.get_arena_capacity_in_arena(level), blocks * block_slots_count);
    }
}

TEST(LevelGroupArenaTest, SegmentedGraphCompactsWithoutChangingLayerNeighbors) {
    using graph_t = dynamic::HierarchicalGraph<index_traits_t>;
    tbb::global_control control(tbb::global_control::max_allowed_parallelism, 4);
    constexpr vertex_num_t vertices = 2 * index_traits_t::slots_per_block + 257;
    // The constructor's size hint must neither preallocate nor limit growth.
    graph_t graph(/*max_level=*/2, /*upper_neighbors=*/3, /*bottom_neighbors=*/5, /*total_vertices=*/1);
    graph.add_vertices(vertices);
    index_traits_t::vector_array_t vectors(vertices, 1);
    for (vertex_num_t vid = 0; vid < vertices; ++vid) {
        // Both retained level groups cross a block boundary. The sparse L2
        // group is demoted by compact, exercising its capacity-based offsets.
        const vertex_num_t highest = vid == vertices - 1 ? 2 : vid % 2;
        graph.assign_layer(vid, highest);
        vectors.get(vid)[0] = static_cast<float>(vid);
        for (vertex_num_t level = 0; level <= highest; ++level) {
            auto neighbors = graph.fetch_layer_nbrs(vid, level);
            ASSERT_EQ(neighbors.size(), level == 0 ? 5u : 3u);
            for (const auto& neighbor : neighbors) ASSERT_TRUE(neighbor.is_invalid());
            neighbors[0] = nbr_t((vid + level + 1) % vertices, 1.0f);
            const auto const_neighbors = std::as_const(graph).fetch_layer_nbrs(vid, level);
            ASSERT_EQ(const_neighbors.data(), neighbors.data());
        }
    }
    const auto compact = HierarchicalGraphCompactor<index_traits_t>::compact_graph(
        graph, vectors, [](const float* left, const float* right) {
            const float delta = left[0] - right[0];
            return delta * delta;
        });
    EXPECT_EQ(compact.get_num_vertices(), vertices);
    EXPECT_EQ(compact.top_occupied_level_id(), 1u);
    EXPECT_LT(compact.entry_point_vid(), vertices);
    for (vertex_num_t vid = 0; vid < vertices; ++vid) {
        const auto highest = std::min<vertex_num_t>(graph.get_highest_level_id(vid), 1);
        EXPECT_EQ(compact.get_highest_level_id(vid), highest);
        for (vertex_num_t level = 0; level <= highest; ++level) {
            const auto neighbors = compact.fetch_layer_nbrs(vid, level);
            ASSERT_EQ(neighbors.size(), level == 0 ? 5u : 3u);
            EXPECT_EQ(neighbors[0], (vid + level + 1) % vertices);
            for (std::size_t entry = 1; entry < neighbors.size(); ++entry) {
                EXPECT_EQ(neighbors[entry], index_traits_t::invalid_vertex_id);
            }
        }
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
