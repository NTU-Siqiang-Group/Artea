// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <artea/cpu/index/practheo/vertex_weight_assigner.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
using namespace artea::cpu;
constexpr auto metric = DistanceMetricsT::EUCLIDEAN;
constexpr vec_dim_t dimension = 16;
using index_t = practheo::index_t<metric, dimension>;
using config_t = practheo::build_config_t<metric, dimension>;
using assigner_t = practheo::vertex_weight_assigner_t<metric, dimension>;
using factory_t = practheo::factory_t<metric, dimension>;
static_assert(std::is_same_v<assigner_t, practheo::VertexWeightAssigner<graph_factory_traits_t<metric, dimension>>>);

TEST(VertexWeightAssigner, AssignsWeightsBeyondOneByte) {
    constexpr vec_num_t count = 4096;
    const vector_array_t base(count, dimension);
    const config_t config(count, 0.25f, 0.99f);
    ASSERT_EQ(config.max_weight, 828u);
    index_t index(base, sl_config_t(0), config);
    assigner_t::assign_vertex_weight(index, base, 0);

    const auto& groups = index.get_vids_by_weight();
    ASSERT_EQ(groups.size(), 829u);
    EXPECT_TRUE(std::any_of(groups.begin() + 256, groups.end(), [](const auto& group) { return !group.empty(); }));
    std::vector<vertex_id_t> ids;
    for (const auto& group : groups) ids.insert(ids.end(), group.begin(), group.end());
    std::sort(ids.begin(), ids.end());
    ASSERT_EQ(ids.size(), count);
    for (vertex_id_t i = 0; i < count; ++i) EXPECT_EQ(ids[i], i);
}

TEST(VertexWeightAssigner, GroupsEveryVertexOnceWithCappedGeometricProbabilities) {
    constexpr vec_num_t count = 32768;
    constexpr ratio_t r = 0.75f;
    const vector_array_t base(count, dimension);
    index_t index(base, sl_config_t(0), config_t(count, 0.25f, r));
    assigner_t::assign_vertex_weight(index, base, 0);

    const auto& groups = index.get_vids_by_weight();
    ASSERT_EQ(groups.size(), 38u);
    std::vector<bool> seen(count, false);
    for (std::size_t j = 0; j < groups.size(); ++j) {
        const double tail = std::pow(static_cast<double>(r), j);
        const double probability = j + 1 == groups.size() ? tail : tail * (1 - r);
        const double stddev = std::sqrt(count * probability * (1 - probability));
        EXPECT_NEAR(groups[j].size(), count * probability, 6 * stddev) << "weight " << j;
        EXPECT_TRUE(std::is_sorted(groups[j].begin(), groups[j].end()));
        for (const auto vid : groups[j]) {
            ASSERT_LT(vid, count);
            EXPECT_FALSE(seen[vid]);
            seen[vid] = true;
        }
    }
    EXPECT_EQ(std::count(seen.begin(), seen.end(), true), count);
}

TEST(VertexWeightAssigner, MatchesAcrossBatchBoundariesAndThreadCounts) {
    constexpr vec_num_t count = 8193;
    const vector_array_t base(count, dimension);
    const config_t config(count, 0.25f, 0.75f);
    index_t whole(base, sl_config_t(0), config);
    {
        tbb::global_control one_thread(tbb::global_control::max_allowed_parallelism, 1);
        assigner_t::assign_vertex_weight(whole, base, 0);
    }

    index_t batched(base, sl_config_t(0), config);
    vec_num_t begin = 0;
    for (vec_num_t end : {1u, 1023u, 1024u, 1573u, count}) {
        assigner_t::assign_vertex_weight(batched, base.extract_subset(begin, end - begin), begin);
        begin = end;
    }
    EXPECT_EQ(batched.get_vids_by_weight(), whole.get_vids_by_weight());
}

TEST(VertexWeightAssigner, EmptyAndOverlappingBatchesPreserveAssignedVertices) {
    const vector_array_t base(6, dimension);
    const config_t config(6, 0.25f);
    index_t index(base, sl_config_t(0), config);
    assigner_t::assign_vertex_weight(index, base.extract_subset(0, 3), 0);
    const auto before = index.get_vids_by_weight();

    assigner_t::assign_vertex_weight(index, vector_array_t{}, base.get_num_vecs());
    EXPECT_EQ(index.get_vids_by_weight(), before);
    EXPECT_THROW(assigner_t::assign_vertex_weight(index, base.extract_subset(2, 2), 2), std::invalid_argument);
    EXPECT_EQ(index.get_vids_by_weight(), before);

    assigner_t::assign_vertex_weight(index, base.extract_subset(3, 3), 3);
    index_t whole(base, sl_config_t(0), config);
    assigner_t::assign_vertex_weight(whole, base, 0);
    EXPECT_EQ(index.get_vids_by_weight(), whole.get_vids_by_weight());
}

TEST(VertexWeightAssigner, UsesBatchCountAndFirstVidIndependentlyOfBaseVectors) {
    const vector_array_t base(0, dimension);
    const vector_array_t batch(3, 1);
    index_t index(base, sl_config_t(0), config_t(12, 0.25f));
    factory_t::assign_vertex_weight(index, batch, 4);
    factory_t::assign_vertex_weight(index, batch, 9);
    std::vector<vertex_id_t> ids;
    for (const auto& group : index.get_vids_by_weight()) ids.insert(ids.end(), group.begin(), group.end());
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ(ids, (std::vector<vertex_id_t>{4, 5, 6, 9, 10, 11}));
}
}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
