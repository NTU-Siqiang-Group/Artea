// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <artea/cpu/framework/type_context/default_context.hpp>
#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace {
using namespace artea::cpu;
constexpr auto metric = DistanceMetricsT::EUCLIDEAN;
constexpr vec_dim_t dimension = 16;
using factory_t = practheo::factory_t<metric, dimension>;
using config_t = practheo::build_config_t<metric, dimension>;

auto line_vectors(vertex_num_t count) -> vector_array_t {
    vector_array_t base(count, dimension);
    for (vertex_id_t i = 0; i < count; ++i) {
        std::fill_n(base.get(i), dimension, 0.0f);
        // The small quadratic term breaks ties between left and right neighbors.
        base.get(i)[0] = static_cast<float>(i) + i * i / 65536.0f;
    }
    return base;
}

auto nearest_on_line(const vector_array_t& base, vertex_id_t source,
                     const std::vector<vertex_id_t>& ids, vertex_num_t k) -> nbr_arr_t {
    nbr_arr_t nearest;
    for (const auto vid : ids) {
        if (vid != source) nearest.emplace_back(vid, std::abs(base.get(vid)[0] - base.get(source)[0]));
    }
    std::sort(nearest.begin(), nearest.end(), strict_nbr_comp_t{});
    if (nearest.size() > k) nearest.resize(k);
    return nearest;
}

TEST(PractheoConstruct, UsesGlobalTopKAndTopKFromEveryGroup) {
    constexpr vertex_num_t count = 24;
    constexpr vertex_num_t k = 2;
    const auto base = line_vectors(count);
    std::vector<vertex_id_t> all(count);
    std::iota(all.begin(), all.end(), vertex_id_t{0});
    for (auto [global_rate, group_rate] : {std::pair{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}, {1.f, 1.f}}) {
        config_t config(count, global_rate, 0.5f, k);
        config.sampling_rate1 = group_rate;
        auto result = factory_t::construct_graph(base, sl_config_t(1), config);
        const auto& groups = result.graph.get_vids_by_weight();
        ASSERT_GT(std::count_if(groups.begin(), groups.end(), [](const auto& g) { return !g.empty(); }), 1);

        vertex_num_t max_degree = 0;
        for (vertex_id_t vid = 0; vid < count; ++vid) {
            SCOPED_TRACE(::testing::Message() << "vid=" << vid << " p0=" << global_rate << " p1=" << group_rate);
            std::vector<bool> selected(count);
            if (global_rate == 1) {
                for (const auto& hit : nearest_on_line(base, vid, all, k)) selected[hit.get_vid()] = true;
            }
            if (group_rate == 1) {
                for (const auto& group : groups) {
                    for (const auto& hit : nearest_on_line(base, vid, group, k)) selected[hit.get_vid()] = true;
                }
            }
            auto expected = nearest_on_line(base, vid, all, count);
            std::erase_if(expected, [&](const auto& hit) { return !selected[hit.get_vid()]; });
            EXPECT_EQ(result.graph.fetch_nbrs(vid), expected);
            max_degree = std::max(max_degree, static_cast<vertex_num_t>(expected.size()));
        }
        EXPECT_EQ(result.graph.layer_config().max_nbr_size(), std::max(vertex_num_t{1}, max_degree));
        if (group_rate == 1) EXPECT_GT(max_degree, k);
    }
}

TEST(PractheoConstruct, SamplesOverlappingCandidatesIndependentlyAndReproducibly) {
    constexpr vertex_num_t count = 128;
    constexpr vertex_num_t k = count - 1;
    const auto base = line_vectors(count);
    const auto build = [&](ratio_t global_rate, ratio_t group_rate) {
        // Taking every other VID makes the global and pooled candidates identical.
        config_t config(count, global_rate, 0.25f, k);
        config.sampling_rate1 = group_rate;
        return factory_t::construct_graph<true>(base, sl_config_t(0), config);
    };
    const auto global = build(0.5f, 0.0f);
    const auto grouped = build(0.0f, 0.5f);
    const auto combined = build(0.5f, 0.5f);

    std::size_t retained = 0;
    for (vertex_id_t vid = 0; vid < count; ++vid) {
        const auto& a = global.graph.fetch_nbrs(vid);
        const auto& b = grouped.graph.fetch_nbrs(vid);
        nbr_arr_t expected;
        std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(expected), strict_nbr_comp_t{});
        EXPECT_EQ(combined.graph.fetch_nbrs(vid), expected);
        retained += expected.size();
    }
    constexpr double probability = 0.75;  // 1 - (1 - 0.5)^2
    EXPECT_NEAR(retained, count * k * probability, 6 * std::sqrt(count * k * probability * (1 - probability)));
    tbb::global_control one_thread(tbb::global_control::max_allowed_parallelism, 1);
    EXPECT_EQ(build(0.5f, 0.5f).graph.get_nbrs_arr(), combined.graph.get_nbrs_arr());
}

TEST(PractheoConstruct, HandlesSmallPoolsZeroKAndCoincidentVectors) {
    for (vertex_num_t count : {0u, 1u, 2u, 5u}) {
        vector_array_t base(count, dimension);
        std::fill_n(base.get_all(), static_cast<std::size_t>(count) * dimension, 0.0f);
        for (vertex_num_t k : {0u, 1u, std::numeric_limits<vertex_num_t>::max()}) {
            // N=2 gives a default sampling_rate1 above one and must retain every group candidate.
            config_t config(std::max(2u, count), 0.0f, 0.25f, k);
            if (count > 2) config.sampling_rate1 = 1;
            auto result = factory_t::construct_graph(base, sl_config_t(0), config);
            for (vertex_id_t vid = 0; vid < count; ++vid) {
                const auto& row = result.graph.fetch_nbrs(vid);
                std::size_t expected_degree = 0;
                for (const auto& group : result.graph.get_vids_by_weight()) {
                    const auto others = group.size() - std::binary_search(group.begin(), group.end(), vid);
                    expected_degree += std::min<std::size_t>(k, others);
                }
                EXPECT_EQ(row.size(), expected_degree);
                std::vector<bool> seen(count);
                for (const auto& nbr : row) {
                    ASSERT_LT(nbr.get_vid(), count);
                    EXPECT_NE(nbr.get_vid(), vid);
                    EXPECT_FALSE(seen[nbr.get_vid()]);
                    seen[nbr.get_vid()] = true;
                    EXPECT_EQ(nbr.get_distance(), 0.0f);
                }
            }
        }
    }
}

TEST(PractheoConstruct, ExcludesSelfByIdWithInnerProductDistance) {
    using dot_factory_t = practheo::factory_t<DistanceMetricsT::DOT, dimension>;
    auto base = line_vectors(4);
    for (vertex_id_t i = 0; i < 4; ++i) base.get(i)[0] = static_cast<float>(1u << i);
    config_t config(4, 1.f, 0.25f, 1);
    config.sampling_rate1 = 0;
    const auto result = dot_factory_t::construct_graph(base, sl_config_t(1), config);
    EXPECT_EQ(result.graph.fetch_nbrs(0), (nbr_arr_t{{3, -8.0f}}));
    EXPECT_EQ(result.graph.fetch_nbrs(3), (nbr_arr_t{{2, -32.0f}}));
}

TEST(PractheoConstruct, NonBruteforceBranchOnlyInitializesWeights) {
    const auto base = line_vectors(8);
    const auto result = factory_t::construct_graph<false>(base, sl_config_t(2), config_t(8, 1.f, 0.25f, 2));
    std::size_t assigned = 0;
    for (const auto& group : result.graph.get_vids_by_weight()) assigned += group.size();
    EXPECT_EQ(assigned, 8u);
    for (const auto& row : result.graph.get_nbrs_arr()) EXPECT_TRUE(row.empty());
}
}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
