// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include <artea/cpu/framework/type_traits/base_traits.hpp>
#include <artea/cpu/framework/type_traits/index_traits.hpp>
#include <artea/cpu/index/exact_artea/configs.hpp>

namespace {
using Traits = artea::cpu::IndexTraits<artea::cpu::BaseTraits<uint32_t, float>>;
using RGraphConfig = Traits::exact_artea::rgraph_config_t;
using PruningConfig = Traits::exact_artea::pruning_config_t;
static_assert(std::is_same_v<RGraphConfig, Traits::stacked_rgraph::rgraph_config_t>);
static_assert(std::is_same_v<RGraphConfig, Traits::artea_graph::rgraph_config_t>);
static_assert(std::is_same_v<decltype(std::declval<const RGraphConfig&>().early_stop_threshold(
    1, std::declval<const PruningConfig&>())), Traits::distance_t>);

TEST(RGraphThresholds, KnownARCAndEarlyStopValuesWithThetaOne) {
    // R=[2,8,16,32,...], alpha=2, theta=1 => gamma=4; tau*rho=1.
    const RGraphConfig config(2, 3, 0.5f, 2, 16);
    const PruningConfig pruning(2, 0.5f, 2);
    EXPECT_FLOAT_EQ(config.early_stop_threshold(1, pruning), 44);
    EXPECT_FLOAT_EQ(config.early_stop_threshold(2, pruning), 108);
    EXPECT_FLOAT_EQ(config.early_stop_threshold(3, pruning), 236);
    // Regression values for ARC after sharing its factors with early stopping.
    EXPECT_FLOAT_EQ(config.aspect_ratio_constraint(0, pruning), 23.5f);
    EXPECT_FLOAT_EQ(config.aspect_ratio_constraint(1, pruning), 14.875f);
    EXPECT_FLOAT_EQ(config.aspect_ratio_constraint(2, pruning), 16.4375f);
    EXPECT_FLOAT_EQ(config.aspect_ratio_constraint(3, pruning), 17.21875f);
}

TEST(RGraphThresholds, MatchesPDFGeometricRadiusClosedForm) {
    // tau_k=beta-1 recovers the PDF's R_h=rho*beta^h. For these parameters,
    // gamma*psi_h*R_h = 16*2^h-4, independently of the radius-sum implementation.
    const RGraphConfig config(2, 1, 0.5f, 2, 16);
    const PruningConfig pruning(2, 0.5f, 2);
    for (uint32_t h : {1u, 2u, 3u, 10u, 30u}) {
        SCOPED_TRACE(h);
        EXPECT_FLOAT_EQ(config.early_stop_threshold(h, pruning),
                        static_cast<float>(std::ldexp(16.0, h) - 4));
    }
}

TEST(RGraphThresholds, EarlyStopScalesAsDistanceWhileARCIsDimensionless) {
    const RGraphConfig original(2, 3, 0.5f, 2, 16);
    const RGraphConfig scaled(2, 3, 0.5f, 20, 16);
    const PruningConfig pruning(2, 0.5f, 2);
    const PruningConfig scaled_pruning(2, 0.5f, 20);
    for (uint32_t h : {1u, 2u, 3u}) {
        EXPECT_FLOAT_EQ(scaled.early_stop_threshold(h, scaled_pruning),
                        10 * original.early_stop_threshold(h, pruning));
        EXPECT_FLOAT_EQ(scaled.aspect_ratio_constraint(h, scaled_pruning),
                        original.aspect_ratio_constraint(h, pruning));
    }
}

TEST(RGraphThresholds, UsesActualL1RadiusAndConsistentARCEntryBound) {
    const RGraphConfig config(2, 0, 0.5f, 2, 16);
    const PruningConfig pruning(2, 0.5f, 2);
    EXPECT_FLOAT_EQ(config.early_stop_threshold(1, pruning), 20);
    EXPECT_FLOAT_EQ(config.early_stop_threshold(2, pruning), 36);
    // Delta_h * R_h = psi_h * R_h + T_{h+1}; here gamma=4.
    for (uint32_t h : {1u, 2u, 3u}) {
        EXPECT_FLOAT_EQ(config.aspect_ratio_constraint(h, pruning) * config.radius_at(h),
            config.early_stop_threshold(h, pruning) / 4 + config.early_stop_threshold(h + 1, pruning));
    }
}

TEST(RGraphThresholds, ReadsCurrentPruningValuesWithoutSubstitutingRGraphFields) {
    const RGraphConfig config(2, 3, 20, 2, 16);
    PruningConfig pruning(2, 0.5f, 10);
    EXPECT_FLOAT_EQ(config.early_stop_threshold(1, pruning), 60);
    pruning.shifted_coeffs(0);
    EXPECT_FLOAT_EQ(config.early_stop_threshold(1, pruning), 40);
    pruning.scale_coeffs(3); // gamma=3
    EXPECT_FLOAT_EQ(config.early_stop_threshold(1, pruning), 30);
}

TEST(RGraphThresholds, RejectsBottomLayerAndInvalidLayerSentinel) {
    const RGraphConfig config(2, 3, 0.5f, 2, 16);
    const PruningConfig pruning(2, 0.5f, 2);
    EXPECT_THROW(config.early_stop_threshold(0, pruning), std::out_of_range);
    EXPECT_THROW(config.early_stop_threshold(std::numeric_limits<uint32_t>::max(), pruning), std::out_of_range);
    EXPECT_THROW(config.aspect_ratio_constraint(std::numeric_limits<uint32_t>::max(), pruning), std::out_of_range);
}

TEST(RGraphThresholds, RejectsInvalidPruningParameters) {
    const RGraphConfig config(2, 3, 0.5f, 2, 16);
    const auto check = [&](const PruningConfig& pruning) {
        EXPECT_THROW(config.early_stop_threshold(1, pruning), std::invalid_argument);
        EXPECT_THROW(config.aspect_ratio_constraint(1, pruning), std::invalid_argument);
    };
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (float alpha : {-1.0f, 0.0f, 1.0f, inf, nan}) check(PruningConfig(alpha, 0.5f, 2));
    for (float tau : {-1.0f, inf, nan}) check(PruningConfig(2, tau, 2));
    for (float rho : {-1.0f, 0.0f, inf, nan}) check(PruningConfig(2, 0.5f, rho));
}

TEST(RGraphThresholds, RejectsUnrepresentableLayerRadius) {
    const RGraphConfig config(2, 3, 0.5f, 2, 16);
    const PruningConfig pruning(2, 0.5f, 2);
    EXPECT_THROW(config.early_stop_threshold(256, pruning), std::overflow_error);
    EXPECT_THROW(config.aspect_ratio_constraint(256, pruning), std::overflow_error);
}

TEST(RGraphThresholds, DistanceOverflowDoesNotInvalidateDimensionlessARC) {
    const RGraphConfig config(2, 0, 0, std::numeric_limits<float>::max() / 4, 16);
    const PruningConfig pruning(2, 0, 1);
    EXPECT_THROW(config.early_stop_threshold(1, pruning), std::overflow_error);
    EXPECT_FLOAT_EQ(config.aspect_ratio_constraint(1, pruning), 18);
}

TEST(RGraphThresholds, EarlyStopDoesNotRequireNextLayerRadius) {
    const RGraphConfig config(std::numeric_limits<float>::max(), 0, 0, 2, 16);
    const PruningConfig pruning(2, 0, 2);
    EXPECT_FLOAT_EQ(config.early_stop_threshold(1, pruning), 16);
    EXPECT_THROW(config.aspect_ratio_constraint(1, pruning), std::overflow_error);
}
} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
