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

#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>

using namespace artea::cpu;

namespace {

template <std::size_t Unroll>
void check_known_distances() {
    constexpr vec_dim_t dim = 112;  // Seven SIMD chunks exercise unroll tails.
    computer_traits_t<DistanceMetricsT::EUCLIDEAN, dim>::simd_dist_t<Unroll> l2;
    computer_traits_t<DistanceMetricsT::EUCLIDEAN_SQR, dim>::simd_dist_t<Unroll> l2_sqr;
    std::array<float, dim> a{}, b{};
    b.front() = 3.0f;
    b.back() = -4.0f;

    EXPECT_FLOAT_EQ(l2(a.data(), b.data()), 5.0f);
    EXPECT_FLOAT_EQ(l2(b.data(), a.data()), 5.0f);
    EXPECT_FLOAT_EQ(l2_sqr(a.data(), b.data()), 25.0f);
    EXPECT_FLOAT_EQ(l2(a.data(), a.data()), 0.0f);
    EXPECT_FLOAT_EQ(l2(b.data(), b.data()), 0.0f);
}

template <vec_dim_t Dim>
void check_scalar_reference() {
    simdu1_dist_t<DistanceMetricsT::EUCLIDEAN, Dim> u1;
    simdu2_dist_t<DistanceMetricsT::EUCLIDEAN, Dim> u2;
    simdu4_dist_t<DistanceMetricsT::EUCLIDEAN, Dim> u4;
    std::array<float, Dim> a{}, b{};
    for (unsigned seed = 0; seed < 16; ++seed) {
        double squared_sum = 0.0;
        for (std::size_t i = 0; i < Dim; ++i) {
            a[i] = static_cast<float>(static_cast<int>((i * 13 + seed * 17) % 101) - 50) / 7.0f;
            b[i] = static_cast<float>(static_cast<int>((i * 29 + seed * 11) % 97) - 48) / 9.0f;
            const double diff = static_cast<double>(a[i]) - b[i];
            squared_sum += diff * diff;
        }
        const double expected = std::sqrt(squared_sum);
        const double tolerance = expected * 2e-6;
        SCOPED_TRACE(::testing::Message() << "dim=" << Dim << ", seed=" << seed);
        EXPECT_NEAR(u1(a.data(), b.data()), expected, tolerance);
        EXPECT_NEAR(u2(a.data(), b.data()), expected, tolerance);
        EXPECT_NEAR(u4(a.data(), b.data()), expected, tolerance);
    }
}

}  // namespace

TEST(EuclideanKernel, KnownDistancesAndZeroAllUnrolls) {
    check_known_distances<1>();
    check_known_distances<2>();
    check_known_distances<4>();
}

TEST(EuclideanKernel, MatchesScalarReferenceAllUnrolls) {
    check_scalar_reference<16>();
    check_scalar_reference<48>();
    check_scalar_reference<96>();
    check_scalar_reference<112>();
    check_scalar_reference<128>();
    check_scalar_reference<304>();
    check_scalar_reference<384>();
    check_scalar_reference<960>();
}

TEST(EuclideanKernel, DispatchAndAcquireAllSupportedDimensions) {
    std::array<float, 960> a{}, b{};
    for (const vec_dim_t dim : {96, 112, 128, 304, 384, 960}) {
        b.fill(0.0f);
        b.front() = 3.0f;
        b[dim - 1] = 4.0f;
        for (const char* name : {"euclidean", "l2"}) {
            SCOPED_TRACE(::testing::Message() << "metric=" << name << ", dim=" << dim);
            const auto distance = infra_dispatch(DatasetInfra{parse_metric(name), dim},
                ARTEA_METRIC_LAMBDA(float) {
                    EXPECT_EQ(Metric, DistanceMetricsT::EUCLIDEAN);
                    EXPECT_EQ(Dim, dim);
                    auto dist = acquire<Metric, Dim, InfraKind::SimdDistance>();
                    return dist(a.data(), b.data());
                });
            EXPECT_FLOAT_EQ(distance, 5.0f);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
