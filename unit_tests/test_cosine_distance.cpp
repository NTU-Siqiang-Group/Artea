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
 * @FilePath: /Artea/unit_tests/test_cosine_distance.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Cosine distance kernel + infra_dispatcher tests. Self-contained
 *               (synthetic vectors at compile-time dimensions), so it needs no
 *               dataset. Verifies:
 *                 - SIMDDistance<COSINE> == 1 - dot/(||a||*||b||) (U1/U2/U4)
 *                 - raw dot_product() and negated DOT distance (U1/U2/U4)
 *                 - identical-direction -> 0, opposite -> 2, zero-vec -> 1
 *                 - infra_dispatcher: parse_metric / infra_t / acquire /
 *                   dispatch resolve the right <Metric, Dim>.
 */

#include <array>
#include <cmath>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <artea/cpu/framework/type_context/infra_dispatcher.hpp>

using namespace artea;
using namespace artea::cpu;

namespace {

constexpr DistanceMetricsT COS = DistanceMetricsT::COSINE;
constexpr DistanceMetricsT L2  = DistanceMetricsT::EUCLIDEAN;
constexpr DistanceMetricsT IP  = DistanceMetricsT::DOT;        // inner product / MIPS
constexpr vec_dim_t        DIM = 64;   // multiple of SIMD chunk size (16)

// Scalar reference: cosine distance = 1 - dot/(||a||*||b||).
float ref_cosine(const float* a, const float* b, std::size_t dim) {
    double dot = 0, na = 0, nb = 0;
    for (std::size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    const double denom = std::sqrt(na * nb);
    return denom > 0.0 ? static_cast<float>(1.0 - dot / denom) : 1.0f;
}

// Deterministic pseudo-random vector, varied per seed.
std::vector<float> make_vec(uint32_t seed, std::size_t dim = DIM) {
    std::vector<float> v(dim);
    uint32_t s = seed * 2654435761u + 12345u;
    for (std::size_t i = 0; i < dim; ++i) {
        s = s * 1103515245u + 12345u;
        v[i] = static_cast<float>(static_cast<int>((s >> 9) & 0xFFFF) - 32768) / 1024.0f;
    }
    return v;
}

}  // namespace

TEST(CosineKernel, MatchesScalarReferenceAllUnrolls) {
    // The dimension is a compile-time trait now; functors are default-constructed.
    simdu1_dist_t<COS, DIM> u1;
    simdu2_dist_t<COS, DIM> u2;
    simdu4_dist_t<COS, DIM> u4;

    for (uint32_t i = 0; i < 16; ++i) {
        auto a = make_vec(i * 2 + 1);
        auto b = make_vec(i * 2 + 2);
        const float gt = ref_cosine(a.data(), b.data(), DIM);
        EXPECT_NEAR(u1(a.data(), b.data()), gt, 1e-5) << "U1 @ " << i;
        EXPECT_NEAR(u2(a.data(), b.data()), gt, 1e-5) << "U2 @ " << i;
        EXPECT_NEAR(u4(a.data(), b.data()), gt, 1e-5) << "U4 @ " << i;
    }
}

TEST(CosineKernel, EdgeCases) {
    dist_func_t<COS, DIM> d;
    auto a = make_vec(7);

    // Identical direction -> distance 0.
    EXPECT_NEAR(d(a.data(), a.data()), 0.0f, 1e-5);

    // Scaled same direction -> still 0 (cosine ignores magnitude).
    auto a_scaled = a;
    for (auto& x : a_scaled) x *= 3.5f;
    EXPECT_NEAR(d(a.data(), a_scaled.data()), 0.0f, 1e-5);

    // Opposite direction -> distance 2.
    auto a_neg = a;
    for (auto& x : a_neg) x = -x;
    EXPECT_NEAR(d(a.data(), a_neg.data()), 2.0f, 1e-5);

    // Zero-length vector -> guarded to max distance 1.
    std::vector<float> z(DIM, 0.0f);
    EXPECT_NEAR(d(z.data(), a.data()), 1.0f, 1e-5);
    EXPECT_NEAR(d(z.data(), z.data()), 1.0f, 1e-5);
}

TEST(InnerProductKernel, MatchesNegatedDot) {
    // dot_product() exposes the algebraic result needed by projection users;
    // inner-product distance negates it so smaller remains more similar.
    simdu1_dist_t<IP, DIM> u1;
    simdu2_dist_t<IP, DIM> u2;
    simdu4_dist_t<IP, DIM> u4;
    for (uint32_t i = 0; i < 16; ++i) {
        auto a = make_vec(i * 2 + 1);
        auto b = make_vec(i * 2 + 2);
        double dot = 0;
        for (std::size_t k = 0; k < DIM; ++k) dot += static_cast<double>(a[k]) * b[k];
        const float dot_gt = static_cast<float>(dot);
        const float dist_gt = -dot_gt;
        const float tol = 1e-3f * std::max(1.0f, std::fabs(dot_gt));

        EXPECT_NEAR(u1.dot_product(a.data(), b.data()), dot_gt, tol) << "raw U1 @ " << i;
        EXPECT_NEAR(u2.dot_product(a.data(), b.data()), dot_gt, tol) << "raw U2 @ " << i;
        EXPECT_NEAR(u4.dot_product(a.data(), b.data()), dot_gt, tol) << "raw U4 @ " << i;
        EXPECT_NEAR(u1(a.data(), b.data()), dist_gt, tol) << "distance U1 @ " << i;
        EXPECT_NEAR(u2(a.data(), b.data()), dist_gt, tol) << "distance U2 @ " << i;
        EXPECT_NEAR(u4(a.data(), b.data()), dist_gt, tol) << "distance U4 @ " << i;
    }
}

TEST(InfraDispatcher, ParseMetric) {
    // parse_metric turns workload/CLI metric spellings (plus aliases) into the
    // DistanceMetricsT consumed by infra_dispatch(); callers pair it with the loaded
    // dataset's SIMD-padded dim to form a DatasetInfra.
    EXPECT_EQ(parse_metric("euclidean"),     L2);
    EXPECT_EQ(parse_metric("l2"),            L2);
    EXPECT_EQ(parse_metric("inner_product"), IP);
    EXPECT_EQ(parse_metric("dot"),           IP);
    EXPECT_EQ(parse_metric("ip"),            IP);
    EXPECT_EQ(parse_metric("mips"),          IP);
    EXPECT_EQ(parse_metric("cosine"),        COS);
    EXPECT_EQ(parse_metric("angular"),       COS);
    EXPECT_THROW(parse_metric("no-such-metric"), std::runtime_error);

    // metric_name round-trips the canonical spellings.
    EXPECT_STREQ(metric_name(L2),  "euclidean");
    EXPECT_STREQ(metric_name(IP),  "inner_product");
    EXPECT_STREQ(metric_name(COS), "cosine");
}

TEST(InfraDispatcher, InfraTypeAndAcquire) {
    // infra_t resolves to the metric/dim-correct alias.
    static_assert(std::is_same_v<infra_t<COS, DIM, InfraKind::SimdDistance>,
                                 dist_func_t<COS, DIM>>);
    static_assert(std::is_same_v<infra_t<L2, 128, InfraKind::HierRouter>,
                                 hierarchical_graph_router_t<L2, 128>>);

    // acquire constructs a usable functor (no ctor dim arg).
    auto d = acquire<COS, DIM, InfraKind::SimdDistance>();
    auto a = make_vec(3), b = make_vec(4);
    EXPECT_NEAR(d(a.data(), b.data()), ref_cosine(a.data(), b.data(), DIM), 1e-5);
}

TEST(InfraDispatcher, DispatchResolvesCompileTimePair) {
    // dispatch bridges a runtime (metric, dim) to the compile-time <Metric, Dim>.
    // Infos are caller-built (metric parsed from input, dim from the loaded
    // dataset); use a synthetic COSINE one to confirm the cosine branch
    // resolves the right compile-time pair.
    const DatasetInfra cos_info{COS, 112};
    const int cos_tag = infra_dispatch(cos_info, ARTEA_METRIC_LAMBDA(int) {
        static_assert(std::is_same_v<infra_t<Metric, Dim, InfraKind::SimdDistance>,
                                     dist_func_t<Metric, Dim>>);
        return (Metric == COS && Dim == 112) ? 42 : 0;
    });
    EXPECT_EQ(cos_tag, 42);

    // And the euclidean branch (e.g. sift-1m's padded dim).
    const int l2_tag = infra_dispatch(DatasetInfra{L2, 128}, ARTEA_METRIC_LAMBDA(int) {
            return (Metric == L2 && Dim == 128) ? 7 : 0;
        });
    EXPECT_EQ(l2_tag, 7);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
