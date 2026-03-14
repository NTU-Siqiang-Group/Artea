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
 * @FilePath: /Artea/tests/test_simd_distance.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Test suite for SIMD distance calculations,
 *               using Google Test for correctness verification.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <filesystem>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>
#include <artea/cpu/framework/artea.hpp>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <hnswlib/hnswlib.h>

using namespace artea;
using namespace artea::cpu;

using base_traits_t = BaseTraits<uint32_t, float, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using vector_dataset_t = typename base_traits_t::vector_dataset_t;
// Define aliases for all unroll sizes
template <std::size_t U> using artea_simd_dist_t = computer_traits_t::template simd_dist_t<U>;

struct TestConfig {
    std::string config_path, dataset_name;
    int num_samples;
    bool verbose;
} g_config;

class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }
    void init() {
        auto dataset = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dim_ = dataset->get_base_vecs().get_vec_dim();
        query_vec_.resize(dim_);
        targets_.resize(g_config.num_samples * dim_);
        std::memcpy(query_vec_.data(), dataset->get_query_vecs().get(0), dim_ * sizeof(float));
        for(int i = 0; i < g_config.num_samples; ++i) {
            std::memcpy(targets_.data() + i * dim_, dataset->get_base_vecs().get(i), dim_ * sizeof(float));
        }
    }
    uint32_t get_dim() const { return dim_; }
    float* get_query() { return query_vec_.data(); }
    float* get_target(int idx) { return targets_.data() + idx * dim_; }

private:
    DataProvider() = default;
    uint32_t dim_ = 0;
    std::vector<float> query_vec_, targets_;
};

// Ground Truth
float cpp_L2sqr(const float* x, const float* y, const size_t d) {
    float res = 0.0f;
    for (size_t i = 0; i < d; ++i) { float diff = x[i] - y[i]; res += diff * diff; }
    return res;
}

TEST(DistanceCorrectness, VerifyMultiTargetAndEngines) {
    auto& provider = DataProvider::instance();
    uint32_t dim = provider.get_dim();
    float* q = provider.get_query();

    // Verify against multiple targets to ensure stability
    for (int i = 0; i < g_config.num_samples; ++i) {
        float* t = provider.get_target(i);
        float gt = cpp_L2sqr(q, t, dim);

        // 1. Artea Variants (1, 2, 4)
        artea_simd_dist_t<1> u1(dim);
        artea_simd_dist_t<2> u2(dim);
        artea_simd_dist_t<4> u4(dim);

        EXPECT_NEAR(u1(q, t), gt, 1e-5) << "Artea U1 failed at target " << i;
        EXPECT_NEAR(u2(q, t), gt, 1e-5) << "Artea U2 failed at target " << i;
        EXPECT_NEAR(u4(q, t), gt, 1e-5) << "Artea U4 failed at target " << i;

        // 2. Faiss
        EXPECT_NEAR(faiss::fvec_L2sqr(q, t, dim), gt, 1e-5) << "Faiss failed at target " << i;

        // 3. HNSWLib
        hnswlib::L2Space l2space(dim);
        float hnsw_res = l2space.get_dist_func()(q, t, l2space.get_dist_func_param());
        EXPECT_NEAR(hnsw_res, gt, 1e-5) << "HNSWLib failed at target " << i;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    argparse::ArgumentParser program("test_simd_distance");
    program.add_argument("-c", "--config").default_value(std::string("./configs/datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("-s", "--samples").default_value(10).scan<'i', int>().help("Number of samples to test");
    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try { program.parse_args(argc, argv); } catch (const std::runtime_error& err) { return 1; }
    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.num_samples = program.get<int>("--samples");
    g_config.verbose = program.get<bool>("--verbose");

    DataProvider::instance().init();
    return RUN_ALL_TESTS();
}