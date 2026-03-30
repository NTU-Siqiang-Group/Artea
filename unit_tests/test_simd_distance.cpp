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
 * @FilePath: /Artea/unit_tests/test_simd_distance.cpp
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
#include <artea/cpu/framework/type_context/default_context.hpp>
#include <hnswlib/hnswlib.h>

using namespace artea;
using namespace artea::cpu;

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
        _dim = dataset->get_base_vecs().get_vec_dim();
        _query_vec.resize(_dim);
        _targets.resize(g_config.num_samples * _dim);
        std::memcpy(_query_vec.data(), dataset->get_query_vecs().get(0), _dim * sizeof(float));
        for (int i = 0; i < g_config.num_samples; ++i) {
            std::memcpy(_targets.data() + i * _dim, dataset->get_base_vecs().get(i), _dim * sizeof(float));
        }
    }
    uint32_t get_dim() const { return _dim; }
    float* get_query() { return _query_vec.data(); }
    float* get_target(int idx) { return _targets.data() + idx * _dim; }

private:
    DataProvider() = default;
    uint32_t _dim = 0;
    std::vector<float> _query_vec, _targets;
};

/** @brief Scalar ground truth: simple for-loop L2 squared distance. */
static float simple_L2sqr(const float* a, const float* b, uint32_t dim) {
    float result = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}

TEST(DistanceCorrectness, VerifyMultiTargetAndEngines) {
    auto& p = DataProvider::instance();
    uint32_t dim = p.get_dim();
    float* q = p.get_query();

    for (int i = 0; i < g_config.num_samples; ++i) {
        float* t = p.get_target(i);
        float gt = simple_L2sqr(q, t, dim);

        // 1. Artea SIMDDistance U1/U2/U4
        artea_simd_dist_t<1> u1(dim);
        artea_simd_dist_t<2> u2(dim);
        artea_simd_dist_t<4> u4(dim);

        EXPECT_NEAR(u1(q, t), gt, 1e-3) << "Artea U1 failed at target " << i;
        EXPECT_NEAR(u2(q, t), gt, 1e-3) << "Artea U2 failed at target " << i;
        EXPECT_NEAR(u4(q, t), gt, 1e-3) << "Artea U4 failed at target " << i;

        // 2. HNSWLib
        hnswlib::L2Space l2space(dim);
        float hnsw_res = l2space.get_dist_func()(q, t, l2space.get_dist_func_param());
        EXPECT_NEAR(hnsw_res, gt, 1e-3) << "HNSWLib failed at target " << i;
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
