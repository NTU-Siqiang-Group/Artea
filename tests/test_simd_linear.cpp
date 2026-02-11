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
 * @FilePath: /Artea/tests/test_simd_linear.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Test suite for SIMD Linear Transformation (Dot Product + Bias)
 *               using Google Test for correctness verification.
 */

#include <gtest/gtest.h>
#include <argparse/argparse.hpp>
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/utils/simd_linear.hpp>
#include <faiss/utils/distances.h>
#include <vector>
#include <cstring>

using namespace artea;
using namespace artea::cpu;

using base_traits_t = BaseTraits<uint32_t, float, false>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::DOT>;
template <std::size_t U> using artea_simd_linear_t = SIMDLinear<computer_traits_t, U>;

struct TestConfig {
    std::string config_path, dataset_name;
    int num_samples;
    bool verbose;
} g_config;

class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }
    void init() {
        auto dataset = std::make_unique<typename base_traits_t::vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dim_ = dataset->get_base_vecs().get_vec_dim();
        a_.resize(dim_);
        // Resize based on requested samples
        x_.resize(g_config.num_samples * dim_);
        bias_ = 4.25f;

        std::memcpy(a_.data(), dataset->get_query_vecs().get(0), dim_ * sizeof(float));
        for(int i=0; i < g_config.num_samples; ++i)
            std::memcpy(x_.data() + i * dim_, dataset->get_base_vecs().get(i), dim_ * sizeof(float));
    }
    uint32_t get_dim() const { return dim_; }
    float* get_a() { return a_.data(); }
    float* get_x(int i) { return x_.data() + i * dim_; }
    float get_b() const { return bias_; }
private:
    uint32_t dim_; float bias_;
    std::vector<float> a_, x_;
};

TEST(LinearCorrectness, VerifyAxPlusB_Multi) {
    auto& p = DataProvider::instance();

    for(int i=0; i < g_config.num_samples; ++i) {
        float* x = p.get_x(i);

        // Ground Truth: manual calculation
        float dot = 0;
        for(size_t j=0; j<p.get_dim(); ++j) dot += p.get_a()[j] * x[j];
        float gt = dot + p.get_b();

        // 1. Artea Linear (1, 2, 4)
        artea_simd_linear_t<1> u1(p.get_dim());
        artea_simd_linear_t<2> u2(p.get_dim());
        artea_simd_linear_t<4> u4(p.get_dim());

        EXPECT_NEAR(u1(p.get_a(), x, p.get_b()), gt, 1e-4) << "Artea U1 failed at " << i;
        EXPECT_NEAR(u2(p.get_a(), x, p.get_b()), gt, 1e-4) << "Artea U2 failed at " << i;
        EXPECT_NEAR(u4(p.get_a(), x, p.get_b()), gt, 1e-4) << "Artea U4 failed at " << i;

        // 2. Cross Check with Faiss (IP + Bias)
        float faiss_res = faiss::fvec_inner_product(p.get_a(), x, p.get_dim()) + p.get_b();
        EXPECT_NEAR(faiss_res, gt, 1e-4) << "Faiss IP+Bias mismatch at " << i;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    argparse::ArgumentParser program("test_simd_linear");
    program.add_argument("-c", "--config").default_value(std::string("./datasets.json"));
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("-s", "--samples").default_value(10).scan<'i', int>().help("Number of samples to test");
    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try { program.parse_args(argc, argv); } catch(...) { return 1; }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.num_samples = program.get<int>("--samples");
    g_config.verbose = program.get<bool>("--verbose");

    DataProvider::instance().init();
    return RUN_ALL_TESTS();
}