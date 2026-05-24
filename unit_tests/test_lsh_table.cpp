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
 * @FilePath: /Artea/tests/test_lsh_table.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description:
 */

#include <iostream>
#include <vector>
#include <memory>
#include <filesystem>
#include <fmt/format.h>
#include <argparse/argparse.hpp>
#include <gtest/gtest.h>
#include <tbb/parallel_for.h>
#include <gtest/gtest.h>

// Artea Headers
#include <artea/cpu/framework/artea.hpp>

using namespace artea;
using namespace artea::cpu;

// Typename Definitions
using vec_num_t = uint32_t;
using vec_ele_t = float;
static constexpr BufferPolicyT buffer_policy = BufferPolicyT::LOCKED_BUFFER_WITH_MUTEX;
static constexpr std::size_t buffer_size = 256;
using base_traits_t = BaseTraits<vec_num_t, vec_ele_t>;
using computer_traits_t = ComputerTraits<base_traits_t, DistanceMetricsT::EUCLIDEAN>;
using index_traits_t = IndexTraits<base_traits_t>;
using vertex_generator_traits_t = VertexGeneratorTraits<computer_traits_t, index_traits_t>;
using lsh_table_t = typename vertex_generator_traits_t::lsh_table_t;
using ortho_lsh_generator_t = typename vertex_generator_traits_t::ortho_lsh_generator_t;
using pstable_lsh_generator_t = typename vertex_generator_traits_t::pstable_lsh_generator_t;
using hash_num_t = typename vertex_generator_traits_t::hash_num_t;
using vector_array_t = typename vertex_generator_traits_t::vector_array_t;
using vector_dataset_t = typename vertex_generator_traits_t::vector_dataset_t;
using simd_dispatcher_t = SIMDDistanceDispatcher<computer_traits_t, 1>;
using fma_func_t = typename vertex_generator_traits_t::fma_func_t;

static constexpr hash_num_t num_hashes = 8;
static constexpr vec_ele_t bucket_scale = 1.0f;

struct TestConfig {
    std::string config_path;
    std::string dataset_name;
    int num_samples;
    bool verbose;
} g_config;

// Singleton DataProvider to load dataset once

class DataProvider {
public:
    static DataProvider& instance() { static DataProvider inst; return inst; }

    void init() {
        if (!std::filesystem::exists(g_config.config_path)) {
            throw std::runtime_error("Config file not found: " + g_config.config_path);
        }
        ARTEA_INFO(fmt::format("Loading Dataset: {} from {}", g_config.dataset_name, g_config.config_path));
        dataset_ = std::make_unique<vector_dataset_t>(g_config.config_path, g_config.dataset_name);
        dispatcher_ = std::make_unique<simd_dispatcher_t>(dataset_->get_base_vecs().get_vec_dim());
        fma_func_ = std::make_unique<fma_func_t>(dataset_->get_base_vecs().get_vec_dim());
    }

    vector_dataset_t& get_dataset() { return *dataset_; }
    simd_dispatcher_t& get_dispatcher() { return *dispatcher_; }
    fma_func_t& get_fma_func() { return *fma_func_; }

private:
    DataProvider() = default;
    std::unique_ptr<vector_dataset_t> dataset_;
    std::unique_ptr<simd_dispatcher_t> dispatcher_;
    std::unique_ptr<fma_func_t> fma_func_;
};

class LSHTableCorrectnessTest : public ::testing::Test {};

TEST_F(LSHTableCorrectnessTest, VerifyRecallAccuracy) {
    auto& provider = DataProvider::instance();
    auto& dataset = provider.get_dataset();
    auto& fma_func = provider.get_fma_func();

    const auto& base_vecs = dataset.get_base_vecs();

    pstable_lsh_generator_t pstable_lsh_generator(fma_func);
    ortho_lsh_generator_t ortho_lsh_generator(fma_func);

    /** @brief Test P-Stable LSH Generator. */
    auto pstable_lsh_table = pstable_lsh_generator.generate(base_vecs.get_vec_dim(), num_hashes, bucket_scale);
    /** @brief Test Ortho LSH Generator. */
    auto ortho_lsh_table = ortho_lsh_generator.generate(base_vecs.get_vec_dim(), num_hashes, bucket_scale);

    vector_array_t res_vec_array1(g_config.num_samples, num_hashes);
    vector_array_t res_vec_array2(g_config.num_samples, num_hashes);

    /** @brief Test P-Stable LSH Table Computing */
    tbb::parallel_for(tbb::blocked_range<vec_num_t>(0, g_config.num_samples),
        [&](const tbb::blocked_range<vec_num_t>& r) {
            for (vec_num_t i = r.begin(); i != r.end(); ++i) {
                pstable_lsh_table.compute(base_vecs.get(i), res_vec_array1.get(i));
            }
        }
    );
    /** @brief Test Ortho LSH Table Computing */
    tbb::parallel_for(tbb::blocked_range<vec_num_t>(0, g_config.num_samples),
        [&](const tbb::blocked_range<vec_num_t>& r) {
            for (vec_num_t i = r.begin(); i != r.end(); ++i) {
                ortho_lsh_table.compute(base_vecs.get(i), res_vec_array2.get(i));
            }
        }
    );
    /** @brief Test P-Stable LSH Table Update Bucket Scale */
    pstable_lsh_table.update_bucket_scale(bucket_scale * 2.0f);
    /** @brief Test Ortho LSH Table Update Bucket Scale */
    ortho_lsh_table.update_bucket_scale(bucket_scale * 2.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    argparse::ArgumentParser program("test_lsh_table");

    program.add_argument("-c", "--config").default_value(artea::default_dataset_config_path());
    program.add_argument("-d", "--dataset").default_value(std::string("sift-1m"));
    program.add_argument("-s", "--samples").default_value(100).scan<'i', int>().help("Number of samples (queries) to test [Ignored for full batch query]");
    program.add_argument("-v", "--verbose").default_value(false).implicit_value(true);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    g_config.config_path = program.get<std::string>("--config");
    g_config.dataset_name = program.get<std::string>("--dataset");
    g_config.num_samples = program.get<int>("--samples");
    g_config.verbose = program.get<bool>("--verbose");

    // Initialize data before running tests
    DataProvider::instance().init();

    return RUN_ALL_TESTS();
}