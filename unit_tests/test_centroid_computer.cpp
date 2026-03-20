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
 * @FilePath: /Artea/tests/test_centroid_computer.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Test for CentroidComputer comparing SIMD vs Serial implementation.
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

// Serial implementation for comparison
class SerialCentroidComputer {
public:
    static auto compute(const vector_array_t& vecs) -> std::vector<vec_ele_t> {
        const vertex_num_t num_vecs = vecs.get_num_vecs();
        const vec_dim_t vec_dim = vecs.get_vec_dim();

        if (num_vecs == 0) {
            return std::vector<vec_ele_t>(vec_dim, 0.0f);
        }

        std::vector<vec_ele_t> centroid(vec_dim, 0.0f);

        // Sum all vectors
        for (vertex_num_t i = 0; i < num_vecs; ++i) {
            const vec_ele_t* vec = vecs.get(i);
            for (vec_dim_t d = 0; d < vec_dim; ++d) {
                centroid[d] += vec[d];
            }
        }

        // Divide by number of vectors to get mean
        const vec_ele_t inv_num_vecs = static_cast<vec_ele_t>(1.0) / static_cast<vec_ele_t>(num_vecs);
        for (vec_dim_t d = 0; d < vec_dim; ++d) {
            centroid[d] *= inv_num_vecs;
        }

        return centroid;
    }
};

class CentroidComputerTest : public ::testing::Test {
protected:
    static constexpr vec_dim_t vec_dim = 128;
    static constexpr vertex_num_t num_vecs = 1000;

    std::unique_ptr<vector_array_t> vecs;

    void SetUp() override {
        // Generate random vectors
        std::mt19937 rng(42);
        std::uniform_real_distribution<vec_ele_t> dist(-1.0f, 1.0f);

        vecs = std::make_unique<vector_array_t>(num_vecs, vec_dim);

        for (vertex_num_t i = 0; i < num_vecs; ++i) {
            vec_ele_t* vec = vecs->get(i);
            for (vec_dim_t d = 0; d < vec_dim; ++d) {
                vec[d] = dist(rng);
            }
        }
    }

    // Helper function to compare two centroids
    bool are_centroids_equal(const std::vector<vec_ele_t>& c1,
                            const std::vector<vec_ele_t>& c2,
                            vec_ele_t tolerance = 1e-5f) {
        if (c1.size() != c2.size()) {
            return false;
        }

        for (size_t i = 0; i < c1.size(); ++i) {
            if (std::abs(c1[i] - c2[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
};

TEST_F(CentroidComputerTest, CompareAVX512WithSerial) {
    // Compute centroid using SIMD vectorization
    auto centroid_simd = centroid_computer_t::compute(*vecs);

    // Compute centroid using serial implementation
    auto centroid_serial = SerialCentroidComputer::compute(*vecs);

    // Compare results
    ASSERT_EQ(centroid_simd.size(), vec_dim);
    ASSERT_EQ(centroid_serial.size(), vec_dim);

    EXPECT_TRUE(are_centroids_equal(centroid_simd, centroid_serial))
        << "SIMD and Serial centroids differ";

    // Print first few dimensions for manual verification
    std::cout << "First 10 dimensions comparison:\n";
    std::cout << "SIMD: ";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << centroid_simd[i] << " ";
    }
    std::cout << "\nSerial: ";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << centroid_serial[i] << " ";
    }
    std::cout << "\n";
}

TEST_F(CentroidComputerTest, EmptyVectorArray) {
    vector_array_t empty_vecs(0, vec_dim);

    auto centroid_simd = centroid_computer_t::compute(empty_vecs);
    auto centroid_serial = SerialCentroidComputer::compute(empty_vecs);

    ASSERT_EQ(centroid_simd.size(), vec_dim);
    ASSERT_EQ(centroid_serial.size(), vec_dim);

    // All elements should be zero
    for (vec_dim_t d = 0; d < vec_dim; ++d) {
        EXPECT_FLOAT_EQ(centroid_simd[d], 0.0f);
        EXPECT_FLOAT_EQ(centroid_serial[d], 0.0f);
    }
}

TEST_F(CentroidComputerTest, SingleVector) {
    vector_array_t single_vec(1, vec_dim);

    // Fill with known values
    vec_ele_t* vec = single_vec.get(0);
    for (vec_dim_t d = 0; d < vec_dim; ++d) {
        vec[d] = static_cast<vec_ele_t>(d);
    }

    auto centroid_simd = centroid_computer_t::compute(single_vec);
    auto centroid_serial = SerialCentroidComputer::compute(single_vec);

    EXPECT_TRUE(are_centroids_equal(centroid_simd, centroid_serial));

    // Centroid should equal the single vector
    for (vec_dim_t d = 0; d < vec_dim; ++d) {
        EXPECT_FLOAT_EQ(centroid_simd[d], static_cast<vec_ele_t>(d));
    }
}

TEST_F(CentroidComputerTest, LargeDataset) {
    constexpr vertex_num_t large_num_vecs = 10000;
    vector_array_t large_vecs(large_num_vecs, vec_dim);

    std::mt19937 rng(123);
    std::uniform_real_distribution<vec_ele_t> dist(-10.0f, 10.0f);

    for (vertex_num_t i = 0; i < large_num_vecs; ++i) {
        vec_ele_t* vec = large_vecs.get(i);
        for (vec_dim_t d = 0; d < vec_dim; ++d) {
            vec[d] = dist(rng);
        }
    }

    auto centroid_simd = centroid_computer_t::compute(large_vecs);
    auto centroid_serial = SerialCentroidComputer::compute(large_vecs);

    EXPECT_TRUE(are_centroids_equal(centroid_simd, centroid_serial, 1e-4f))
        << "SIMD and Serial centroids differ for large dataset";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
