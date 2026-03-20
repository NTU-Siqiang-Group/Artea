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

#include <gtest/gtest.h>
#include <vector>
#include <cstdint>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

// Helper function to create idlist_array_t from raw data
static idlist_array_t create_idlist_array(const std::vector<vertex_id_t>& data, uint32_t num_vecs, uint32_t dim) {
    idlist_array_t result(dim);  // Only specify dimension, not num_vecs
    for (uint32_t i = 0; i < num_vecs; ++i) {
        result.append_vec(&data[i * dim]);
    }
    return result;
}

class RecallEstimatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize recall estimator
        estimator_ = std::make_unique<recall_estimator_t>();
    }

    std::unique_ptr<recall_estimator_t> estimator_;
};

TEST_F(RecallEstimatorTest, PerfectRecall) {
    // Create identical predictions and ground truth
    constexpr uint32_t num_queries = 100;
    constexpr uint32_t k = 10;

    std::vector<vertex_id_t> data(num_queries * k);
    for (uint32_t i = 0; i < num_queries; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            data[i * k + j] = i * k + j;
        }
    }

    idlist_array_t predictions = create_idlist_array(data, num_queries, k);
    idlist_array_t groundtruth = create_idlist_array(data, num_queries, k);

    double recall = estimator_->calculate_recall_at_k(predictions, groundtruth);

    EXPECT_DOUBLE_EQ(recall, 1.0) << "Perfect match should yield 100% recall";
}

TEST_F(RecallEstimatorTest, ZeroRecall) {
    // Create completely different predictions and ground truth
    constexpr uint32_t num_queries = 100;
    constexpr uint32_t k = 10;

    std::vector<vertex_id_t> pred_data(num_queries * k);
    std::vector<vertex_id_t> gt_data(num_queries * k);

    for (uint32_t i = 0; i < num_queries; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            pred_data[i * k + j] = i * k + j;
            gt_data[i * k + j] = (i * k + j) + 10000;  // Completely different IDs
        }
    }

    idlist_array_t predictions = create_idlist_array(pred_data, num_queries, k);
    idlist_array_t groundtruth = create_idlist_array(gt_data, num_queries, k);

    double recall = estimator_->calculate_recall_at_k(predictions, groundtruth);

    EXPECT_DOUBLE_EQ(recall, 0.0) << "No overlap should yield 0% recall";
}

TEST_F(RecallEstimatorTest, PartialRecall) {
    // Create predictions with 50% overlap
    constexpr uint32_t num_queries = 100;
    constexpr uint32_t k = 10;

    std::vector<vertex_id_t> pred_data(num_queries * k);
    std::vector<vertex_id_t> gt_data(num_queries * k);

    for (uint32_t i = 0; i < num_queries; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            gt_data[i * k + j] = i * k + j;
            // First half matches, second half doesn't
            if (j < k / 2) {
                pred_data[i * k + j] = i * k + j;
            } else {
                pred_data[i * k + j] = (i * k + j) + 10000;
            }
        }
    }

    idlist_array_t predictions = create_idlist_array(pred_data, num_queries, k);
    idlist_array_t groundtruth = create_idlist_array(gt_data, num_queries, k);

    double recall = estimator_->calculate_recall_at_k(predictions, groundtruth);

    EXPECT_DOUBLE_EQ(recall, 0.5) << "50% overlap should yield 50% recall";
}

TEST_F(RecallEstimatorTest, UnorderedMatching) {
    // Test that order doesn't matter
    constexpr uint32_t num_queries = 10;
    constexpr uint32_t k = 5;

    std::vector<vertex_id_t> pred_data(num_queries * k);
    std::vector<vertex_id_t> gt_data(num_queries * k);

    for (uint32_t i = 0; i < num_queries; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            // Ground truth in ascending order
            gt_data[i * k + j] = i * k + j;
            // Predictions in reverse order
            pred_data[i * k + j] = i * k + (k - 1 - j);
        }
    }

    idlist_array_t predictions = create_idlist_array(pred_data, num_queries, k);
    idlist_array_t groundtruth = create_idlist_array(gt_data, num_queries, k);

    double recall = estimator_->calculate_recall_at_k(predictions, groundtruth);

    EXPECT_DOUBLE_EQ(recall, 1.0) << "Order should not affect recall";
}

TEST_F(RecallEstimatorTest, DuplicatePredictions) {
    // Test handling of duplicate predictions
    constexpr uint32_t num_queries = 10;
    constexpr uint32_t k = 10;

    std::vector<vertex_id_t> pred_data(num_queries * k);
    std::vector<vertex_id_t> gt_data(num_queries * k);

    for (uint32_t i = 0; i < num_queries; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            gt_data[i * k + j] = i * k + j;
            // All predictions are the same (duplicates)
            pred_data[i * k + j] = i * k;
        }
    }

    idlist_array_t predictions = create_idlist_array(pred_data, num_queries, k);
    idlist_array_t groundtruth = create_idlist_array(gt_data, num_queries, k);

    double recall = estimator_->calculate_recall_at_k(predictions, groundtruth);

    // Only one unique prediction matches, so recall = 1/k
    EXPECT_DOUBLE_EQ(recall, 0.1) << "Duplicates should be handled correctly";
}

TEST_F(RecallEstimatorTest, SingleQuery) {
    // Test with a single query
    constexpr uint32_t k = 5;

    std::vector<vertex_id_t> pred_data = {0, 1, 2, 3, 4};
    std::vector<vertex_id_t> gt_data = {0, 1, 2, 10, 11};

    idlist_array_t predictions = create_idlist_array(pred_data, 1, k);
    idlist_array_t groundtruth = create_idlist_array(gt_data, 1, k);

    double recall = estimator_->calculate_recall_at_k(predictions, groundtruth);

    // 3 out of 5 match
    EXPECT_DOUBLE_EQ(recall, 0.6) << "Single query recall should be correct";
}

TEST_F(RecallEstimatorTest, LargeK) {
    // Test with larger k value
    constexpr uint32_t num_queries = 50;
    constexpr uint32_t k = 100;

    std::vector<vertex_id_t> pred_data(num_queries * k);
    std::vector<vertex_id_t> gt_data(num_queries * k);

    for (uint32_t i = 0; i < num_queries; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            gt_data[i * k + j] = i * k + j;
            // 75% overlap
            if (j < k * 3 / 4) {
                pred_data[i * k + j] = i * k + j;
            } else {
                pred_data[i * k + j] = (i * k + j) + 10000;
            }
        }
    }

    idlist_array_t predictions = create_idlist_array(pred_data, num_queries, k);
    idlist_array_t groundtruth = create_idlist_array(gt_data, num_queries, k);

    double recall = estimator_->calculate_recall_at_k(predictions, groundtruth);

    EXPECT_DOUBLE_EQ(recall, 0.75) << "Large k should work correctly";
}

TEST_F(RecallEstimatorTest, MismatchedQueryCount) {
    // Test error handling for mismatched query counts
    constexpr uint32_t k = 10;

    std::vector<vertex_id_t> pred_data(50 * k);
    std::vector<vertex_id_t> gt_data(100 * k);

    for (size_t i = 0; i < pred_data.size(); ++i) {
        pred_data[i] = i;
    }
    for (size_t i = 0; i < gt_data.size(); ++i) {
        gt_data[i] = i;
    }

    idlist_array_t predictions = create_idlist_array(pred_data, 50, k);
    idlist_array_t groundtruth = create_idlist_array(gt_data, 100, k);

    // Should throw exception due to mismatched query count
    EXPECT_THROW(estimator_->calculate_recall_at_k(predictions, groundtruth), std::runtime_error);
}

TEST_F(RecallEstimatorTest, GroundTruthSmallerThanK) {
    // Test error handling when ground truth dimension < k
    constexpr uint32_t num_queries = 10;
    constexpr uint32_t k = 10;
    constexpr uint32_t gt_dim = 5;

    std::vector<vertex_id_t> pred_data(num_queries * k);
    std::vector<vertex_id_t> gt_data(num_queries * gt_dim);

    for (size_t i = 0; i < pred_data.size(); ++i) {
        pred_data[i] = i;
    }
    for (size_t i = 0; i < gt_data.size(); ++i) {
        gt_data[i] = i;
    }

    idlist_array_t predictions = create_idlist_array(pred_data, num_queries, k);
    idlist_array_t groundtruth = create_idlist_array(gt_data, num_queries, gt_dim);

    // Should throw exception due to GT dimension < k
    EXPECT_THROW(estimator_->calculate_recall_at_k(predictions, groundtruth), std::runtime_error);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
