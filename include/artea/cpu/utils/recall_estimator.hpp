/*
 * @FilePath: /Artea/include/artea/cpu/utils/recall_estimator.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: TBB-accelerated Recall Estimator with soft validation.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cmath>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <fmt/format.h>
#include <artea/common/logger.hpp>

/**
 * @brief Structure to hold recall evaluation results (Used for Recall@1).
 */
struct RecallMetrics {
    double strict_recall; // Recall based purely on ID matching
    double soft_recall;   // Recall considering distance tolerance (epsilon)
};

namespace artea {
namespace cpu {

template <typename ComputerTraitsT>
class RecallEstimator final {

    using vertex_id_t = typename ComputerTraitsT::vertex_id_t;
    using vertex_num_t = typename ComputerTraitsT::vertex_num_t;
    using vec_ele_t = typename ComputerTraitsT::vec_ele_t;
    using distance_t = typename ComputerTraitsT::distance_t;
    using dist_func_t = typename ComputerTraitsT::dist_func_t;
    using vector_array_t = typename ComputerTraitsT::vector_array_t;
    using base_vecs_t = typename ComputerTraitsT::base_vecs_t;
    using query_vecs_t = typename ComputerTraitsT::query_vecs_t;
    using ground_truth_t = typename ComputerTraitsT::ground_truth_t;
    using idlist_array_t = typename ComputerTraitsT::idlist_array_t;
    static constexpr std::size_t inversed_epsilon = 1000000; // Default epsilon = 1e-6
    static constexpr distance_t epsilon = static_cast<distance_t>(1.0) / static_cast<distance_t>(inversed_epsilon);

public:

    /**
     * @brief Construct a new Recall Estimator.
     * @param dist_func The distance function used for verification.
     */
    explicit RecallEstimator(const dist_func_t& dist_func)
        : _dist_func(dist_func) {}

    /**
     * @brief Calculate Recall@K comparing predictions against ground truth using TBB parallelism.
     *        This method includes both strict (ID match) and soft (distance tolerance) recall.
     *
     * @param predictions VectorArray of predicted IDs (num_queries vectors, each with dimension k).
     *                    Each vector contains the top-k predictions for one query.
     * @param gt_vecs Ground truth vector array (stores IDs).
     * @param query_vecs Original query vector data (needed for distance check).
     * @param base_vecs Original base vector data (needed for distance check).
     * @return RecallMetrics Containing both strict and soft recall scores.
     */
    auto calculate_recall_at_k(
        const idlist_array_t& predictions,
        const ground_truth_t& gt_vecs,
        const query_vecs_t& query_vecs,
        const base_vecs_t& base_vecs
    ) const -> RecallMetrics {
        std::size_t num_queries = gt_vecs.get_num_vecs();
        std::size_t k = predictions.get_vec_dim();

        // Assertion: Check predictions size
        if (predictions.get_num_vecs() != num_queries) {
            throw std::invalid_argument(fmt::format(
                "Predictions size mismatch: expected {} queries, got {}",
                num_queries, predictions.get_num_vecs()
            ));
        }

        // Helper struct for TBB reduction
        struct CorrectCounts {
            std::size_t strict = 0;
            std::size_t soft = 0;
        };

        // Parallel Reduction
        CorrectCounts total_counts = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, num_queries),
            CorrectCounts {},
            [&](const tbb::blocked_range<std::size_t>& r, CorrectCounts local_counts) -> CorrectCounts {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const vertex_id_t gt_id = gt_vecs.get(i)[0]; // Nearest GT
                    const vertex_id_t* pred_vec = predictions.get(i);
                    const vec_ele_t* q_vec = query_vecs.get(i);
                    const vec_ele_t* gt_vec_data = base_vecs.get(gt_id);
                    distance_t dist_gt = _dist_func(q_vec, gt_vec_data);

                    bool strict_found = false;
                    bool soft_found = false;

                    // Iterate through top-K predictions
                    for (std::size_t j = 0; j < k; ++j) {
                        vertex_id_t pred_id = pred_vec[j];

                        // Strict ID matching
                        if (pred_id == gt_id) {
                            strict_found = true;
                            soft_found = true;
                            break;
                        }

                        // Soft distance-based tolerance check
                        if (!soft_found) {
                            const vec_ele_t* pred_vec_data = base_vecs.get(pred_id);
                            distance_t dist_pred = _dist_func(q_vec, pred_vec_data);

                            if (dist_pred <= dist_gt * (static_cast<distance_t>(1.0) + epsilon)) {
                                soft_found = true;
                            }
                        }
                    }

                    if (strict_found) local_counts.strict++;
                    if (soft_found) local_counts.soft++;
                }
                return local_counts;
            },
            [](CorrectCounts a, CorrectCounts b) -> CorrectCounts {
                return CorrectCounts{ a.strict + b.strict, a.soft + b.soft };
            }
        );

        double score_strict = static_cast<double>(total_counts.strict) / num_queries;
        double score_soft = static_cast<double>(total_counts.soft) / num_queries;

        logger.info(fmt::format("Evaluation (Recall@{}, Threshold: {:.1e}):", k, epsilon));
        logger.info(fmt::format("   -> Strict Recall@{}: {:.2f}%", k, score_strict * 100.0));
        logger.info(fmt::format("   -> Soft Recall@{}:   {:.2f}%", k, score_soft * 100.0));

        return RecallMetrics { score_strict, score_soft };
    }

private:
    const dist_func_t& _dist_func;

};  // class RecallEstimator

}   // namespace cpu
}   // namespace artea