/*
 * @FilePath: /Artea/include/artea/cpu/recall_estimator.hpp
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

#include <artea/definitions.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/logger.hpp>

/**
 * @brief Structure to hold recall evaluation results.
 */
struct RecallMetrics {
    double strict_recall; // Recall based purely on ID matching
    double soft_recall;   // Recall considering distance tolerance (epsilon)
};

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    typename dist_func_t,
    std::size_t inversed_epsilon = 1000000 // Default epsilon = 1e-6
>
class RecallEstimator final {

    using vertex_id_t = vertex_num_t;
    using distance_t = vec_ele_t;
    static constexpr distance_t epsilon = static_cast<distance_t>(1.0) / static_cast<distance_t>(inversed_epsilon);

public:

    /**
     * @brief Construct a new Recall Estimator.
     * @param dist_func The distance function used for verification.
     */
    explicit RecallEstimator(const dist_func_t& dist_func)
        : _dist_func(dist_func) {}

    /**
     * @brief Calculate Recall@1 comparing predictions against ground truth using TBB parallelism.
     *
     * @param predictions List of predicted nearest neighbor IDs for each query.
     * @param gt_vecs Ground truth vector array (stores IDs).
     * @param query_vecs Original query vector data (needed for distance check).
     * @param base_vecs Original base vector data (needed for distance check).
     * @return RecallMetrics Containing both strict and soft recall scores.
     */
    auto calculate_recall_at_1(
        const std::vector<vertex_id_t>& predictions,
        const VectorArray<vertex_num_t, vertex_id_t>* gt_vecs,
        const VectorArray<vertex_num_t, vec_ele_t>* query_vecs,
        const VectorArray<vertex_num_t, vec_ele_t>* base_vecs
    ) const -> RecallMetrics {

        std::size_t num_queries = predictions.size();

        // Helper struct for TBB reduction
        struct CorrectCounts {
            std::size_t strict = 0;
            std::size_t soft = 0;
        };

        // Parallel Reduction
        CorrectCounts total_counts = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<std::size_t>(0, num_queries),
            // Identity
            CorrectCounts {},
            // Lambda processor: Process a chunk of queries
            [&](const tbb::blocked_range<std::size_t>& r, CorrectCounts local_counts) -> CorrectCounts {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const vertex_id_t* gt_row = gt_vecs->get(i);
                    vertex_id_t gt_id = gt_row[0];
                    vertex_id_t pred_id = predictions[i];

                    /* Strict ID matching */
                    if (pred_id == gt_id) {
                        local_counts.strict++;
                        local_counts.soft++;
                    } else {
                        /* Distance-based tolerance check (Soft Match) */
                        const distance_t* q_vec = query_vecs->get(i);
                        const distance_t* gt_vec_data = base_vecs->get(gt_id);
                        const distance_t* pred_vec_data = base_vecs->get(pred_id);

                        /* Re-calculate distances */
                        distance_t dist_gt = _dist_func(q_vec, gt_vec_data);
                        distance_t dist_pred = _dist_func(q_vec, pred_vec_data);

                        /* Check if prediction is within epsilon tolerance of GT */
                        if (dist_pred <= dist_gt + epsilon) {
                            local_counts.soft++;
                        }
                    }
                }
                return local_counts;
            },
            // Lambda reducer: Join results from threads
            [](CorrectCounts a, CorrectCounts b) -> CorrectCounts {
                return CorrectCounts{ a.strict + b.strict, a.soft + b.soft };
            }
        );

        double score_strict = static_cast<double>(total_counts.strict) / num_queries;
        double score_soft = static_cast<double>(total_counts.soft) / num_queries;

        // Log results here for convenience
        logger.info(fmt::format("Evaluation (Threshold: {:.1e}):", epsilon));
        logger.info(fmt::format("   -> Strict Recall@1: {:.2f}%", score_strict * 100.0));
        logger.info(fmt::format("   -> Soft Recall@1:   {:.2f}%", score_soft * 100.0));

        return RecallMetrics { score_strict, score_soft };
    }

private:
    const dist_func_t& _dist_func;

};  // class RecallEstimator

}   // namespace cpu
}   // namespace artea