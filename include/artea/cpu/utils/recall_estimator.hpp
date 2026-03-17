/*
 * @FilePath: /Artea/include/artea/cpu/utils/recall_estimator.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: TBB-accelerated Recall Estimator with zero-overhead design.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <fmt/format.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT>
class RecallEstimator final {

    using vertex_id_t = typename ComputerTraitsT::vertex_id_t;
    using vertex_num_t = typename ComputerTraitsT::vertex_num_t;
    using idlist_array_t = typename ComputerTraitsT::idlist_array_t;

public:

    /**
     * @brief Construct a new Recall Estimator.
     */
    RecallEstimator() = default;

    /**
     * @brief Calculate Recall@K comparing predictions against ground truth using TBB parallelism.
     *
     * Recall@K = (|predicted_top_K ∩ groundtruth_top_K|) / K
     *
     * Performance optimizations:
     * - Thread-local buffers pre-allocated based on K (no runtime reallocation)
     * - Direct memory write via resize() instead of back_inserter (zero branch overhead)
     * - assign() for idiomatic memory copy (enables memmove optimization)
     * - Avoids unnecessary erase() by passing logical end iterator to set_intersection
     *
     * @param predictions VectorArray of predicted IDs (num_queries vectors, each with dimension k).
     *                    Each vector contains the top-k predictions for one query.
     * @param gt_vecs Ground truth vector array (stores IDs).
     * @return double Recall@K score.
     */
    auto calculate_recall_at_k(
        const idlist_array_t& predictions,
        const idlist_array_t& gt_vecs
    ) const -> double {
        std::size_t num_queries = gt_vecs.get_num_vecs();
        std::size_t k = predictions.get_vec_dim();

        // Assertion: Check predictions size
        if (predictions.get_num_vecs() != num_queries) {
            logger.error(fmt::format(
                "Predictions size mismatch: expected {} queries, got {}",
                num_queries, predictions.get_num_vecs()
            ));
            return 0.0;
        }

        // Assertion: Check ground truth has at least k elements
        if (gt_vecs.get_vec_dim() < k) {
            logger.error(fmt::format(
                "Ground truth dimension {} is less than k={}",
                gt_vecs.get_vec_dim(), k
            ));
            return 0.0;
        }

        // Thread-local buffers pre-allocated based on K
        struct ThreadLocalBuffers {
            std::vector<vertex_id_t> sorted_gt;
            std::vector<vertex_id_t> sorted_pred;
            std::vector<vertex_id_t> intersect_cache;

            explicit ThreadLocalBuffers(std::size_t k) {
                sorted_gt.reserve(k);
                sorted_pred.reserve(k);
                intersect_cache.resize(k);  // Pre-allocate for direct write
            }
        };

        tbb::enumerable_thread_specific<ThreadLocalBuffers> thread_buffers([k]() {
            return ThreadLocalBuffers(k);
        });

        // Parallel Reduction to count total matches
        std::size_t total_matches = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, num_queries),
            std::size_t(0),
            [&](const tbb::blocked_range<std::size_t>& r, std::size_t local_matches) -> std::size_t {
                auto& buffers = thread_buffers.local();

                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const vertex_id_t* gt_vec = gt_vecs.get(i);
                    const vertex_id_t* pred_vec = predictions.get(i);

                    // Prepare sorted ground truth IDs using assign (enables memmove optimization)
                    buffers.sorted_gt.assign(gt_vec, gt_vec + k);
                    std::sort(buffers.sorted_gt.begin(), buffers.sorted_gt.end());

                    // Prepare sorted prediction IDs using assign
                    buffers.sorted_pred.assign(pred_vec, pred_vec + k);
                    std::sort(buffers.sorted_pred.begin(), buffers.sorted_pred.end());

                    // Remove duplicates from predictions (prevents inflated recall)
                    auto pred_end = std::unique(buffers.sorted_pred.begin(), buffers.sorted_pred.end());

                    // Compute intersection using direct memory write (zero branch overhead)
                    auto intersect_end = std::set_intersection(
                        buffers.sorted_gt.begin(), buffers.sorted_gt.end(),
                        buffers.sorted_pred.begin(), pred_end,
                        buffers.intersect_cache.begin()
                    );

                    // Count intersection size
                    local_matches += std::distance(buffers.intersect_cache.begin(), intersect_end);
                }
                return local_matches;
            },
            [](std::size_t a, std::size_t b) -> std::size_t {
                return a + b;
            }
        );

        // Recall@K = total_matches / (num_queries * k)
        double recall = static_cast<double>(total_matches) / (num_queries * k);

        return recall;
    }

};  // class RecallEstimator

}   // namespace cpu
}   // namespace artea
