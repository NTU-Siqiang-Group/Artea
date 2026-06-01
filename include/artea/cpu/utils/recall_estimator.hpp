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
#include <artea/cpu/router/data_structures/result_entry_concept.hpp>

namespace artea {
namespace cpu {

template <typename ComputerTraitsT>
class RecallEstimator final {

    using vertex_id_t = typename ComputerTraitsT::vertex_id_t;
    using vertex_num_t = typename ComputerTraitsT::vertex_num_t;
    using idlist_array_t = typename ComputerTraitsT::idlist_array_t;
    // Pulled in for Soft Recall@1 (distance-based hit test); mirror the
    // type surface ADREstimator consumes from the same traits.
    using vec_ele_t      = typename ComputerTraitsT::vec_ele_t;
    using distance_t     = typename ComputerTraitsT::distance_t;
    using vector_array_t = typename ComputerTraitsT::vector_array_t;
    using dist_func_t    = typename ComputerTraitsT::dist_func_t;

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
            ARTEA_ERROR(fmt::format(
                "Predictions size mismatch: expected {} queries, got {}",
                num_queries, predictions.get_num_vecs()
            ));
            return 0.0;
        }

        // Assertion: Check ground truth has at least k elements
        if (gt_vecs.get_vec_dim() < k) {
            ARTEA_ERROR(fmt::format(
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

    /**
     * @brief Calculate Recall@K from a flat knn_results_t array.
     *
     * @param predictions Flat array of num_queries * topk result entries in row-major order.
     * @param gt_vecs Ground truth vector array (stores IDs).
     * @param topk Number of nearest neighbors per query.
     * @param num_queries Number of queries.
     * @return double Recall@K score.
     */
    template <ResultEntry ResultEntryT>
    auto calculate_recall_at_k(
        const std::vector<ResultEntryT>& predictions,
        const idlist_array_t& gt_vecs,
        std::size_t topk,
        std::size_t num_queries
    ) const -> double {
        const std::size_t k = topk;

        if (predictions.size() != num_queries * k) {
            ARTEA_ERROR(fmt::format(
                "Predictions size mismatch: expected {}*{}={}, got {}",
                num_queries, k, num_queries * k, predictions.size()
            ));
            return 0.0;
        }

        if (gt_vecs.get_vec_dim() < k) {
            ARTEA_ERROR(fmt::format(
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
                intersect_cache.resize(k);
            }
        };

        tbb::enumerable_thread_specific<ThreadLocalBuffers> thread_buffers([k]() {
            return ThreadLocalBuffers(k);
        });

        std::size_t total_matches = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, num_queries),
            std::size_t(0),
            [&](const tbb::blocked_range<std::size_t>& r, std::size_t local_matches) -> std::size_t {
                auto& buffers = thread_buffers.local();

                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const vertex_id_t* gt_vec = gt_vecs.get(i);
                    const ResultEntryT* pred_row = predictions.data() + i * k;

                    buffers.sorted_gt.assign(gt_vec, gt_vec + k);
                    std::sort(buffers.sorted_gt.begin(), buffers.sorted_gt.end());

                    buffers.sorted_pred.resize(k);
                    for (std::size_t j = 0; j < k; ++j) {
                        buffers.sorted_pred[j] = pred_row[j].get_vid();
                    }
                    std::sort(buffers.sorted_pred.begin(), buffers.sorted_pred.end());

                    auto pred_end = std::unique(buffers.sorted_pred.begin(), buffers.sorted_pred.end());

                    auto intersect_end = std::set_intersection(
                        buffers.sorted_gt.begin(), buffers.sorted_gt.end(),
                        buffers.sorted_pred.begin(), pred_end,
                        buffers.intersect_cache.begin()
                    );

                    local_matches += std::distance(buffers.intersect_cache.begin(), intersect_end);
                }
                return local_matches;
            },
            [](std::size_t a, std::size_t b) -> std::size_t {
                return a + b;
            }
        );

        return static_cast<double>(total_matches) / (num_queries * k);
    }

    /**
     * @brief Soft Recall@1 — credit an equidistant retrieval as a hit.
     *
     * Standard Recall@1 compares the retrieved top-1 vertex *id* against
     * the ground-truth id (gt column 0). On datasets with duplicate / tied
     * base vectors (e.g. tiny5m, where ~5.6% of queries have >=2 base
     * vectors at the true-NN distance, one cluster reaching 1814 copies)
     * this under-counts: the router may return a DIFFERENT id that is
     * nonetheless an exact equidistant neighbour, while the GT — produced
     * by a non-stable argsort — stored one arbitrary member of the tie
     * set. Such a result is a correct 1-NN by distance but a miss by id,
     * so hard Recall@1 plateaus below 1.0 no matter the search budget.
     *
     * Soft Recall@1 fixes the *measurement*, not the retrieval: a query is
     * a hit when the retrieved top-1 sits at the same distance as the true
     * NN, within a small relative tolerance. Since gt[0] is the global
     * minimum, d_retrieved >= d_true always, so the test reduces to
     * d_retrieved <= d_true * (1 + tol). Distances are recomputed via
     * @p dist_func for self-consistency, exactly as ADREstimator does.
     *
     * Defined for the top-1 entry only (caller must gate on topk == 1).
     * Unlike ADR, queries with d_true == 0 are KEPT: a duplicate exact
     * match still counts as a hit because d_retrieved == 0 <= 0. The
     * divisor is therefore the full query count.
     *
     * @tparam ResultEntryT  Anything satisfying @c ResultEntry (exposes
     *                       @c get_vid()); @c CandidateEntry qualifies.
     * @param predictions    Flat array sized @p num_queries * @p topk,
     *                       indexed as @c predictions[i * topk] for the
     *                       top-1 entry of query i.
     * @return Soft Recall@1 in [0, 1].
     */
    template <ResultEntry ResultEntryT>
    auto calculate_soft_recall_at_1(
        const std::vector<ResultEntryT>& predictions,
        std::size_t                      topk,
        const idlist_array_t&            gt_vecs,
        const vector_array_t&            query_vecs,
        const vector_array_t&            base_vecs,
        const dist_func_t&               dist_func,
        std::size_t                      num_queries
    ) const -> double {
        if (num_queries == 0 || topk == 0) return 0.0;

        if (predictions.size() != num_queries * topk) {
            ARTEA_ERROR(fmt::format(
                "Predictions size mismatch: expected {}*{}={}, got {}",
                num_queries, topk, num_queries * topk, predictions.size()));
            return 0.0;
        }
        if (gt_vecs.get_num_vecs() < num_queries || gt_vecs.get_vec_dim() < 1) {
            ARTEA_ERROR(fmt::format(
                "Ground truth too small for Soft Recall@1: {} rows x {} dim, "
                "need >= {} rows x 1 dim",
                gt_vecs.get_num_vecs(), gt_vecs.get_vec_dim(), num_queries));
            return 0.0;
        }

        // Relative slack on (squared-)L2 so a genuine geometric tie that
        // differs by a few ULPs still counts; bit-identical duplicates
        // match exactly regardless. When d_true == 0 the threshold is 0,
        // so only an exact d_retrieved == 0 match is credited.
        constexpr double kRelTol = 1e-5;

        const std::size_t hits = tbb::parallel_reduce(
            tbb::blocked_range<std::size_t>(0, num_queries),
            std::size_t(0),
            [&](const tbb::blocked_range<std::size_t>& r, std::size_t local) -> std::size_t {
                for (std::size_t i = r.begin(); i != r.end(); ++i) {
                    const vertex_id_t retrieved = predictions[i * topk].get_vid();
                    const vertex_id_t true_nn   = gt_vecs.get(i)[0];
                    const vec_ele_t*  q_vec     = query_vecs.get(i);
                    const distance_t  d_retrieved = dist_func(q_vec, base_vecs.get(retrieved));
                    const distance_t  d_true      = dist_func(q_vec, base_vecs.get(true_nn));
                    const double thresh = static_cast<double>(d_true) * (1.0 + kRelTol);
                    if (static_cast<double>(d_retrieved) <= thresh) ++local;
                }
                return local;
            },
            [](std::size_t a, std::size_t b) -> std::size_t { return a + b; }
        );

        return static_cast<double>(hits) / static_cast<double>(num_queries);
    }

};  // class RecallEstimator

}   // namespace cpu
}   // namespace artea
