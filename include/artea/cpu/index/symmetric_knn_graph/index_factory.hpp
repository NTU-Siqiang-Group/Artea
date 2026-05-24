/*
 * @FilePath: /Artea/include/artea/cpu/index/symmetric_knn_graph/index_factory.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Symmetric KNN graph factory. Builds a KNN graph then adds
 *               reverse edges (without truncation) to make it symmetric.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <chrono>
#include <functional>
#include <fmt/format.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/spin_mutex.h>

#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {
namespace symmetric_knn_graph {

template <typename GraphFactoryTraitsT>
class IndexFactory {

    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vec_ele_t = typename GraphFactoryTraitsT::vec_ele_t;
    using iter_t = typename GraphFactoryTraitsT::iter_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using vector_dataset_t = typename GraphFactoryTraitsT::vector_dataset_t;
    // dist_func type is per-method template arg (DistFuncT); deduced from caller.
    using layer_config_t = typename GraphFactoryTraitsT::layer_config_t;
    using this_index_t = typename GraphFactoryTraitsT::symmetric_knn_graph::index_t;
    using propagate_config_t = typename GraphFactoryTraitsT::symmetric_knn_graph::propagate_config_t;
    template <typename DistFuncT> using random_eg_t        = typename GraphFactoryTraitsT::template random_eg_t<DistFuncT>;
    template <typename DistFuncT> using propagate_engine_t = typename GraphFactoryTraitsT::template propagate_engine_t<DistFuncT>;
    template <typename DistFuncT> using triangle_updater_t = typename GraphFactoryTraitsT::template triangle_updater_t<DistFuncT>;
    template <typename DistFuncT> using reverse_updater_t  = typename GraphFactoryTraitsT::template reverse_updater_t<DistFuncT>;
    template <typename DistFuncT> using routing_updater_t  = typename GraphFactoryTraitsT::template routing_updater_t<DistFuncT>;
    template <typename DistFuncT> using truncate_updater_t = typename GraphFactoryTraitsT::template truncate_updater_t<DistFuncT>;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using query_vecs_t = typename GraphFactoryTraitsT::query_vecs_t;
    using ground_truth_t = typename GraphFactoryTraitsT::ground_truth_t;
    using recall_estimator_t = typename GraphFactoryTraitsT::recall_estimator_t;
    template <typename DistFuncT> using single_layer_router_t = typename GraphFactoryTraitsT::template single_layer_router_t<DistFuncT>;
    using knn_graph = typename GraphFactoryTraitsT::knn_graph;

public:
    /** @brief Wall-clock breakdown returned by @ref construct_graph.
     *         symmetric_knn_graph is a single-layer flat graph with no
     *         upper / bottom split, so we report end-to-end wall-clock
     *         only (covers both the build loop and the symmetrizing
     *         reverse-edge pass). */
    struct ConstructResult {
        this_index_t graph;
        double total_time_ms = 0.0;
    };

    /** @brief Construct a symmetric KNN graph from vector array. */
    template <typename DistFuncT>
    static auto construct_graph(
        const vector_array_t&    base_vecs,
        const layer_config_t     layer_config,
        const propagate_config_t propagate_config,
        const DistFuncT&         dist_func
    ) -> ConstructResult {
        const auto t_start = std::chrono::high_resolution_clock::now();
        this_index_t graph_index(base_vecs, layer_config, propagate_config);
        _build_loop(graph_index, dist_func, propagate_config);
        const auto t_end = std::chrono::high_resolution_clock::now();
        return ConstructResult{
            std::move(graph_index),
            std::chrono::duration<double, std::milli>(t_end - t_start).count()
        };
    }

    /**
     * @brief Construct a symmetric KNN graph from an existing knn_graph by
     *        taking ownership of its edges, then adding reverse edges.
     *
     * @warning This function moves from the input knn_graph. After the call,
     *          the input is left in a valid but unspecified state.
     *
     * @param knn_graph_index  The knn_graph whose edges will be consumed (moved).
     * @return A symmetric KNN graph with reverse edges added.
     */
    template <typename DistFuncT>
    static auto construct_graph(
        typename knn_graph::index_t&& knn_graph_index,
        const DistFuncT&              dist_func
    ) -> ConstructResult {
        const auto t_start = std::chrono::high_resolution_clock::now();
        this_index_t graph_index(std::move(knn_graph_index));

        const vertex_num_t num_vertices = graph_index.get_num_vertices();

        propagate_engine_t<DistFuncT> propagate_engine(dist_func);
        propagate_engine.set_graph(graph_index.get_refining_graph());

        auto reverse_updater = propagate_engine.template make_updater<reverse_updater_t<DistFuncT>>();

        propagate_engine.next(reverse_updater);

        const auto t_end = std::chrono::high_resolution_clock::now();
        return ConstructResult{
            std::move(graph_index),
            std::chrono::duration<double, std::milli>(t_end - t_start).count()
        };
    }

    /** @brief Construct with per-build-loop recall/throughput profiling. */
    template <typename DistFuncT>
    static auto profile_graph_quality(
        const vector_dataset_t&  dataset,
        const layer_config_t     layer_config,
        const propagate_config_t propagate_config,
        const DistFuncT&         dist_func
    ) -> void {
        const vector_array_t& base_vecs = dataset.get_base_vecs();
        const query_vecs_t& query_vecs = dataset.get_query_vecs();
        const ground_truth_t& groundtruth = dataset.get_gt_vecs();

        this_index_t graph_index(base_vecs, layer_config, propagate_config);

        recall_estimator_t recall_estimator;
        const vertex_num_t topk = 20;
        const vertex_num_t candidate_queue_size = 40;
        single_layer_router_t<DistFuncT> router(
            base_vecs, dist_func, topk, candidate_queue_size);
        router.initialize();

        _build_loop(graph_index, dist_func, propagate_config,
            [&](iter_t build_loop) {
                auto t0 = std::chrono::high_resolution_clock::now();
                auto results = router.batch_query(query_vecs, graph_index.get_refining_graph());
                auto t1 = std::chrono::high_resolution_clock::now();
                double qps = query_vecs.get_num_vecs() * 1e6 /
                    std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
                double recall = recall_estimator.calculate_recall_at_k(
                    results, groundtruth, topk, query_vecs.get_num_vecs());
                ARTEA_INFO(fmt::format(
                    "BuildLoop {}: Recall@{}={:.4f}, QPS={:.2f}, Candidate={}",
                    build_loop, topk, recall, qps, candidate_queue_size
                ));
            }
        );
    }

private:

    /**
     * @brief Core build loop: runs the KNN graph build schedule, then adds
     *        reverse edges without truncation to make it symmetric.
     */
    template <typename DistFuncT>
    static auto _build_loop(
        this_index_t& graph_index,
        const DistFuncT& dist_func,
        const propagate_config_t& propagate_config,
        std::function<void(iter_t)> on_iter_end = nullptr
    ) -> void {
        const vertex_num_t num_vertices = graph_index.get_num_vertices();
        const vertex_num_t max_nbr_size = graph_index.layer_config().max_nbr_size();
        const vertex_num_t init_nbr_size = static_cast<vertex_num_t>(max_nbr_size * propagate_config.prefill_ratio());

        /** -------------------- Optimazation ------------------------------------- ***/
        /** @brief A sparse graph is effecient enough to search nearest neighbors     */
        graph_index.layer_config().max_nbr_size(max_nbr_size / 2);
        /** ----------------------------------------------------------------------- ***/

        random_eg_t<DistFuncT> random_eg(dist_func);
        random_eg.generate(graph_index.get_refining_graph(), init_nbr_size);

        propagate_engine_t<DistFuncT> propagate_engine(dist_func);
        propagate_engine.set_graph(graph_index.get_refining_graph());

        // symmetric_knn_graph dropped PruningConfig in a prior commit;
        // pass plain RNG coefficients (scale=1, shift=0) to preserve the
        // ori_dist threshold TriangleUpdater used before gaining params.
        auto triangle_updater  = propagate_engine.template make_updater<triangle_updater_t<DistFuncT>>(
            ratio_t{1}, ratio_t{0});
        auto reverse_updater   = propagate_engine.template make_updater<reverse_updater_t<DistFuncT>>();
        const vertex_num_t routing_topk = propagate_config.resolve_routing_topk(max_nbr_size);
        const vertex_num_t routing_queue_size = propagate_config.resolve_routing_queue_size(max_nbr_size);
        auto routing_updater   = propagate_engine.template make_updater<routing_updater_t<DistFuncT>>(routing_topk, routing_queue_size);
        auto truncate_updater  = propagate_engine.template make_updater<truncate_updater_t<DistFuncT>>();

        for (iter_t build_loop = 0; build_loop < propagate_config.num_build_loops(); ++build_loop) {
            propagate_engine.run(propagate_config.num_triu_iters(), triangle_updater)
                            .next(reverse_updater).next(truncate_updater);
            if (on_iter_end) { on_iter_end(build_loop); }
        }

        /** -------------------- Optimazation --------------------------------------- ***/
        /** @brief Reconstructed as dense graph with original edge number requirements  */
        graph_index.layer_config().max_nbr_size(max_nbr_size);
        /** ------------------------------------------------------------------------- ***/

        for (iter_t routing_loop = 0; routing_loop < propagate_config.num_routing_loops(); ++routing_loop) {
            propagate_engine.next(routing_updater).next(truncate_updater);
            if (on_iter_end) { on_iter_end(propagate_config.num_build_loops() + routing_loop); }
        }

        /** @brief Add reverse edges to make the graph symmetric (no truncation). */
        propagate_engine.next(reverse_updater);
    }

};  // class IndexFactory

}   // namespace symmetric_knn_graph
}   // namespace cpu
}   // namespace artea
