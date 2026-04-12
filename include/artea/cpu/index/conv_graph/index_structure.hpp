/*
 * @FilePath: /Artea/include/artea/cpu/index/conv_graph/index_structure.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Convergent graph index structure holding a BottomGraph by
 *               composition via @c std::unique_ptr. Exposes the inner graph
 *               via @c get_bottom_graph so callers can transfer or reuse
 *               it, and forwards the common @c BottomGraph public methods
 *               so existing template call sites (propagate engine, updaters,
 *               routers, file manager, compactor) keep working unchanged.
 */

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace artea {
namespace cpu {
namespace conv_graph {

/**
 * @brief Convergent graph index. Composes a @c BottomGraph plus the
 *        algorithm-specific pruning and propagation configs.
 *
 * This class used to inherit from @c BottomGraph via CRTP. It now holds
 * the graph through a @c std::unique_ptr, which:
 *   - makes move-assigning @c IndexStructure safe (the legacy @c BottomGraph
 *     move-assign operator leaves its @c _vecs_data reference stale because
 *     a reference cannot be reseated — going through @c unique_ptr
 *     sidesteps that by swapping the whole object);
 *   - gives the containing class a clean ownership story;
 *   - lets callers transfer ownership of the inner graph to another index
 *     type without touching internal fields;
 *   - brings the holding style in line with
 *     @c stacked_rgraph::IndexStructure (which is forced to use
 *     @c unique_ptr anyway because @c HierarchicalGraph is move-deleted).
 *
 * External consumers that previously relied on inherited methods see the
 * same interface via the forwarders below.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class IndexStructure {

    using bottom_graph_t    = typename IndexTraitsT::template bottom_graph_t<IndexStructure<IndexTraitsT>>;
    using vertex_num_t       = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t        = typename IndexTraitsT::vertex_id_t;
    using bnbr_arr_t         = typename IndexTraitsT::bnbr_arr_t;
    using vector_array_t     = typename IndexTraitsT::vector_array_t;
    using layer_config_t     = typename IndexTraitsT::layer_config_t;
    using propagate_config_t = typename IndexTraitsT::conv_graph::propagate_config_t;
    using pruning_config_t   = typename IndexTraitsT::conv_graph::pruning_config_t;

public:
    /**
     * @brief Construct a new convergent graph index.
     * @param vecs_data        Reference to the vector data for this layer.
     * @param layer_config     Layer configuration (max_nbr_size and reserved_nbr_size).
     * @param pruning_config   Pruning configuration (scale_coeffs and shifted_coeffs).
     * @param propagate_config Propagation configuration (num_build_loops, num_triu_iters, prefill_ratio).
     */
    IndexStructure(
        const vector_array_t& vecs_data,
        const layer_config_t layer_config,
        const pruning_config_t pruning_config,
        const propagate_config_t propagate_config
    ) : _bottom_graph(std::make_unique<bottom_graph_t>(vecs_data, layer_config)),
        _pruning_config(pruning_config),
        _propagate_config(propagate_config)
    {}

    IndexStructure(const IndexStructure&) = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;

    // Move is O(1): the @c unique_ptr swap avoids touching any of
    // @c BottomGraph's internal fields (including the @c _vecs_data
    // reference, which the legacy move-assign had to leave stale).
    IndexStructure(IndexStructure&&) noexcept = default;
    IndexStructure& operator=(IndexStructure&&) noexcept = default;

    // --- Composed graph accessor ---

    /**
     * @brief Access the underlying @c BottomGraph instance. Use this to
     *        pass the graph directly to utilities that only need the
     *        graph core, or to read it for inspection.
     *
     * The returned reference is valid as long as this @c IndexStructure
     * holds the graph — i.e. until the next move-assign or destruction.
     */
    __attribute__((always_inline))
    auto get_bottom_graph() -> bottom_graph_t& { return *_bottom_graph; }

    __attribute__((always_inline))
    auto get_bottom_graph() const -> const bottom_graph_t& { return *_bottom_graph; }

    // --- BottomGraph public API forwarders ---
    //
    // These keep call sites that previously relied on inheritance working
    // unchanged. Every forwarder is a thin inline call to the matching
    // method on @c *_bottom_graph.

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _bottom_graph->get_num_vertices();
    }

    __attribute__((always_inline))
    auto layer_config() const -> const layer_config_t& {
        return _bottom_graph->layer_config();
    }

    __attribute__((always_inline))
    auto layer_config() -> layer_config_t& {
        return _bottom_graph->layer_config();
    }

    __attribute__((always_inline))
    auto get_nbrs_arr() -> std::vector<bnbr_arr_t>& {
        return _bottom_graph->get_nbrs_arr();
    }

    __attribute__((always_inline))
    auto get_nbrs_arr() const -> const std::vector<bnbr_arr_t>& {
        return _bottom_graph->get_nbrs_arr();
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) const -> const bnbr_arr_t& {
        return _bottom_graph->fetch_nbrs(src);
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) -> bnbr_arr_t& {
        return _bottom_graph->fetch_nbrs(src);
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return _bottom_graph->get_vecs_data();
    }

    auto get_base_metadata() const -> nlohmann::json {
        return _bottom_graph->get_base_metadata();
    }

    // --- Config accessors ---

    __attribute__((always_inline))
    auto pruning_config() const -> const pruning_config_t& { return _pruning_config; }

    __attribute__((always_inline))
    auto pruning_config() -> pruning_config_t& { return _pruning_config; }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& { return _propagate_config; }

    __attribute__((always_inline))
    auto propagate_config() -> propagate_config_t& { return _propagate_config; }

    // --- Metadata hooks ---

    auto get_metadata() const -> nlohmann::json {
        nlohmann::json meta;
        meta["pruning_config"] = {
            {"scale_coeffs", _pruning_config.scale_coeffs()},
            {"shifted_coeffs", _pruning_config.shifted_coeffs()}
        };
        meta["propagate_config"] = {
            {"num_build_loops", _propagate_config.num_build_loops()},
            {"num_triu_iters", _propagate_config.num_triu_iters()},
            {"prefill_ratio", _propagate_config.prefill_ratio()},
            {"num_routing_loops", _propagate_config.num_routing_loops()},
            {"routing_topk", _propagate_config.routing_topk()},
            {"routing_queue_size", _propagate_config.routing_queue_size()}
        };
        return meta;
    }

    static auto from_metadata(
        const nlohmann::json& meta,
        const vector_array_t& vecs_data,
        const layer_config_t& layer_config
    ) -> IndexStructure {
        pruning_config_t pruning_config(
            meta["pruning_config"]["scale_coeffs"].get<float>(),
            meta["pruning_config"]["shifted_coeffs"].get<float>()
        );
        propagate_config_t propagate_config(
            meta["propagate_config"]["num_build_loops"].get<uint32_t>(),
            meta["propagate_config"]["num_triu_iters"].get<uint32_t>(),
            meta["propagate_config"]["prefill_ratio"].get<float>(),
            meta["propagate_config"]["num_routing_loops"].get<uint32_t>(),
            meta["propagate_config"].value("routing_topk", uint32_t(64)),
            meta["propagate_config"].value("routing_queue_size", uint32_t(96))
        );
        return IndexStructure(vecs_data, layer_config, pruning_config, propagate_config);
    }

private:
    /**
     * @brief Composed @c BottomGraph — owns the graph topology and
     *        layer config. Held through @c unique_ptr so @c IndexStructure
     *        move operations are O(1) pointer swaps and the legacy
     *        @c BottomGraph move-assign (which cannot reseat its
     *        @c _vecs_data reference) is never invoked.
     */
    std::unique_ptr<bottom_graph_t> _bottom_graph;

    /** @brief Pruning configuration. */
    pruning_config_t _pruning_config;

    /** @brief Propagation configuration. */
    propagate_config_t _propagate_config;

};  // class IndexStructure

}   // namespace conv_graph
}   // namespace cpu
}   // namespace artea
