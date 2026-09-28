/*
 * @FilePath: /Artea/include/artea/cpu/index/conv_graph/index_structure.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Convergent graph index structure holding a RefiningGraph by
 *               composition via @c std::unique_ptr. Exposes the inner graph
 *               via @c get_refining_graph so callers can transfer or reuse
 *               it, and forwards the common @c RefiningGraph public methods
 *               so existing template call sites (propagate engine, updaters,
 *               routers, file manager, compactor) keep working unchanged.
 */

#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <artea/cpu/index/dataset_index.hpp>
#include <artea/cpu/index/compactor/refining_graph_compactor.hpp>

namespace artea {
namespace cpu {
namespace conv_graph {

/**
 * @brief Convergent graph index. Composes a @c RefiningGraph plus the
 *        algorithm-specific pruning and propagation configs.
 *
 * This class used to inherit from @c RefiningGraph via CRTP. It now holds
 * the graph through a @c std::unique_ptr, which:
 *   - makes move-assigning @c IndexStructure safe (the legacy @c RefiningGraph
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
class IndexStructure : public DatasetIndex<IndexTraitsT> {
    using data_base_t = DatasetIndex<IndexTraitsT>;
    using knn_index_t = typename IndexTraitsT::knn_graph::index_t;

    using refining_graph_t    = typename IndexTraitsT::dynamic::refining_graph_t;
    using vertex_num_t       = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t        = typename IndexTraitsT::vertex_id_t;
    using nbr_arr_t         = typename IndexTraitsT::nbr_arr_t;
    using vector_array_t     = typename IndexTraitsT::vector_array_t;
    using layer_config_t     = typename IndexTraitsT::layer_config_t;
    using propagate_config_t = typename IndexTraitsT::conv_graph::propagate_config_t;
    using pruning_config_t   = typename IndexTraitsT::conv_graph::pruning_config_t;

public:
    using vector_dataset_t = typename IndexTraitsT::vector_dataset_t;
    using compact_graph_t = typename IndexTraitsT::compact::refining_graph_t;

    /**
     * @brief Construct a new convergent graph index.
     * @param vecs_data        Reference to the vector data for this layer.
     * @param layer_config     Layer configuration (max_nbr_size).
     * @param pruning_config   Pruning configuration (scale_coeffs and shifted_coeffs).
     * @param propagate_config Propagation configuration (num_build_loops, num_triu_iters, prefill_ratio).
     */
    IndexStructure(
        const vector_array_t& vecs_data,
        const layer_config_t layer_config,
        const pruning_config_t pruning_config,
        const propagate_config_t propagate_config
    ) : data_base_t(vecs_data), _refining_graph(std::make_unique<refining_graph_t>(vecs_data, layer_config)),
        _pruning_config(pruning_config),
        _propagate_config(propagate_config)
    {}

    /** @brief Own one dataset across builds and queries; prepare_build() creates the graph. */
    explicit IndexStructure(std::unique_ptr<vector_dataset_t> dataset)
        : data_base_t(std::move(dataset)) {}

    IndexStructure(std::unique_ptr<vector_dataset_t> dataset, const layer_config_t layer_config,
                   const pruning_config_t pruning_config, const propagate_config_t propagate_config)
        : data_base_t(std::move(dataset)) {
        prepare_build(layer_config, pruning_config, propagate_config);
    }

    IndexStructure(vector_array_t&&, layer_config_t, pruning_config_t, propagate_config_t) = delete;
    IndexStructure(const vector_array_t&&, layer_config_t, pruning_config_t, propagate_config_t) = delete;

    /** @brief Replace graph state and configs; preserve dataset addresses even if allocation fails.
     *  Invalidates graph references. Call only after queries on the previous graph have finished. */
    auto prepare_build(const layer_config_t layer_config, const pruning_config_t pruning_config,
                       const propagate_config_t propagate_config) -> IndexStructure& {
        _compact_graph.reset();
        _refining_graph.reset();
        auto graph = std::make_unique<refining_graph_t>(this->get_base_vecs(), layer_config);
        _pruning_config = pruning_config;
        _propagate_config = propagate_config;
        _refining_graph = std::move(graph);
        return *this;
    }

    /** @brief Compact successfully before releasing construction state; retain all vector data. */
    auto compact(vertex_num_t extracted_nbr_size) -> void {
        using compactor_t = typename IndexTraitsT::refining_graph_compactor_t;
        auto graph = compactor_t::compact_graph(get_refining_graph(), extracted_nbr_size);
        auto compact_graph = std::make_unique<compact_graph_t>(std::move(graph));
        _compact_graph = std::move(compact_graph);
        _refining_graph.reset();
    }

    /** @brief Search graph available after compaction and until the next rebuild. */
    auto get_compact_graph() const -> const compact_graph_t& {
        if (!_compact_graph) throw std::logic_error("No compact graph; call compact first");
        return *_compact_graph;
    }

    /** @brief Release construction state while retaining dataset and any compact graph. */
    auto release() -> void { _refining_graph.reset(); }

    /** @brief Convert a KNN index by transferring its dataset and entire building graph together. */
    IndexStructure(knn_index_t&& source, const pruning_config_t pruning_config)
        : data_base_t(_require_building_index(source)),
          _refining_graph(std::move(source._refining_graph)),
          _pruning_config(pruning_config), _propagate_config(propagate_config_t(0, 0)) {}

    IndexStructure(const IndexStructure&) = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;

    // Move is O(1): the @c unique_ptr swap avoids touching any of
    // @c RefiningGraph's internal fields (including the @c _vecs_data
    // reference, which the legacy move-assign had to leave stale).
    IndexStructure(IndexStructure&&) noexcept = default;
    /** @brief Destroy old vector references before replacing their owner. */
    auto operator=(IndexStructure&& other) noexcept -> IndexStructure& {
        if (this != &other) {
            _compact_graph.reset();
            _refining_graph.reset();
            data_base_t::operator=(std::move(other));
            _refining_graph = std::move(other._refining_graph);
            _compact_graph = std::move(other._compact_graph);
            _pruning_config = std::move(other._pruning_config);
            _propagate_config = std::move(other._propagate_config);
        }
        return *this;
    }

    // --- Composed graph accessor ---

    /**
     * @brief Access the underlying @c RefiningGraph instance. Use this to
     *        pass the graph directly to utilities that only need the
     *        graph core, or to read it for inspection.
     *
     * The returned reference is valid as long as this @c IndexStructure
     * holds the graph — i.e. until the next move-assign or destruction.
     */
    __attribute__((always_inline))
    auto get_refining_graph() -> refining_graph_t& {
        if (!_refining_graph) throw std::logic_error("No building graph; call prepare_build first");
        return *_refining_graph;
    }

    __attribute__((always_inline))
    auto get_refining_graph() const -> const refining_graph_t& {
        if (!_refining_graph) throw std::logic_error("No building graph; call prepare_build first");
        return *_refining_graph;
    }

    // --- RefiningGraph public API forwarders ---
    //
    // These keep call sites that previously relied on inheritance working
    // unchanged. Every forwarder is a thin inline call to the matching
    // method on @c *_refining_graph.

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return get_refining_graph().get_num_vertices();
    }

    __attribute__((always_inline))
    auto layer_config() const -> const layer_config_t& {
        return get_refining_graph().layer_config();
    }

    __attribute__((always_inline))
    auto layer_config() -> layer_config_t& {
        return get_refining_graph().layer_config();
    }

    __attribute__((always_inline))
    auto get_nbrs_arr() -> std::vector<nbr_arr_t>& {
        return get_refining_graph().get_nbrs_arr();
    }

    __attribute__((always_inline))
    auto get_nbrs_arr() const -> const std::vector<nbr_arr_t>& {
        return get_refining_graph().get_nbrs_arr();
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) const -> const nbr_arr_t& {
        return get_refining_graph().fetch_nbrs(src);
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) -> nbr_arr_t& {
        return get_refining_graph().fetch_nbrs(src);
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return this->get_base_vecs();
    }

    __attribute__((always_inline))
    auto is_identity_mapped() const -> bool {
        return get_refining_graph().is_identity_mapped();
    }

    __attribute__((always_inline))
    auto get_storage_vid(const vertex_num_t local_idx) const -> vertex_id_t {
        return get_refining_graph().get_storage_vid(local_idx);
    }

    auto get_base_metadata() const -> nlohmann::json {
        return get_refining_graph().get_base_metadata();
    }

    // --- Config accessors ---

    __attribute__((always_inline))
    auto pruning_config() const -> const pruning_config_t& { return _pruning_config.value(); }

    __attribute__((always_inline))
    auto pruning_config() -> pruning_config_t& { return _pruning_config.value(); }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& { return _propagate_config.value(); }

    __attribute__((always_inline))
    auto propagate_config() -> propagate_config_t& { return _propagate_config.value(); }

    // --- Metadata hooks ---

    auto get_metadata() const -> nlohmann::json {
        nlohmann::json meta;
        meta["pruning_config"] = {
            {"scale_coeffs", _pruning_config.value().scale_coeffs()},
            {"shifted_coeffs", _pruning_config.value().shifted_coeffs()}
        };
        meta["propagate_config"] = {
            {"num_build_loops", _propagate_config.value().num_build_loops()},
            {"num_triu_iters", _propagate_config.value().num_triu_iters()},
            {"prefill_ratio", _propagate_config.value().prefill_ratio()},
            {"num_routing_loops", _propagate_config.value().num_routing_loops()},
            {"routing_topk", _propagate_config.value().routing_topk()},
            {"routing_queue_size", _propagate_config.value().routing_queue_size()}
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
    static auto _require_building_index(knn_index_t& source) -> knn_index_t&& {
        source.get_refining_graph();
        return std::move(source);
    }

    /**
     * @brief Composed @c RefiningGraph — owns the graph topology and
     *        layer config. Held through @c unique_ptr so @c IndexStructure
     *        move operations are O(1) pointer swaps and the legacy
     *        @c RefiningGraph move-assign (which cannot reseat its
     *        @c _vecs_data reference) is never invoked.
     */
    std::unique_ptr<refining_graph_t> _refining_graph;

    /** @brief Search topology also borrowing that vector array; destroyed before the base. */
    std::unique_ptr<compact_graph_t> _compact_graph;

    /** @brief Pruning configuration. */
    std::optional<pruning_config_t> _pruning_config;

    /** @brief Propagation configuration. */
    std::optional<propagate_config_t> _propagate_config;

};  // class IndexStructure

}   // namespace conv_graph
}   // namespace cpu
}   // namespace artea
