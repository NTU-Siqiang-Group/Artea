/*
 * @FilePath: /Artea/include/artea/cpu/index/knn_graph/index_structure.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: KNN graph index structure. Composes a RefiningGraph plus
 *               propagation config. No RNG pruning config is stored —
 *               the KNN build pipeline does not invoke PruningUpdater.
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
namespace knn_graph {

/**
 * @brief KNN graph index. Composes a @c RefiningGraph with a
 *        caller-supplied propagation config.
 *
 * The KNN build pipeline (random init + triangle + reverse + truncate +
 * routing) does not consume any RNG pruning coefficients, so no
 * @c PruningConfig is stored.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class IndexStructure : public DatasetIndex<IndexTraitsT> {
    using data_base_t = DatasetIndex<IndexTraitsT>;

    using refining_graph_t    = typename IndexTraitsT::dynamic::refining_graph_t;
    using vertex_num_t       = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t        = typename IndexTraitsT::vertex_id_t;
    using nbr_arr_t         = typename IndexTraitsT::nbr_arr_t;
    using vector_array_t     = typename IndexTraitsT::vector_array_t;
    using layer_config_t     = typename IndexTraitsT::layer_config_t;
    using propagate_config_t = typename IndexTraitsT::knn_graph::propagate_config_t;

public:
    using vector_dataset_t = typename IndexTraitsT::vector_dataset_t;
    using compact_graph_t = typename IndexTraitsT::compact::refining_graph_t;

    /**
     * @brief Construct a new KNN graph index.
     * @param vecs_data        Reference to the vector data.
     * @param layer_config     Layer configuration (max_nbr_size).
     * @param propagate_config Propagation configuration.
     */
    IndexStructure(
        const vector_array_t& vecs_data,
        const layer_config_t layer_config,
        const propagate_config_t propagate_config
    ) : data_base_t(vecs_data), _refining_graph(std::make_unique<refining_graph_t>(vecs_data, layer_config)),
        _propagate_config(propagate_config)
    {}

    /** @brief Own one dataset across builds and queries; prepare_build() creates the graph. */
    explicit IndexStructure(std::unique_ptr<vector_dataset_t> dataset)
        : data_base_t(std::move(dataset)) {}

    IndexStructure(std::unique_ptr<vector_dataset_t> dataset, const layer_config_t layer_config,
                   const propagate_config_t propagate_config)
        : data_base_t(std::move(dataset)) {
        prepare_build(layer_config, propagate_config);
    }

    IndexStructure(vector_array_t&&, layer_config_t, propagate_config_t) = delete;
    IndexStructure(const vector_array_t&&, layer_config_t, propagate_config_t) = delete;

    /** @brief Replace graph state and configs; preserve dataset addresses even if allocation fails.
     *  Invalidates graph references. Call only after queries on the previous graph have finished. */
    auto prepare_build(const layer_config_t layer_config,
                   const propagate_config_t propagate_config) -> IndexStructure& {
        _compact_graph.reset();
        _refining_graph.reset();
        auto graph = std::make_unique<refining_graph_t>(this->get_base_vecs(), layer_config);
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

    IndexStructure(const IndexStructure&) = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;

    IndexStructure(IndexStructure&&) noexcept = default;
    /** @brief Destroy old vector references before replacing their owner. */
    auto operator=(IndexStructure&& other) noexcept -> IndexStructure& {
        if (this != &other) {
            _compact_graph.reset();
            _refining_graph.reset();
            data_base_t::operator=(std::move(other));
            _refining_graph = std::move(other._refining_graph);
            _compact_graph = std::move(other._compact_graph);
            _propagate_config = std::move(other._propagate_config);
        }
        return *this;
    }

    // --- Composed graph accessor ---

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
    auto propagate_config() const -> const propagate_config_t& { return _propagate_config.value(); }

    __attribute__((always_inline))
    auto propagate_config() -> propagate_config_t& { return _propagate_config.value(); }

private:
    friend class conv_graph::IndexStructure<IndexTraitsT>;

    /** @brief Building topology referencing the stable vector array in the data-owning base. */
    std::unique_ptr<refining_graph_t> _refining_graph;

    /** @brief Search topology also borrowing that vector array; destroyed before the base. */
    std::unique_ptr<compact_graph_t> _compact_graph;
    std::optional<propagate_config_t> _propagate_config;
};

}   // namespace knn_graph
}   // namespace cpu
}   // namespace artea
