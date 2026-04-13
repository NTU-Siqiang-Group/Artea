/*
 * @FilePath: /Artea/include/artea/cpu/index/knn_graph/index_structure.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: KNN graph index structure. Composes a RefiningGraph plus
 *               propagation config. Pruning is hardcoded to identity
 *               (scale_coeffs=1.0, shifted_coeffs=0.0) since a KNN graph
 *               does not use RNG-style pruning.
 */

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace artea {
namespace cpu {
namespace knn_graph {

/**
 * @brief KNN graph index. Composes a @c RefiningGraph with a fixed
 *        identity pruning config and a caller-supplied propagation config.
 *
 * Unlike @c conv_graph::IndexStructure, the pruning configuration is NOT
 * a constructor parameter — it is always @c {1.0, 0.0} (no RNG pruning).
 * This makes the API cleaner for KNN-graph callers who should never have
 * to think about pruning coefficients.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class IndexStructure {

    using refining_graph_t    = typename IndexTraitsT::dynamic::refining_graph_t;
    using vertex_num_t       = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t        = typename IndexTraitsT::vertex_id_t;
    using nbr_arr_t         = typename IndexTraitsT::nbr_arr_t;
    using vector_array_t     = typename IndexTraitsT::vector_array_t;
    using layer_config_t     = typename IndexTraitsT::layer_config_t;
    using propagate_config_t = typename IndexTraitsT::knn_graph::propagate_config_t;
    using pruning_config_t   = typename IndexTraitsT::knn_graph::pruning_config_t;
    using ratio_t            = typename IndexTraitsT::ratio_t;

public:
    /**
     * @brief Construct a new KNN graph index.
     * @param vecs_data        Reference to the vector data.
     * @param layer_config     Layer configuration (max_nbr_size, reserved_nbr_size).
     * @param propagate_config Propagation configuration.
     */
    IndexStructure(
        const vector_array_t& vecs_data,
        const layer_config_t layer_config,
        const propagate_config_t propagate_config
    ) : _refining_graph(std::make_unique<refining_graph_t>(vecs_data, layer_config)),
        _pruning_config(ratio_t(1.0), ratio_t(0.0)),
        _propagate_config(propagate_config)
    {}

    IndexStructure(const IndexStructure&) = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;

    IndexStructure(IndexStructure&&) noexcept = default;
    IndexStructure& operator=(IndexStructure&&) noexcept = default;

    // --- Composed graph accessor ---

    __attribute__((always_inline))
    auto get_refining_graph() -> refining_graph_t& { return *_refining_graph; }

    __attribute__((always_inline))
    auto get_refining_graph() const -> const refining_graph_t& { return *_refining_graph; }

    // --- RefiningGraph public API forwarders ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _refining_graph->get_num_vertices();
    }

    __attribute__((always_inline))
    auto layer_config() const -> const layer_config_t& {
        return _refining_graph->layer_config();
    }

    __attribute__((always_inline))
    auto layer_config() -> layer_config_t& {
        return _refining_graph->layer_config();
    }

    __attribute__((always_inline))
    auto get_nbrs_arr() -> std::vector<nbr_arr_t>& {
        return _refining_graph->get_nbrs_arr();
    }

    __attribute__((always_inline))
    auto get_nbrs_arr() const -> const std::vector<nbr_arr_t>& {
        return _refining_graph->get_nbrs_arr();
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) const -> const nbr_arr_t& {
        return _refining_graph->fetch_nbrs(src);
    }

    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t src) -> nbr_arr_t& {
        return _refining_graph->fetch_nbrs(src);
    }

    __attribute__((always_inline))
    auto get_vecs_data() const -> const vector_array_t& {
        return _refining_graph->get_vecs_data();
    }

    __attribute__((always_inline))
    auto is_identity_mapped() const -> bool {
        return _refining_graph->is_identity_mapped();
    }

    __attribute__((always_inline))
    auto vid_at(const vertex_num_t local_idx) const -> vertex_id_t {
        return _refining_graph->vid_at(local_idx);
    }

    template <typename Fn>
    auto parallel_for_each_vertex(Fn&& fn) const -> void {
        _refining_graph->parallel_for_each_vertex(std::forward<Fn>(fn));
    }

    auto get_base_metadata() const -> nlohmann::json {
        return _refining_graph->get_base_metadata();
    }

    // --- Config accessors ---

    __attribute__((always_inline))
    auto pruning_config() const -> const pruning_config_t& { return _pruning_config; }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& { return _propagate_config; }

    __attribute__((always_inline))
    auto propagate_config() -> propagate_config_t& { return _propagate_config; }

private:
    std::unique_ptr<refining_graph_t> _refining_graph;
    pruning_config_t   _pruning_config;
    propagate_config_t _propagate_config;
};

}   // namespace knn_graph
}   // namespace cpu
}   // namespace artea
