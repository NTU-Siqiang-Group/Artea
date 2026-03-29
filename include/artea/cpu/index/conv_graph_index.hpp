/*
 * @FilePath: /Artea/include/artea/cpu/index/conv_graph_index.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Convergent graph index structure extending FlatGraph via CRTP.
 */

#pragma once

namespace artea {
namespace cpu {
namespace conv_graph {

/**
 * @brief Convergent graph index, extending FlatGraph with pruning and propagation configs.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class GraphIndex :
    public IndexTraitsT::template flat_graph_t<GraphIndex<IndexTraitsT>>
{
    using base_t = typename IndexTraitsT::template flat_graph_t<GraphIndex<IndexTraitsT>>;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;
    using propagate_config_t = typename IndexTraitsT::conv_graph::propagate_config_t;
    using pruning_config_t = typename IndexTraitsT::conv_graph::pruning_config_t;

public:
    /**
     * @brief Construct a new convergent graph index.
     * @param vecs_data Reference to the vector data for this layer.
     * @param layer_config Layer configuration (max_nbr_size and reserved_nbr_size).
     * @param pruning_config Pruning configuration (scale_coeffs and shifted_coeffs).
     * @param propagate_config Propagation configuration (num_build_loops, num_triu_iters, prefill_ratio).
     */
    GraphIndex(
        const vector_array_t& vecs_data,
        const layer_config_t layer_config,
        const pruning_config_t pruning_config,
        const propagate_config_t propagate_config
    ) : base_t(vecs_data, layer_config),
        _pruning_config(pruning_config),
        _propagate_config(propagate_config)
    {}

    GraphIndex(const GraphIndex&) = delete;
    GraphIndex& operator=(const GraphIndex&) = delete;

    GraphIndex(GraphIndex&&) noexcept = default;
    GraphIndex& operator=(GraphIndex&&) noexcept = default;

    // --- Config accessors ---

    __attribute__((always_inline))
    auto pruning_config() const -> const pruning_config_t& { return _pruning_config; }

    __attribute__((always_inline))
    auto pruning_config() -> pruning_config_t& { return _pruning_config; }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& { return _propagate_config; }

    __attribute__((always_inline))
    auto propagate_config() -> propagate_config_t& { return _propagate_config; }

private:
    /** @brief Pruning configuration. */
    pruning_config_t _pruning_config;

    /** @brief Propagation configuration. */
    propagate_config_t _propagate_config;

};  // class GraphIndex

}   // namespace conv_graph
}   // namespace cpu
}   // namespace artea
