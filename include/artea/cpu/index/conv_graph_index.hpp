/*
 * @FilePath: /Artea/include/artea/cpu/index/conv_graph_index.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Convergent graph index structure extending FlatGraph via CRTP.
 */

#pragma once

#include <nlohmann/json.hpp>

namespace artea {
namespace cpu {
namespace conv_graph {

/**
 * @brief Convergent graph index, extending FlatGraph with pruning and propagation configs.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class IndexStructure :
    public IndexTraitsT::template flat_graph_t<IndexStructure<IndexTraitsT>>
{
    using base_t = typename IndexTraitsT::template flat_graph_t<IndexStructure<IndexTraitsT>>;
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
    IndexStructure(
        const vector_array_t& vecs_data,
        const layer_config_t layer_config,
        const pruning_config_t pruning_config,
        const propagate_config_t propagate_config
    ) : base_t(vecs_data, layer_config),
        _pruning_config(pruning_config),
        _propagate_config(propagate_config)
    {}

    IndexStructure(const IndexStructure&) = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;

    IndexStructure(IndexStructure&&) noexcept = default;
    IndexStructure& operator=(IndexStructure&&) noexcept = default;

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
            {"num_routing_loops", _propagate_config.num_routing_loops()}
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
            meta["propagate_config"]["num_routing_loops"].get<uint32_t>()
        );
        return IndexStructure(vecs_data, layer_config, pruning_config, propagate_config);
    }

private:
    /** @brief Pruning configuration. */
    pruning_config_t _pruning_config;

    /** @brief Propagation configuration. */
    propagate_config_t _propagate_config;

};  // class IndexStructure

}   // namespace conv_graph
}   // namespace cpu
}   // namespace artea
