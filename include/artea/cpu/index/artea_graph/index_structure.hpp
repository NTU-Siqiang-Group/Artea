/*
 * @FilePath: /Artea/include/artea/cpu/index/artea_graph/index_structure.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Artea hierarchical graph index structure extending HierarchicalGraph via CRTP.
 */

#pragma once

#include <nlohmann/json.hpp>
#include <artea/common/logger.hpp>
#include <fmt/format.h>

namespace artea {
namespace cpu {
namespace artea_graph {

/**
 * @brief Artea hierarchical graph index, extending HierarchicalGraph with
 *        pruning, propagation, and R-net configs.
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class IndexStructure :
    public IndexTraitsT::template hierarchical_graph_t<
        IndexStructure<IndexTraitsT>, typename IndexTraitsT::conv_graph::index_t>
{
    using base_t = typename IndexTraitsT::template hierarchical_graph_t<
        IndexStructure<IndexTraitsT>, typename IndexTraitsT::conv_graph::index_t>;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using layer_config_t = typename IndexTraitsT::layer_config_t;
    using propagate_config_t = typename IndexTraitsT::artea_graph::propagate_config_t;
    using pruning_config_t = typename IndexTraitsT::artea_graph::pruning_config_t;
    using rnet_config_t = typename IndexTraitsT::artea_graph::rnet_config_t;

public:
    using layer_graph_t = typename IndexTraitsT::conv_graph::index_t;

    /**
     * @brief Construct a new Artea hierarchical graph index.
     */
    IndexStructure(
        const vector_array_t& base_vecs,
        const layer_config_t bottom_layer_config,
        const layer_config_t upper_layer_config,
        const pruning_config_t bottom_pruning_config,
        const pruning_config_t upper_pruning_config,
        const propagate_config_t propagate_config,
        const rnet_config_t rnet_config
    ) : base_t(base_vecs, bottom_layer_config, upper_layer_config),
        _bottom_pruning_config(bottom_pruning_config),
        _upper_pruning_config(upper_pruning_config),
        _propagate_config(propagate_config),
        _rnet_config(rnet_config)
    {}

    IndexStructure(const IndexStructure&) = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;

    IndexStructure(IndexStructure&&) noexcept = default;
    IndexStructure& operator=(IndexStructure&&) noexcept = default;

    // --- Config accessors ---

    __attribute__((always_inline))
    auto bottom_pruning_config() const -> const pruning_config_t& { return _bottom_pruning_config; }

    __attribute__((always_inline))
    auto bottom_pruning_config() -> pruning_config_t& { return _bottom_pruning_config; }

    __attribute__((always_inline))
    auto upper_pruning_config() const -> const pruning_config_t& { return _upper_pruning_config; }

    __attribute__((always_inline))
    auto upper_pruning_config() -> pruning_config_t& { return _upper_pruning_config; }

    __attribute__((always_inline))
    auto propagate_config() const -> const propagate_config_t& { return _propagate_config; }

    __attribute__((always_inline))
    auto propagate_config() -> propagate_config_t& { return _propagate_config; }

    __attribute__((always_inline))
    auto rnet_config() const -> const rnet_config_t& { return _rnet_config; }

    __attribute__((always_inline))
    auto rnet_config() -> rnet_config_t& { return _rnet_config; }

    // --- Metadata hooks ---

    auto get_metadata() const -> nlohmann::json {
        nlohmann::json meta;
        meta["bottom_pruning_config"] = {
            {"scale_coeffs", _bottom_pruning_config.scale_coeffs()},
            {"shifted_coeffs", _bottom_pruning_config.shifted_coeffs()}
        };
        meta["upper_pruning_config"] = {
            {"scale_coeffs", _upper_pruning_config.scale_coeffs()},
            {"shifted_coeffs", _upper_pruning_config.shifted_coeffs()}
        };
        meta["propagate_config"] = {
            {"num_build_loops", _propagate_config.num_build_loops()},
            {"num_triu_iters", _propagate_config.num_triu_iters()},
            {"prefill_ratio", _propagate_config.prefill_ratio()},
            {"num_routing_loops", _propagate_config.num_routing_loops()},
            {"routing_topk", _propagate_config.routing_topk()},
            {"routing_queue_size", _propagate_config.routing_queue_size()}
        };
        meta["rnet_config"] = {
            {"mis_radix", _rnet_config.mis_radix()},
            {"mis_max_power", _rnet_config.mis_max_power()},
            {"rnet_beta", _rnet_config.rnet_beta()}
        };
        return meta;
    }

    static auto from_metadata(
        const nlohmann::json& meta,
        const vector_array_t& base_vecs,
        const layer_config_t& bottom_layer_config,
        const layer_config_t& upper_layer_config
    ) -> IndexStructure {
        using ratio_t = typename IndexTraitsT::ratio_t;
        using iter_t = typename IndexTraitsT::iter_t;
        using distance_t = typename IndexTraitsT::distance_t;
        using vertex_num_t = typename IndexTraitsT::vertex_num_t;

        pruning_config_t bottom_pruning_config(
            meta["bottom_pruning_config"]["scale_coeffs"].get<ratio_t>(),
            meta["bottom_pruning_config"]["shifted_coeffs"].get<ratio_t>()
        );
        pruning_config_t upper_pruning_config(
            meta["upper_pruning_config"]["scale_coeffs"].get<ratio_t>(),
            meta["upper_pruning_config"]["shifted_coeffs"].get<ratio_t>()
        );
        propagate_config_t propagate_config(
            meta["propagate_config"]["num_build_loops"].get<iter_t>(),
            meta["propagate_config"]["num_triu_iters"].get<iter_t>(),
            meta["propagate_config"]["prefill_ratio"].get<ratio_t>(),
            meta["propagate_config"]["num_routing_loops"].get<iter_t>(),
            meta["propagate_config"].value("routing_topk", vertex_num_t(64)),
            meta["propagate_config"].value("routing_queue_size", vertex_num_t(96))
        );
        rnet_config_t rnet_config(
            meta["rnet_config"].value("mis_radix", distance_t(1.2)),
            meta["rnet_config"].value("mis_max_power", uint32_t(12)),
            meta["rnet_config"].value("rnet_beta", ratio_t(1.44))
        );

        return IndexStructure(
            base_vecs, bottom_layer_config, upper_layer_config,
            bottom_pruning_config, upper_pruning_config,
            propagate_config, rnet_config
        );
    }

private:
    /** @brief Pruning configuration for bottom layer. */
    pruning_config_t _bottom_pruning_config;

    /** @brief Pruning configuration for upper layers. */
    pruning_config_t _upper_pruning_config;

    /** @brief Propagation configuration (shared between layers). */
    propagate_config_t _propagate_config;

    /** @brief R-net configuration for GraphMIS vertex selection. */
    rnet_config_t _rnet_config;

};  // class IndexStructure

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
