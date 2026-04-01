/*
 * @FilePath: /Artea/include/artea/cpu/index/artea_graph/index_structure.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Artea hierarchical graph index structure extending HierarchicalGraph via CRTP.
 */

#pragma once

#include <variant>
#include <nlohmann/json.hpp>
#include <artea/common/logger.hpp>
#include <fmt/format.h>

namespace artea {
namespace cpu {
namespace artea_graph {

/**
 * @brief Artea hierarchical graph index, extending HierarchicalGraph with
 *        pruning, propagation, and vertices builder configs.
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
    using greedy_vertices_builder_config_t = typename IndexTraitsT::greedy_vertices_builder_config_t;
    using random_vertices_builder_config_t = typename IndexTraitsT::random_vertices_builder_config_t;
    using vertices_builder_config_t = typename IndexTraitsT::vertices_builder_config_t;

public:
    using layer_graph_t = typename IndexTraitsT::conv_graph::index_t;
    /**
     * @brief Construct a new Artea hierarchical graph index.
     * @param base_vecs Reference to the base layer vector data.
     * @param bottom_layer_config Configuration for bottom layer.
     * @param upper_layer_config Configuration for upper layers.
     * @param bottom_pruning_config Pruning configuration for bottom layer.
     * @param upper_pruning_config Pruning configuration for upper layers.
     * @param propagate_config Propagation configuration (shared between layers).
     * @param vertices_builder_config Configuration for vertices builder (greedy or random).
     */
    IndexStructure(
        const vector_array_t& base_vecs,
        const layer_config_t bottom_layer_config,
        const layer_config_t upper_layer_config,
        const pruning_config_t bottom_pruning_config,
        const pruning_config_t upper_pruning_config,
        const propagate_config_t propagate_config,
        const vertices_builder_config_t vertices_builder_config
    ) : base_t(base_vecs, bottom_layer_config, upper_layer_config),
        _bottom_pruning_config(bottom_pruning_config),
        _upper_pruning_config(upper_pruning_config),
        _propagate_config(propagate_config),
        _vertices_builder_config(vertices_builder_config)
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
    auto vertices_builder_config() const -> const vertices_builder_config_t& { return _vertices_builder_config; }

    __attribute__((always_inline))
    auto vertices_builder_config() -> vertices_builder_config_t& { return _vertices_builder_config; }

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
            {"num_routing_loops", _propagate_config.num_routing_loops()}
        };
        // Save vertices_builder_config based on which variant is active
        if (std::holds_alternative<greedy_vertices_builder_config_t>(_vertices_builder_config)) {
            const auto& config = std::get<greedy_vertices_builder_config_t>(_vertices_builder_config);
            meta["vertices_builder_config"] = {
                {"type", "approx_rnet"},
                {"min_radius", config.min_radius()},
                {"beta", config.beta()},
                {"coverage_ratio", config.coverage_ratio()},
                {"confidence", config.confidence()},
                {"max_result_ratio", config.max_result_ratio()},
                {"sampling_batch_size", config.sampling_batch_size()}
            };
        } else if (std::holds_alternative<random_vertices_builder_config_t>(_vertices_builder_config)) {
            const auto& config = std::get<random_vertices_builder_config_t>(_vertices_builder_config);
            meta["vertices_builder_config"] = {
                {"type", "random"},
                {"random_result_ratio", config.random_result_ratio()}
            };
        }
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
            meta["propagate_config"]["num_routing_loops"].get<iter_t>()
        );

        const std::string vb_type = meta["vertices_builder_config"]["type"].get<std::string>();
        vertices_builder_config_t vertices_builder_config = [&]() -> vertices_builder_config_t {
            if (vb_type == "approx_rnet") {
                return greedy_vertices_builder_config_t(
                    meta["vertices_builder_config"]["min_radius"].get<distance_t>(),
                    meta["vertices_builder_config"]["beta"].get<ratio_t>(),
                    meta["vertices_builder_config"]["coverage_ratio"].get<ratio_t>(),
                    meta["vertices_builder_config"]["confidence"].get<ratio_t>(),
                    meta["vertices_builder_config"]["max_result_ratio"].get<ratio_t>(),
                    meta["vertices_builder_config"]["sampling_batch_size"].get<vertex_num_t>()
                );
            } else if (vb_type == "random") {
                return random_vertices_builder_config_t(
                    meta["vertices_builder_config"]["random_result_ratio"].get<ratio_t>()
                );
            }
            ARTEA_ERROR(fmt::format("Unknown vertices_builder_config type: {}", vb_type));
            __builtin_unreachable();
        }();

        return IndexStructure(
            base_vecs, bottom_layer_config, upper_layer_config,
            bottom_pruning_config, upper_pruning_config,
            propagate_config, vertices_builder_config
        );
    }

private:
    /** @brief Pruning configuration for bottom layer. */
    pruning_config_t _bottom_pruning_config;

    /** @brief Pruning configuration for upper layers. */
    pruning_config_t _upper_pruning_config;

    /** @brief Propagation configuration (shared between layers). */
    propagate_config_t _propagate_config;

    /** @brief Vertices builder configuration (greedy or random). */
    vertices_builder_config_t _vertices_builder_config;

};  // class IndexStructure

}   // namespace artea_graph
}   // namespace cpu
}   // namespace artea
