// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <artea/cpu/index/compact_structure/single_layer_graph.hpp>
#include <artea/cpu/index/compactor/single_layer_graph_compactor.hpp>
#include <artea/cpu/index/dataset_index.hpp>
#include <artea/cpu/index/dynamic_structure/single_layer_graph.hpp>
#include <artea/cpu/index/practheo/configs.hpp>
#include <artea/cpu/index/single_layer_config.hpp>

namespace artea::cpu::practheo {

/** @brief Flat graph with one adjacency row per base vector, indexed by its vertex ID. */
template <typename IndexTraitsT>
class IndexStructure : public DatasetIndex<IndexTraitsT> {
    using data_base_t = DatasetIndex<IndexTraitsT>;
    using vector_array_t = typename IndexTraitsT::vector_array_t;
    using sl_config_t = typename IndexTraitsT::sl_config_t;
    using build_config_t = typename IndexTraitsT::practheo::build_config_t;
    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using nbr_arr_t = typename IndexTraitsT::nbr_arr_t;

public:
    using vector_dataset_t = typename IndexTraitsT::vector_dataset_t;
    using single_layer_graph_t = typename IndexTraitsT::dynamic::single_layer_graph_t;
    using compact_graph_t = typename IndexTraitsT::compact::single_layer_graph_t;

    IndexStructure(std::unique_ptr<vector_dataset_t> dataset, const build_config_t& build_config)
        : data_base_t(std::move(dataset)), _build_config(build_config) { initialize_weight_groups(); }

    IndexStructure(std::unique_ptr<vector_dataset_t> dataset, sl_config_t layer_config,
                   const build_config_t& build_config)
        : IndexStructure(std::move(dataset), build_config) { prepare_build(layer_config); }

    /** @brief Borrow base vectors that outlive the index and its graph. */
    IndexStructure(const vector_array_t& vectors, sl_config_t layer_config, const build_config_t& build_config)
        : data_base_t(vectors), _build_config(build_config) {
        initialize_weight_groups();
        prepare_build(layer_config);
    }

    IndexStructure(vector_array_t&&, sl_config_t, const build_config_t&) = delete;
    IndexStructure(const vector_array_t&&, sl_config_t, const build_config_t&) = delete;
    IndexStructure(IndexStructure&&) noexcept = default;
    auto operator=(IndexStructure&& other) noexcept -> IndexStructure& {
        if (this != &other) {
            _single_layer_graph.reset();
            _compact_graph.reset();
            data_base_t::operator=(std::move(other));
            _build_config = std::move(other._build_config);
            _vids_by_weight = std::move(other._vids_by_weight);
            _single_layer_graph = std::move(other._single_layer_graph);
            _compact_graph = std::move(other._compact_graph);
        }
        return *this;
    }

    /** @brief Start a fresh build with empty adjacency rows and vertex weight groups. */
    auto prepare_build(sl_config_t layer_config) -> IndexStructure& {
        _single_layer_graph = std::make_unique<single_layer_graph_t>(this->get_base_vecs(), layer_config);
        _compact_graph.reset();
        for (auto& group : _vids_by_weight) group.clear();
        return *this;
    }

    auto get_single_layer_graph() -> single_layer_graph_t& {
        if (!_single_layer_graph) throw std::logic_error("Call prepare_build() before accessing the building graph");
        return *_single_layer_graph;
    }

    auto get_single_layer_graph() const -> const single_layer_graph_t& {
        if (!_single_layer_graph) throw std::logic_error("Call prepare_build() before accessing the building graph");
        return *_single_layer_graph;
    }

    auto get_num_vertices() const -> vertex_num_t { return get_single_layer_graph().get_num_vertices(); }
    auto build_config() const -> const build_config_t& { return _build_config; }
    /** @brief Global vertex IDs in base order for each weight from zero through the configured max_weight. */
    auto get_vids_by_weight() const -> const auto& { return _vids_by_weight; }
    auto get_vecs_data() const -> const vector_array_t& { return this->get_base_vecs(); }
    auto layer_config() const -> const sl_config_t& { return get_single_layer_graph().layer_config(); }
    auto layer_config() -> sl_config_t& { return get_single_layer_graph().layer_config(); }
    auto get_nbrs_arr() const -> const auto& { return get_single_layer_graph().get_nbrs_arr(); }
    auto get_nbrs_arr() -> auto& { return get_single_layer_graph().get_nbrs_arr(); }
    auto fetch_nbrs(vertex_id_t vid) const -> const nbr_arr_t& {
        return get_single_layer_graph().fetch_nbrs(vid);
    }
    auto fetch_nbrs(vertex_id_t vid) -> nbr_arr_t& {
        return get_single_layer_graph().fetch_nbrs(vid);
    }
    auto is_identity_mapped() const -> bool { return get_single_layer_graph().is_identity_mapped(); }
    auto get_storage_vid(vertex_num_t local_idx) const -> vertex_id_t {
        return get_single_layer_graph().get_storage_vid(local_idx);
    }

    /** @brief Replace the mutable graph with the shared compact single-layer representation. */
    auto compact(vertex_num_t extracted_nbr_size) -> void {
        using compactor_t = typename IndexTraitsT::single_layer_graph_compactor_t;
        _compact_graph = std::make_unique<compact_graph_t>(
            compactor_t::compact_graph(get_single_layer_graph(), extracted_nbr_size));
        _single_layer_graph.reset();
    }

    auto get_compact_graph() const -> const compact_graph_t& {
        if (!_compact_graph) throw std::logic_error("Call compact() before accessing the compact graph");
        return *_compact_graph;
    }

private:
    template <typename GraphFactoryTraitsT> friend class VertexWeightAssigner;

    auto initialize_weight_groups() -> void {
        // BuildConfig's public inputs may have been edited before the index took its copy.
        _build_config.update_weight_limits();
        _vids_by_weight.resize(static_cast<std::size_t>(_build_config.max_weight) + 1);
    }

    build_config_t _build_config;
    std::vector<std::vector<vertex_id_t>> _vids_by_weight;
    std::unique_ptr<single_layer_graph_t> _single_layer_graph;
    std::unique_ptr<compact_graph_t> _compact_graph;
};

}  // namespace artea::cpu::practheo
