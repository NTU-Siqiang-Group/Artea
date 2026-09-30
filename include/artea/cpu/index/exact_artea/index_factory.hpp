// Copyright 2026 Weitang Ye
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <artea/cpu/index/exact_artea/index_structure.hpp>
#include <artea/cpu/index/exact_artea/rnets_factory.hpp>
#include <artea/cpu/refiner/updaters/pruning_updater.hpp>

namespace artea {
namespace cpu {
namespace exact_artea {

/** @brief Exact FPS, layer-by-layer construction and lossless hierarchical CSR merge. */
template <typename GraphFactoryTraitsT>
class IndexFactory {
    using this_index_t = typename GraphFactoryTraitsT::exact_artea::index_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;
    using vertex_num_t = typename GraphFactoryTraitsT::vertex_num_t;
    using layer_id_t = typename GraphFactoryTraitsT::layer_id_t;
    using distance_t = typename GraphFactoryTraitsT::distance_t;
    using ratio_t = typename GraphFactoryTraitsT::ratio_t;
    using dist_func_t = typename GraphFactoryTraitsT::dist_func_t;
    using nbr_t = typename GraphFactoryTraitsT::nbr_t;
    using nbr_arr_t = typename GraphFactoryTraitsT::nbr_arr_t;
    using hierarchical_graph_t = typename this_index_t::hierarchical_graph_t;

public:
    using upper_pruning_updater_t = CandidatePruningUpdater<
        GraphFactoryTraitsT, PruningConditionT::scaled_ineq>;
    using bottom_pruning_updater_t = CandidatePruningUpdater<
        GraphFactoryTraitsT, PruningConditionT::scaled_shifted_ineq>;

    struct PruningUpdaters {
        upper_pruning_updater_t upper;
        bottom_pruning_updater_t bottom;
    };

    /** @brief Bind both paper pruning rules; vectors and distance must outlive the result.
     * Neither updater reads graph capacities or skips OLD/OLD pairs. Inputs must
     * be sorted as returned by get_arc_candidates. Empty inputs are supported.
     */
    static auto make_pruning_updaters(const this_index_t& index, const dist_func_t& dist_func)
        -> PruningUpdaters {
        const auto& config = index.pruning_config();
        const double alpha = config.scale_coeffs();
        const double tau = config.shifted_coeffs();
        const double rho = config.l0_min_distance();
        if (!std::isfinite(alpha) || alpha <= 1 || !std::isfinite(tau) || tau < 0 ||
            !std::isfinite(rho) || rho <= 0) {
            throw std::invalid_argument("Exact pruning requires alpha > 1, tau >= 0 and rho > 0");
        }
        // The shared updater subtracts a bare shift after division by alpha.
        const double shift = (alpha + 1) * tau * rho / alpha;
        if (!std::isfinite(shift) || shift > std::numeric_limits<ratio_t>::max()) {
            throw std::overflow_error("Exact pruning shift is not representable");
        }
        return {{dist_func, index.get_base_vecs(), static_cast<ratio_t>(alpha)},
                {dist_func, index.get_base_vecs(), static_cast<ratio_t>(alpha),
                 static_cast<ratio_t>(shift)}};
    }

    /**
     * @brief Scan the supplied layer for candidates within Delta_h * R_h of vid.
     *
     * Delta_h is rgraph_config.aspect_ratio_constraint(h, pruning_config);
     * R_h is rgraph_config.radius_at(h). Include distances equal to the cutoff,
     * exclude vid itself, and retain other vertices with identical coordinates.
     * layer_vids supplies the layer membership as unique global vector IDs;
     * it may be in any order and may include vid. No neighbor capacity applies.
     * The index must have configs, but layer assignment / edge construction
     * need not have started. Membership comes from layer_vids, not graph storage.
     *
     * Scan and sort run sequentially on the calling thread, evaluating every
     * other supplied vector exactly once. Sort by (distance, vertex ID) for
     * deterministic results. There is no approximate search, sampling,
     * candidate limit or epsilon.
     *
     * @param dist_func Distance functor using the same units as
     *                  radius_at(h), consistent with RNetsFactory::fps_generator.
     *                  Exactness is with respect to its returned distances.
     * @return NEW neighbor records sorted by distance, with ties broken by ID.
     * @throws std::out_of_range If vid or a layer ID is outside the base dataset.
     * @throws std::domain_error If an evaluated distance is negative/nonfinite.
     * @throws std::overflow_error If the distance cutoff is unrepresentable.
     * Configuration and layer errors propagate from aspect_ratio_constraint.
     * The index and its immutable base vectors are never modified.
     */
    static auto get_arc_candidates(
        const this_index_t& index,
        const vertex_id_t vid,
        const layer_id_t h,
        const std::span<const vertex_id_t> layer_vids,
        const dist_func_t& dist_func
    ) -> std::vector<nbr_t> {
        const auto& vectors = index.get_base_vecs();
        const std::size_t count = vectors.get_num_vecs();
        if (vid >= count) {
            throw std::out_of_range("ARC candidate vertex is outside the base dataset");
        }
        const auto& config = index.rgraph_config();
        const double arc = config.aspect_ratio_constraint(h, index.pruning_config());
        const double cutoff = arc * static_cast<double>(config.radius_at(h));
        if (!std::isfinite(cutoff) || cutoff <= 0 ||
            cutoff > std::numeric_limits<distance_t>::max()) {
            throw std::overflow_error("ARC distance cutoff is not representable");
        }

        std::vector<nbr_t> candidates;
        const auto* center = vectors.get(vid);
        for (const auto candidate : layer_vids) {
            if (candidate >= count) {
                throw std::out_of_range("ARC layer vertex is outside the base dataset");
            }
            if (candidate == vid) continue;
            const distance_t distance = dist_func(center, vectors.get(candidate));
            if (!std::isfinite(distance) || distance < distance_t(0)) {
                throw std::domain_error("ARC scan requires finite nonnegative distances");
            }
            if (static_cast<double>(distance) <= cutoff) {
                candidates.emplace_back(candidate, distance, true);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const nbr_t& a, const nbr_t& b) {
            return a.get_distance() < b.get_distance() ||
                   (a.get_distance() == b.get_distance() && a.get_vid() < b.get_vid());
        });
        return candidates;
    }

    /** @brief ARC scan followed by complete layer-specific pruning, with no degree cap.
     * Returned records are OLD and sorted. This does not write to fixed-size
     * graph slots: the caller owns the complete variable-length neighbor list.
     */
    static auto prune_arc_candidates(
        const this_index_t& index,
        const vertex_id_t vid,
        const layer_id_t h,
        const std::span<const vertex_id_t> layer_vids,
        const dist_func_t& dist_func
    ) -> std::vector<nbr_t> {
        auto candidates = get_arc_candidates(index, vid, h, layer_vids, dist_func);
        const auto updaters = make_pruning_updaters(index, dist_func);
        if (h == 0) updaters.bottom(candidates);
        else updaters.upper(candidates);
        return candidates;
    }

    /** @brief Construct one layer in parallel across vertices, publishing only on success.
     * Each vertex's ARC scan and pruning remain sequential. All rows are built
     * in private buffers; a failing distance calculation leaves the layer intact.
     */
    static auto build_layer(this_index_t& index, layer_id_t h, const dist_func_t& dist_func) -> void {
        const auto vids = index.layer_vids(h);
        const auto updaters = make_pruning_updaters(index, dist_func);
        std::vector<nbr_arr_t> rows(vids.size());
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, vids.size(), 16),
            [&](const tbb::blocked_range<std::size_t>& range) {
                for (std::size_t i = range.begin(); i != range.end(); ++i) {
                    auto candidates = get_arc_candidates(index, vids[i], h, vids, dist_func);
                    if (h == 0) updaters.bottom(candidates);
                    else updaters.upper(candidates);
                    rows[i] = std::move(candidates);
                }
            });
        index.get_layer_graph(h).get_nbrs_arr().swap(rows);
    }

    /** @brief Build every existing FPS layer, in bottom-to-top order. */
    static auto build_layers(this_index_t& index, const dist_func_t& dist_func) -> void {
        if (!index.has_layers()) throw std::logic_error("Call fps_generator before build_layers");
        for (layer_id_t h = 0; h < index.get_num_layers(); ++h) build_layer(index, h, dist_func);
    }

    /** @brief Merge every layer and every neighbor into CSR, preserving their order.
     * No sparse-top trimming, degree truncation, or distance recomputation occurs.
     * Layer graphs remain available. The published CSR is a snapshot; later row
     * mutations require merging again. Failure leaves any previous snapshot intact.
     */
    static auto merge_layers(this_index_t& index) -> void {
        if (!index.has_layers()) throw std::logic_error("Call fps_generator before merge_layers");
        const vertex_num_t count = index.get_num_vertices();
        const layer_id_t top = static_cast<layer_id_t>(index.get_num_layers() - 1);
        std::vector<std::size_t> group_counts(index.get_num_layers(), 0);
        for (vertex_id_t vid = 0; vid < count; ++vid) ++group_counts[index.get_highest_level_id(vid)];

        // Legacy capacity metadata is unused in this non-persistent exact path.
        // Real row lengths alone determine CSR allocation.
        hierarchical_graph_t graph(top, 0, 0, count, group_counts);
        auto& buckets = graph.get_vids_by_highest_level_mut();
        auto& vertex_info = graph.get_vertex_info_table_mut();
        for (layer_id_t h = 0; h <= top; ++h) buckets[h].reserve(group_counts[h]);
        for (vertex_id_t vid = 0; vid < count; ++vid) {
            const auto highest = index.get_highest_level_id(vid);
            vertex_info[vid] = {highest, static_cast<vertex_id_t>(buckets[highest].size())};
            buckets[highest].push_back(vid);
        }
        tbb::parallel_for(tbb::blocked_range<vertex_id_t>(0, count, 64),
            [&](const tbb::blocked_range<vertex_id_t>& range) {
                for (vertex_id_t vid = range.begin(); vid != range.end(); ++vid) {
                    const auto& info = vertex_info[vid];
                    auto row_counts = graph.get_nbr_offsets_mut(info.highest_level);
                    for (layer_id_t h = 0; h <= info.highest_level; ++h) {
                        const auto neighbors = index.fetch_level_nbrs(vid, h);
                        for (const auto& neighbor : neighbors) {
                            const auto id = neighbor.get_vid();
                            if (neighbor.is_invalid() || id >= count || index.get_highest_level_id(id) < h) {
                                throw std::invalid_argument("Exact merge found a neighbor outside its layer");
                            }
                        }
                        const std::size_t row = static_cast<std::size_t>(info.local_vid) *
                            (static_cast<std::size_t>(info.highest_level) + 1) + info.highest_level - h;
                        row_counts[row] = neighbors.size();
                    }
                }
            });
        graph.allocate_neighbors_from_row_counts();
        tbb::parallel_for(tbb::blocked_range<vertex_id_t>(0, count, 64),
            [&](const tbb::blocked_range<vertex_id_t>& range) {
                for (vertex_id_t vid = range.begin(); vid != range.end(); ++vid) {
                    for (layer_id_t h = 0; h <= vertex_info[vid].highest_level; ++h) {
                        const auto source = index.fetch_level_nbrs(vid, h);
                        auto target = graph.fetch_level_nbrs_mut(vid, h);
                        for (std::size_t i = 0; i < source.size(); ++i) target[i] = source[i].get_vid();
                    }
                }
            });
        graph.set_entry_point_vid(index.entry_point_vid());
        index._hierarchical_graph.emplace(std::move(graph));
    }

    /** @brief Full exact construction. The prepared index must not already have layers.
     * dist_func must be thread-safe and use the same metric units as the radii.
     */
    static auto build(this_index_t& index, const dist_func_t& dist_func) -> void {
        RNetsFactory<GraphFactoryTraitsT>::fps_generator(index, dist_func);
        build_layers(index, dist_func);
        merge_layers(index);
    }
};

}   // namespace exact_artea
}   // namespace cpu
}   // namespace artea
