// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <valarray>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/partitioner.h>

#include <artea/cpu/index/practheo/index_structure.hpp>

namespace artea::cpu::practheo {

/** @brief Assign vertex weights and group their global IDs during Practheo construction. */
template <typename GraphFactoryTraitsT>
class VertexWeightAssigner {
    using this_index_t = typename GraphFactoryTraitsT::practheo::index_t;
    using vector_array_t = typename GraphFactoryTraitsT::vector_array_t;
    using vec_num_t = typename GraphFactoryTraitsT::vec_num_t;
    using vertex_id_t = typename GraphFactoryTraitsT::vertex_id_t;

public:
    /** @brief Assign capped geometric weights from zero through the configured max_weight to a batch.
     * Row i maps to global VID first_vid + i. Submit disjoint batches in increasing VID order.
     */
    static auto assign_vertex_weight(
        this_index_t& index,
        const vector_array_t& batch_vecs,
        vertex_id_t first_vid
    ) -> void {
        const std::size_t batch_count = batch_vecs.get_num_vecs();
        if (batch_count == 0) return;
        if (index.build_config().max_weight <= std::numeric_limits<std::uint8_t>::max()) {
            assign_batch<std::uint8_t>(index, batch_count, first_vid);
        } else {
            assign_batch<vec_num_t>(index, batch_count, first_vid);
        }
    }

private:
    template <typename WeightT>
    static auto assign_batch(this_index_t& index, std::size_t batch_count, vertex_id_t first_vid) -> void {
        auto& weight_groups = index._vids_by_weight;
        std::vector<std::size_t> old_group_sizes(weight_groups.size());
        for (std::size_t i = 0; i < weight_groups.size(); ++i) {
            const auto& weight_group = weight_groups[i];
            if (!weight_group.empty() && weight_group.back() >= first_vid) {
                throw std::invalid_argument("Practheo weight batches must be disjoint and in increasing VID order");
            }
            old_group_sizes[i] = weight_group.size();
        }

        const auto max_weight = index.build_config().max_weight;
        // promotion_threshold[i] is the probability that a vertex's weight exceeds i.
        const double promo_prob = index.build_config().weight_promotion_probability;
        std::vector<double> promotion_threshold(max_weight, promo_prob);
        for (std::size_t i = 1; i < max_weight; ++i) {
            promotion_threshold[i] = promotion_threshold[i - 1] * promo_prob;
        }

        const tbb::blocked_range<std::size_t> rows(0, batch_count);
        using group_counts_t = std::valarray<std::size_t>;
        const group_counts_t zero_counts(std::size_t{0}, weight_groups.size());
        auto weights = std::make_unique_for_overwrite<WeightT[]>(batch_count);
        const auto batch_group_sizes = tbb::parallel_reduce(rows, zero_counts,
            [&](const auto& range, group_counts_t counts) {
                for (auto i = range.begin(); i != range.end(); ++i) {
                    const double draw = uniform_draw(first_vid + i);
                    WeightT weight = 0;
                    if constexpr (std::is_same_v<WeightT, std::uint8_t>) {
                        while (weight < max_weight && draw < promotion_threshold[weight]) ++weight;
                    } else {
                        const auto stop = std::lower_bound(promotion_threshold.begin(), promotion_threshold.end(),
                                                           draw, std::greater<>{});
                        weight = static_cast<WeightT>(stop - promotion_threshold.begin());
                    }
                    weights[i] = weight;
                    ++counts[weight];
                }
                return counts;
            }, std::plus<group_counts_t>{}, tbb::static_partitioner{});

        std::vector<vertex_id_t*> group_starts(weight_groups.size());
        try {
            for (std::size_t i = 0; i < weight_groups.size(); ++i) {
                auto& weight_group = weight_groups[i];
                weight_group.resize(old_group_sizes[i] + batch_group_sizes[i]);
                group_starts[i] = weight_group.data() + old_group_sizes[i];
            }
            // The number of earlier rows with this weight gives each VID its ordered slot.
            tbb::parallel_scan(rows, zero_counts,
                [&](const auto& range, group_counts_t counts, bool write) {
                    for (auto i = range.begin(); i != range.end(); ++i) {
                        const auto weight = weights[i];
                        const auto position = counts[weight]++;
                        if (write) group_starts[weight][position] = static_cast<vertex_id_t>(first_vid + i);
                    }
                    return counts;
                }, std::plus<group_counts_t>{});
        } catch (...) {
            for (std::size_t i = 0; i < weight_groups.size(); ++i) weight_groups[i].resize(old_group_sizes[i]);
            throw;
        }
    }

    // SplitMix64 gives each global VID a reproducible uniform draw in [0, 1).
    static auto uniform_draw(std::uint64_t vid) noexcept -> double {
        auto bits = vid + UINT64_C(0x9e3779b97f4a7c15);
        bits = (bits ^ (bits >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        bits = (bits ^ (bits >> 27)) * UINT64_C(0x94d049bb133111eb);
        bits ^= bits >> 31;
        return static_cast<double>(bits >> 11) * 0x1.0p-53;
    }
};

}  // namespace artea::cpu::practheo
