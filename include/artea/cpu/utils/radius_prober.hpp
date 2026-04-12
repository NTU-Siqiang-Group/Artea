/*
 * @FilePath: /Artea/include/artea/cpu/utils/radius_prober.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Graph-based radius prober: estimates nearest-neighbor distance quantiles from a flat graph.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <execution>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Probes the neighbor distance distribution from a flat graph.
 *
 * For each vertex, reads the distance at a specified neighbor rank in its sorted
 * neighbor array, then returns the requested quantile over all vertices.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class RadiusProber {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using bnbr_arr_t = typename IndexTraitsT::bnbr_arr_t;
    using nbr_arr_checker_t = typename IndexTraitsT::nbr_arr_checker_t;

    static constexpr distance_t max_distance = IndexTraitsT::max_distance;

public:

    struct ProbeResult {
        distance_t radius;
        vertex_num_t num_vertices;
        float quantile;
    };

    RadiusProber() = default;

    /**
     * @brief Probe a quantile of the neighbor distance distribution from a graph.
     *
     * @tparam GraphT A flat graph type with fetch_nbrs(vid) and get_num_vertices().
     * @param graph The flat graph to probe.
     * @param nbr_rank The neighbor rank to probe (1-indexed: 1 = nearest neighbor,
     *                 2 = second nearest, etc.). Vertices with fewer than nbr_rank
     *                 neighbors are skipped. Default: 1.
     * @param quantile The quantile to extract (0 = minimum, 0.5 = median). Default: 0.
     * @return ProbeResult containing the quantile radius and vertex count.
     *         Returns max_distance if no vertex has enough neighbors.
     */
    template <typename GraphT>
    auto probe(const GraphT& graph, vertex_num_t nbr_rank = 1, float quantile = 0.0f) const -> ProbeResult {
        const vertex_num_t num_vertices = graph.get_num_vertices();

        #ifndef NDEBUG
        if (nbr_rank == 0) {
            ARTEA_ERROR("[RadiusProber] nbr_rank is 1-indexed and must be >= 1");
        }
        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    const bnbr_arr_t& nbrs = graph.fetch_nbrs(vid);
                    if (nbrs.empty()) {
                        ARTEA_ERROR(fmt::format("[RadiusProber] vertex {} has empty neighbor array", vid));
                    }
                    if (!nbr_arr_checker_t::distance_order_check(nbrs)) {
                        ARTEA_ERROR(fmt::format("[RadiusProber] vertex {} neighbor array is not sorted by distance", vid));
                    }
                }
            }
        );
        #endif

        const vertex_num_t nbr_index = nbr_rank - 1;

        // Collect distances at the specified rank in parallel.
        // Vertices with fewer than nbr_rank neighbors get max_distance as sentinel.
        std::vector<distance_t> distances(num_vertices);

        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    const bnbr_arr_t& nbrs = graph.fetch_nbrs(vid);
                    distances[vid] = (nbrs.size() > nbr_index)
                        ? nbrs[nbr_index].get_distance()
                        : max_distance;
                }
            }
        );

        // Partition: move valid distances (< max_distance) to front, O(N) parallel.
        auto valid_end = std::remove_if(std::execution::par,
            distances.begin(), distances.end(),
            [](distance_t d) { return d == max_distance; });
        const vertex_num_t num_valid = static_cast<vertex_num_t>(
            std::distance(distances.begin(), valid_end));

        #ifndef NDEBUG
        if (num_valid < num_vertices) {
            ARTEA_WARN(fmt::format(
                "[RadiusProber] {} / {} vertices have fewer than {} neighbors",
                num_vertices - num_valid, num_vertices, nbr_rank));
        }
        #endif

        if (num_valid == 0) {
            return ProbeResult{max_distance, 0, quantile};
        }

        // Compute the target index for the requested quantile.
        vertex_num_t target_index = 0;
        if (quantile > 0.0f) {
            target_index = static_cast<vertex_num_t>(quantile * num_valid);
            if (target_index >= num_valid) {
                target_index = num_valid - 1;
            }
        }

        // O(N) partial ordering: only position the element at `target_index` correctly.
        std::nth_element(std::execution::par,
            distances.begin(), distances.begin() + target_index, valid_end);

        ProbeResult result;
        result.radius = distances[target_index];
        result.num_vertices = num_valid;
        result.quantile = quantile;
        return result;
    }

};  // class RadiusProber

}   // namespace cpu
}   // namespace artea
