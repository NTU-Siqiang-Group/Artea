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
 * @brief Probes the nearest-neighbor distance distribution from a flat graph.
 *
 * For each vertex in the graph, the nearest neighbor distance is the distance
 * of the first entry in its sorted neighbor array (nbrs_arr[0]). This class
 * collects these distances and returns the requested quantile.
 *
 * @tparam IndexTraitsT The index traits type.
 */
template <typename IndexTraitsT>
class RadiusProber {

    using vertex_num_t = typename IndexTraitsT::vertex_num_t;
    using vertex_id_t = typename IndexTraitsT::vertex_id_t;
    using distance_t = typename IndexTraitsT::distance_t;
    using nbr_arr_t = typename IndexTraitsT::nbr_arr_t;
    using nbr_arr_checker_t = typename IndexTraitsT::nbr_arr_checker_t;

public:

    struct ProbeResult {
        distance_t radius;
        vertex_num_t num_vertices;
        float quantile;
    };

    RadiusProber() = default;

    /**
     * @brief Probe a quantile of the nearest-neighbor distance distribution from a graph.
     *
     * @tparam GraphT A flat graph type with fetch_nbrs(vid) and get_num_vertices().
     * @param graph The flat graph to probe.
     * @param quantile The quantile to extract (0 = minimum, 0.5 = median). Default: 0.
     * @return ProbeResult containing the quantile radius and vertex count.
     */
    template <typename GraphT>
    auto probe(const GraphT& graph, float quantile = 0.0f) const -> ProbeResult {
        const vertex_num_t num_vertices = graph.get_num_vertices();

        #ifndef NDEBUG
        for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
            const nbr_arr_t& nbrs = graph.fetch_nbrs(vid);
            if (nbrs.empty()) {
                ARTEA_ERROR(fmt::format("[RadiusProber] vertex {} has empty neighbor array", vid));
            }
            if (!nbr_arr_checker_t::distance_order_check(nbrs)) {
                ARTEA_ERROR(fmt::format("[RadiusProber] vertex {} neighbor array is not sorted by distance", vid));
            }
        }
        #endif

        // Collect nearest-neighbor distances in parallel
        std::vector<distance_t> nn_distances(num_vertices);

        tbb::parallel_for(
            tbb::blocked_range<vertex_id_t>(0, num_vertices),
            [&](const tbb::blocked_range<vertex_id_t>& r) {
                for (vertex_id_t vid = r.begin(); vid != r.end(); ++vid) {
                    nn_distances[vid] = graph.fetch_nbrs(vid)[0].get_distance();
                }
            }
        );

        // Sort to extract quantile
        std::sort(std::execution::par, nn_distances.begin(), nn_distances.end());

        vertex_num_t index = 0;
        if (quantile > 0.0f) {
            index = static_cast<vertex_num_t>(quantile * num_vertices);
            if (index >= num_vertices) {
                index = num_vertices - 1;
            }
        }

        ProbeResult result;
        result.radius = nn_distances[index];
        result.num_vertices = num_vertices;
        result.quantile = quantile;
        return result;
    }

};  // class RadiusProber

}   // namespace cpu
}   // namespace artea
