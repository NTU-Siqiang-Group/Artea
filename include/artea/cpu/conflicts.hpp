/*
 * @FilePath: /Artea/include/artea/cpu/conflicts.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-30 20:46:46
 * @Date: 2025-11-11 16:50:41
 * @Description:
 */

#pragma once

#include <cstdint>
#include <utility>

#include <range/v3/view/zip.hpp>

#include <artea/definitions.hpp>
#include <artea/cpu/index/neighbor.hpp>
#include <artea/cpu/containers/array.hpp>
#include <artea/cpu/utils/simd_distance.hpp>

namespace artea {
namespace cpu {

template <
    typename vertex_num_t,
    typename vec_ele_t,
    DistanceMetrics dist_type = DistanceMetrics::EUCLIDEAN
>
class Conflicts {

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = Array<nbr_t>;
    using simd_distance_t = SimdDistance<vec_ele_t, dist_type>;

public:

    Conflicts(
        const simd_distance_t& dist_computer,
        const VectorArray<vertex_num_t, vec_ele_t>& vecs_arr
    ) : _simd_distance(dist_computer), _vecs_arr(vecs_arr) {}

    /**
     * @brief Check if the neighbor is in conflict with the reserved neighbors.
     * @param nbr The neighbor to be checked.
     * @param is_org_nbr Whether the neighbor `nbr` is an original neighbor.
     * @param reserved_nbrs The array of reserved neighbors.
     * @param from_org_flags The array of original flags.
     * @return std::pair<bool, distance_t> A pair of boolean and distance.
     *      The boolean indicates whether the neighbor is accepted.
     *      - If the neighbor is not in conflict with any reserved neighbor, return true. And returned distance is the distance to the central vertex.
     *      - If the neighbor is in conflict with any reserved neighbor, return false. And returned distance is the distance to the conflict reserved neighbor.
     */
    auto rng_strategy(
        nbr_t& nbr,
        bool is_org_nbr,
        const nbr_arr_t& reserved_nbrs,
        const Array<bool>& from_org_flags
    ) -> std::tuple<bool, vertex_id_t, distance_t> {
        vec_ele_t* vec = _vecs_arr.get_vec(nbr.dest);
        vertex_num_t num_reserved_nbrs = reserved_nbrs.size();

        for (vertex_id_t i = 0; i < num_reserved_nbrs; ++i) {
            // this unique implementation is not good
            if ((is_org_nbr and from_org_flags[i]) or reserved_nbrs[i].dest == nbr.dest) {
                continue;
            }
            vec_ele_t* rvec = _vecs_arr.get_vec(reserved_nbrs[i].dest);
            distance_t dist_to_reserved = _simd_distance(vec, rvec);
            if (dist_to_reserved < nbr.distance) {
                return std::make_tuple(false, reserved_nbrs[i].dest, dist_to_reserved);
            }
        }

        return std::make_tuple(true, nbr.dest, nbr.distance);
    }

        /* ------ Deprecated Parallel Implementation ------ */
        /** @note The parallel implementation is not applicable in this algorithm **/

        // if constexpr (not enable_internal_parallel) {
        // }
        // else constexpr {
        //     tbb::parallel_for(tbb::blocked_range<vertex_id_t>(0, num_reserved_nbrs),
        //         [&](tbb::blocked_range<vertex_id_t> r) {
        //         for (vertex_id_t i = r.begin(); i < r.end(); ++i) {
        //             if ((from_org_flags[i] and is_org_nbr) or reserved_nbrs[i].dest == nbr.dest) {
        //                 continue;
        //             }
        //             vec_ele_t* rvec = _vecs_arr.get_vec(reserved_nbrs[i].dest);
        //             distance_t dist_to_reserved = _simd_distance(vec, rvec);
        //             if (dist_to_reserved < nbr.distance) {
        //                 return std::make_pair(false, dist_to_reserved);
        //             }
        //         }
        //     });
        // }

private:

    const simd_distance_t& _dist_computer;

    const VectorArray<vertex_num_t, vec_ele_t>& _vecs_arr;

};  // class Conflicts

}   // namespace cpu
}   // namespace artea