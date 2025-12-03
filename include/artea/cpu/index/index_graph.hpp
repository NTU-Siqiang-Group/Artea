/*
 * @FilePath: /Artea/include/artea/cpu/index/index_graph.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-30 20:46:36
 * @Date: 2025-10-17 15:32:18
 * @Description:
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <variant>
#include <cstdint>
#include <vector>

#include <tbb/parallel_for.h>

#include <artea/definitions.hpp>
#include <artea/cpu/index/neighbor.hpp>
#include <artea/cpu/propagation/recommended_nn.hpp>

namespace artea {
namespace cpu {

/**
 * @brief The index graph class.
 * @tparam{vertex_num_t} The type of vertex number.
 * @tparam{vec_ele_t} The type of vector element.
 * @tparam{direction} The direction of the graph (IN, OUT, HIBRID).
 */
template <
    typename vertex_num_t,
    typename vec_ele_t,
    direction_t direction = direction_t::OUT
>
class IndexGraph {

    using distance_t = vec_ele_t;
    using vertex_id_t = vertex_num_t;
    using nbr_t = Neighbor<vertex_num_t, vec_ele_t>;
    using nbr_arr_t = std::vector<nbr_t>;

public:
    /**
     * @brief Construct a new Index Graph object with aligned memory.
     * @param num_vertices The total number of vertices in the graph.
     * @param edges_limit The maximum number of (in-/out-) neighbors per vertex.
     */
    IndexGraph(
        const vertex_num_t num_vertices,
        const vertex_num_t edges_limit
    ) :
        _num_vertices(num_vertices),
        _edges_limit(edges_limit)
    {
        if constexpr (direction == direction_t::IN or direction == direction_t::HIBRID) {
            _in_nbrs_arr.resize(num_vertices);
            for (vertex_num_t i = 0; i < num_vertices; ++i) {
                _in_nbrs_arr[i] = nbr_arr_t();
                _in_nbrs_arr[i].reserve(edges_limit);
            }
        }
        if constexpr (direction == direction_t::OUT or direction == direction_t::HIBRID) {
            _out_nbrs_arr.resize(num_vertices);
            for (vertex_num_t i = 0; i < num_vertices; ++i) {
                _out_nbrs_arr[i] = nbr_arr_t();
                _out_nbrs_arr[i].reserve(edges_limit);
            }
        }
    }

    ~IndexGraph() = default;

    // default move constructor and assignment
    IndexGraph(IndexGraph&&) noexcept = default;
    IndexGraph& operator=(IndexGraph&&) noexcept = default;

    // Copying is deleted
    IndexGraph(const IndexGraph&) = delete;
    IndexGraph& operator=(const IndexGraph&) = delete;

    // --- Accessors ---

    __attribute__((always_inline))
    auto get_num_vertices() const -> vertex_num_t {
        return _num_vertices;
    }

    __attribute__((always_inline))
    auto get_edges_limit() const -> vertex_num_t {
        return _edges_limit;
    }

    template <direction_t fetch_direction>
    __attribute__((always_inline))
    auto get_nbrs_arr() -> std::vector<nbr_arr_t>& {
        static_assert(
            fetch_direction == direction_t::IN or fetch_direction == direction_t::OUT,
            "fetch_direction must be IN or OUT"
        );

        if constexpr (fetch_direction == direction_t::IN) {
            return _in_nbrs_arr;
        }
        else if constexpr (fetch_direction == direction_t::OUT) {
            return _out_nbrs_arr;
        }
    }

    template <direction_t fetch_direction>
    __attribute__((always_inline))
    auto get_nbrs_arr() const -> const std::vector<nbr_arr_t>& {
        static_assert(
            fetch_direction == direction_t::IN or fetch_direction == direction_t::OUT,
            "fetch_direction must be IN or OUT"
        );
        if constexpr (fetch_direction == direction_t::IN) {
            return _in_nbrs_arr;
        }
        else if constexpr (fetch_direction == direction_t::OUT) {
            return _out_nbrs_arr;
        }
    }

    // // --- Graph Operations ---
    // append_nbr operations is delegated to the RecommendedNN.

    // virtual auto append_nbr(const vertex_id_t& src, const nbr_t& nbr) -> void = 0;

    template <direction_t fetch_direction>
    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t& src) -> nbr_arr_t& {
        static_assert(
            fetch_direction == direction_t::IN or fetch_direction == direction_t::OUT,
            "fetch_direction must be IN or OUT"
        );

        if constexpr (fetch_direction == direction_t::IN) {
            return _in_nbrs_arr[src];
        }
        else if constexpr (fetch_direction == direction_t::OUT) {
            return _out_nbrs_arr[src];
        }
    }

    template <direction_t fetch_direction>
    __attribute__((always_inline))
    auto fetch_nbrs(const vertex_id_t& src) const -> const nbr_arr_t& {
        static_assert(
            fetch_direction == direction_t::IN or fetch_direction == direction_t::OUT,
            "fetch_direction must be IN or OUT"
        );

        if constexpr (fetch_direction == direction_t::IN) {
            return _in_nbrs_arr[src];
        }
        else if constexpr (fetch_direction == direction_t::OUT) {
            return _out_nbrs_arr[src];
        }
    }

protected:
    /** @brief Number of vertices in the graph. */
    vertex_num_t _num_vertices;

    /** @brief Number of neighbors per vertex. */
    vertex_num_t _edges_limit;

    /** @brief Array of in-neighbors for each vertex. */
    std::vector<nbr_arr_t> _in_nbrs_arr;

    /** @brief Array of out-neighbors for each vertex. */
    std::vector<nbr_arr_t> _out_nbrs_arr;

};  // class IndexGraph

}   // namespace cpu
}   // namespace artea
