/*
 * @FilePath: /Artea/include/artea/cpu/concurrent_recom_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-09 18:33:13
 * @Date: 2025-11-06 21:20:10
 * @Description: 
 */

#pragma once

#include <artea/types.hpp>
#include <artea/config.hpp>
#include <artea/cpu/recommended_nn.hpp>

namespace artea {
namespace cpu {

template<typename vertex_num_t, typename >
class ConcurrentRecomNN final: public RecommendedNN {

public:
    /** @brief Construct a new Concurrent Recom NN object. */
    ConcurrentRecomNN(
        const vertex_num_t& recom_buf_size,
        bool enabled = false
    ) : 
        RecommendedNN(recom_buf_size, enabled) {
        _recom_buf.resize(recom_buf_size);
    }

    /** @brief Write an edge to the recommendation buffer. */
    auto append_edge(const vertex_id_t& src, const nbr_t& nbr) -> void override {
        _recom_buf[src].push_back(nbr);
    }

    /** @brief Get the recommended neighbors of a vertex. 
      * @param src The source vertex.
      * @return Array<nbr_t> The recommended neighbors of the source vertex.
      * @note The neighbors are sorted by distance in ascending order.
     */
    auto get_recom_nbrs(const vertex_id_t& src) -> Array<nbr_t> override {
        Array<nbr_t> arr = Array<nbr_t>::from(_recom_buf[src]);
        std::sort(arr.begin(), arr.end(), NeighborComparator);
        return arr;
    }
    
private:

    /** @brief Buffer for appending neighbors. */
    std::vector<tbb::concurrent_vector<nbr_t>> _recom_buf;

};  // class ConcurrentRecomNN

}   // namespace cpu
}   // namespace artea