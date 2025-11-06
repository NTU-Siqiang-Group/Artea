#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <memory>
#include <cstring>

#include <tbb/concurrent_vector.h>

#include <artea/config.hpp>
#include <artea/cpu/index_graph.hpp>
#include <artea/cpu/array.hpp>

namespace artea {
namespace cpu {

template <typename vertex_num_t, typename vec_ele_t>
class ConcurrentGraph: public IndexGraph<vertex_num_t, vec_ele_t> {

public:
    using nbr_t = typename IndexGraph<vertex_num_t, vec_ele_t>::nbr_t;
    using nbr_arr_t = typename IndexGraph<vertex_num_t, vec_ele_t>::nbr_arr_t;
    using vertex_id_t = typename IndexGraph<vertex_num_t, vec_ele_t>::vertex_id_t;

    ConcurrentGraph(
        const vertex_num_t& num_vertices, 
        const vertex_num_t& num_nbrs_per_vertex,
        const vertex_num_t append_buf_size
    ) : IndexGraph<vertex_num_t, vec_ele_t>(num_vertices, num_nbrs_per_vertex)
    {
        _fetch_buf.resize(num_vertices);
        _append_buf.resize(num_vertices);
        for (vertex_id_t vid = 0; vid < num_vertices; ++vid) {
            _append_buf[vid].reserve(append_buf_size);
            _fetch_buf[vid] = nbr_arr_t(num_nbrs_per_vertex); 
        }
    }
    
    // Using explicit return types to avoid auto deduction issues with override
    void append_nbr(const vertex_id_t& src, const nbr_t& nbr) {
        _append_buf[src].push_back(nbr);
    }

    nbr_arr_t& fetch_nbrs(const vertex_id_t& src) {
        return _fetch_buf[src];
    }
    
    void merge_nbrs() {
        auto nbr_comparator = [](const nbr_t& a, const nbr_t& b) { return a.distance < b.distance; };

        #pragma omp parallel for
        for (vertex_id_t v = 0; v < this->_num_vertices; ++v) {
            auto& append_vec_tbb = this->_append_buf[v];
            auto& fetch_arr = this->_fetch_buf[v];
            if (append_vec_tbb.empty()) continue;

            std::vector<nbr_t> append_vec_std(append_vec_tbb.begin(), append_vec_tbb.end());
            append_vec_tbb.clear();

            std::sort(append_vec_std.begin(), append_vec_std.end(), nbr_comparator);

            std::vector<nbr_t> merged_result;
            merged_result.reserve(fetch_arr.get_num_element() + append_vec_std.size());
            
            std::merge(
                append_vec_std.cbegin(), append_vec_std.cend(),
                fetch_arr.data(), fetch_arr.data() + fetch_arr.get_num_element(),
                std::back_inserter(merged_result),
                nbr_comparator
            );
            
            const size_t num_nbrs_capacity = static_cast<size_t>(this->_num_nbrs_per_vertex);
            const size_t final_size = std::min(merged_result.size(), num_nbrs_capacity);

            fetch_arr.set_num_element(final_size);
            if (final_size > 0) {
                std::memcpy(fetch_arr.data(), merged_result.data(), final_size * sizeof(nbr_t));
            }
        }
    }
    
private:
    std::vector<tbb::concurrent_vector<nbr_t>> _append_buf;
    std::vector<nbr_arr_t> _fetch_buf;
};

}   // namespace cpu
}   // namespace artea