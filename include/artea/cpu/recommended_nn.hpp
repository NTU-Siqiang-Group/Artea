/*
 * @FilePath: /Artea/include/artea/cpu/recommended_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-05 19:42:00
 * @Date: 2025-11-02 21:18:08
 * @Description: 
 */

/*
 * @FilePath: /Artea/include/artea/cpu/random_nn.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-05 19:18:18
 * @Date: 2025-11-02 19:41:19
 * @Description: Refactored to use generation type as a class template parameter.
 */

#pragma once

#include <random>
#include <stdexcept>
#include <type_traits>

#include <artea/types.hpp>
#include <artea/config.hpp>
#include <artea/cpu/array.hpp>
#include <artea/cpu/vector_array.hpp>

namespace artea {
namespace cpu {

template <
    typename vec_num_t, 
    typename vec_id_t = vec_num_t
>
class RecommendedNN {

public:
    

    /**
    * @brief Check if the recommended nearest neighbors generator is enabled.
    * @return true If the recommended nearest neighbors generator is enabled.
    * @return false If the recommended nearest neighbors generator is disabled.
    */
    __attribute__((always_inline))
    auto enabled() const -> bool {
        return _enabled;
    }

    /**
    * @brief Disable the recommended nearest neighbors generator.
    */
    __attribute__((always_inline))
    auto disable() -> void {
        _enabled = false;
    }

    /**
    * @brief Activate the recommended nearest neighbors generator.
    */
    __attribute__((always_inline))
    auto activate() -> void {
        _enabled = true;
    }

private:
    bool _enabled = false;

};  // class RecommendedNN

}   // namespace cpu
}   // namespace artea