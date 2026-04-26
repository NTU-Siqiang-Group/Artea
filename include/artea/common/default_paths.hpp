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

/*
 * @FilePath: /Artea/include/artea/common/default_paths.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Shared default path helpers consumed by argparse
 *               default_value(...) calls across apps / unit_tests /
 *               micro_benchmarks / vldb27-exp.
 */

#pragma once

#include <cstdlib>
#include <string>

namespace artea {

/**
 * @brief Default path to the dataset configuration JSON.
 *        Resolves to "$HOME/artea-benchmark/configs/datasets.json"
 *        when HOME is set; falls back to "./configs/datasets.json"
 *        otherwise.
 */
inline auto default_dataset_config_path() -> std::string {
    const char* home_dir = std::getenv("HOME");
    if (home_dir == nullptr || home_dir[0] == '\0') {
        return "./configs/datasets.json";
    }
    return std::string(home_dir) + "/artea-benchmark/configs/datasets.json";
}

/**
 * @brief Expand a leading "$HOME" or "~" in @p path to the value of
 *        the HOME environment variable. Returns @p path unchanged when
 *        no leading placeholder is present, or when HOME is not set.
 *        Used to let JSON-stored paths (e.g. dataset config root_dir)
 *        stay portable across machines.
 */
inline auto expand_home(const std::string& path) -> std::string {
    if (path.empty()) return path;

    auto resolve_home = []() -> const char* {
        const char* home_dir = std::getenv("HOME");
        return (home_dir == nullptr || home_dir[0] == '\0') ? nullptr : home_dir;
    };

    if (path[0] == '~' && (path.size() == 1 || path[1] == '/')) {
        const char* home_dir = resolve_home();
        if (home_dir == nullptr) return path;
        return std::string(home_dir) + path.substr(1);
    }

    static constexpr const char* home_token = "$HOME";
    static constexpr std::size_t home_token_len = 5;
    if (path.size() >= home_token_len &&
        path.compare(0, home_token_len, home_token) == 0 &&
        (path.size() == home_token_len || path[home_token_len] == '/'))
    {
        const char* home_dir = resolve_home();
        if (home_dir == nullptr) return path;
        return std::string(home_dir) + path.substr(home_token_len);
    }

    return path;
}

}   // namespace artea
