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
 * @FilePath: /Artea/include/artea/cpu/utils/index_register_util.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Utility for registering built index binaries in a JSON registry
 */

#pragma once

#include <filesystem>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include <artea/common/logger.hpp>

namespace artea {
namespace cpu {

/**
 * @brief Registry manager for index binary files.
 *
 * This class maintains a JSON registry of all built index files with their
 * construction parameters. It supports multiple algorithms with different parameter sets.
 */
class IndexRegisterUtil {
public:
    /**
     * @brief Register or update an index binary in the registry JSON file.
     *
     * This function maintains a JSON registry of all built index files with their
     * construction parameters. It supports multiple algorithms with different parameter sets.
     *
     * @param registry_path Path to the registry JSON file (will be created if not exists)
     * @param algorithm Algorithm name (e.g., "conv_graph", "hnsw", "nsg")
     * @param params JSON object containing all construction parameters
     * @param file_path Path to the generated index file (relative to project root directory)
     * @return true if registration succeeded, false otherwise
     *
     * @note The function checks if an entry with the same algorithm and parameters
     *       already exists. If found, it updates the file_path; otherwise, it adds
     *       a new entry.
     * @note The file_path should be relative to the project root directory for consistency
     *       across different usage contexts (e.g., benchmarking, testing).
     */
    static auto register_index(
        const std::filesystem::path& registry_path,
        const std::string& algorithm,
        const nlohmann::json& params,
        const std::string& file_path
    ) -> bool {
        nlohmann::json registry = load_registry(registry_path);

        // Create entry for this index
        nlohmann::json entry;
        entry["algorithm"] = algorithm;
        entry["params"] = params;
        entry["file_path"] = file_path;

        // Check if entry with same algorithm and parameters already exists
        bool found = false;
        for (auto& existing_entry : registry) {
            if (is_same_entry(existing_entry, algorithm, params)) {
                // Update existing entry
                existing_entry["file_path"] = file_path;
                found = true;
                ARTEA_INFO(fmt::format("Updated existing registry entry for algorithm '{}'", algorithm));
                break;
            }
        }

        if (!found) {
            registry.push_back(entry);
            ARTEA_INFO(fmt::format("Added new registry entry for algorithm '{}'", algorithm));
        }

        return save_registry(registry_path, registry);
    }

    /**
     * @brief Find an index binary in the registry by algorithm and parameters.
     *
     * @param registry_path Path to the registry JSON file
     * @param algorithm Algorithm name to search for
     * @param params Parameters to match
     * @return File path if found, empty string otherwise
     */
    static auto find_index(
        const std::filesystem::path& registry_path,
        const std::string& algorithm,
        const nlohmann::json& params
    ) -> std::string {
        if (!std::filesystem::exists(registry_path)) {
            return "";
        }

        nlohmann::json registry = load_registry(registry_path);

        for (const auto& entry : registry) {
            if (is_same_entry(entry, algorithm, params) && entry.contains("file_path")) {
                return entry["file_path"].get<std::string>();
            }
        }

        return "";
    }

private:
    /**
     * @brief Load registry from file.
     * @param registry_path Path to the registry JSON file
     * @return JSON array (empty if file doesn't exist or is invalid)
     */
    static auto load_registry(const std::filesystem::path& registry_path) -> nlohmann::json {
        if (!std::filesystem::exists(registry_path)) {
            return nlohmann::json::array();
        }

        std::ifstream registry_file(registry_path);
        if (!registry_file.is_open()) {
            ARTEA_ERROR(fmt::format("Failed to open registry file: {}", registry_path.string()));
            return nlohmann::json::array();
        }

        nlohmann::json registry;
        try {
            registry_file >> registry;
            if (!registry.is_array()) {
                ARTEA_ERROR(fmt::format("Registry file {} is not a JSON array, reinitializing",
                                       registry_path.string()));
                return nlohmann::json::array();
            }
        } catch (const std::exception& e) {
            ARTEA_ERROR(fmt::format("Failed to parse existing registry: {}", e.what()));
            return nlohmann::json::array();
        }

        return registry;
    }

    /**
     * @brief Save registry to file.
     * @param registry_path Path to the registry JSON file
     * @param registry JSON array to save
     * @return true if save succeeded, false otherwise
     */
    static auto save_registry(
        const std::filesystem::path& registry_path,
        const nlohmann::json& registry
    ) -> bool {
        // Create parent directory if needed
        std::filesystem::create_directories(registry_path.parent_path());

        std::ofstream registry_out(registry_path);
        if (!registry_out.is_open()) {
            ARTEA_ERROR(fmt::format("Failed to open registry file for writing: {}",
                                    registry_path.string()));
            return false;
        }

        try {
            registry_out << registry.dump(2);
            ARTEA_INFO(fmt::format("Registry saved to {}", registry_path.string()));
            return true;
        } catch (const std::exception& e) {
            ARTEA_ERROR(fmt::format("Failed to write registry: {}", e.what()));
            return false;
        }
    }

    /**
     * @brief Check if an entry matches the given algorithm and parameters.
     * @param entry JSON entry to check
     * @param algorithm Algorithm name
     * @param params Parameters to match
     * @return true if entry matches, false otherwise
     */
    static auto is_same_entry(
        const nlohmann::json& entry,
        const std::string& algorithm,
        const nlohmann::json& params
    ) -> bool {
        return entry.contains("algorithm") &&
               entry["algorithm"] == algorithm &&
               entry.contains("params") &&
               entry["params"] == params;
    }

};  // class IndexRegisterUtil

}  // namespace cpu
}  // namespace artea
