/*
 * @FilePath: /Artea/include/artea/cpu/vector_dataset.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-10-24 09:00:19
 * @Date: 2025-10-23 13:06:00
 * @Description: 
 */

#pragma once

// Reference: https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
#include <nlohmann/json.hpp> // Requiured for loading dataset config
#include <fstream>
#include <format>
#include <filesystem>

#include <artea/types.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/logger.hpp>
#include <artea/config.hpp>

namespace artea {
namespace cpu {

template <
    typename vecs_num_t, 
    typename vec_ele_t,
    typename vec_id_t = vecs_num_t
>
class VectorDataset {

public:
    VectorDataset() : _base_vecs(nullptr), _query_vecs(nullptr), _gt_vecs(nullptr) {}

    VectorDataset(const std::string& config_path, const std::string& dataset_name) : 
        _base_vecs(nullptr), _query_vecs(nullptr), _gt_vecs(nullptr) {   
        from_config(config_path, dataset_name);
    }

    ~VectorDataset() {
        if (_base_vecs) {
            delete _base_vecs;
            _base_vecs = nullptr;
        }
        if (_query_vecs) {
            delete _query_vecs;
            _query_vecs = nullptr;
        }
        if (_gt_vecs) {
            delete _gt_vecs;
            _gt_vecs = nullptr;
        }
    }

    __attribute__((always_inline))
    void from_config(const std::string& config_path, const std::string& dataset_name) {
        _load_config(config_path);
        _load_datasets(dataset_name);
    }

    __attribute__((always_inline))
    auto get_base_vecs() -> VectorArray<vecs_num_t, vec_ele_t>* {
        return _base_vecs;
    }

    __attribute__((always_inline))
    auto get_query_vecs() -> VectorArray<vecs_num_t, vec_ele_t>* {
        return _query_vecs;
    }

    __attribute__((always_inline))
    auto get_gt_vecs() -> VectorArray<vecs_num_t, vec_id_t>* {
        return _gt_vecs;
    }

private:
    nlohmann::json _config;

    VectorArray<vecs_num_t, vec_ele_t>* _base_vecs;
    VectorArray<vecs_num_t, vec_ele_t>* _query_vecs;
    VectorArray<vecs_num_t, vec_id_t>* _gt_vecs;

    auto _load_config(const std::string& config_path) -> void {
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            throw std::runtime_error("Failed to open dataset config file.");
        }
        _config = nlohmann::json::parse(config_file);
        config_file.close();

        ArteaLogger logger("VectorDataset::load_config", system_log_level);
        logger.success(std::format("Successfully loaded dataset config from {}", config_path));
    }

    auto _load_datasets(const std::string& dataset_name) -> void {
        auto dataset_config = _config["datasets"][dataset_name];

        std::filesystem::path root_dir = _config["root_dir"];
        std::filesystem::path dataset_dir = root_dir / dataset_config["dataset_dir"];
        std::filesystem::path base_vecs_path = dataset_dir / dataset_config["base_path"];
        std::filesystem::path query_vecs_path = dataset_dir / dataset_config["query_path"];
        std::filesystem::path gt_vecs_path = dataset_dir / dataset_config["gt_path"];

        ArteaLogger logger("VectorDataset::load_datasets", system_log_level);
        logger.info(std::format("Loading dataset {} from {} ...", dataset_name, dataset_dir.string()));
        _base_vecs = new VectorArray<vecs_num_t, vec_ele_t>(base_vecs_path);
        _query_vecs = new VectorArray<vecs_num_t, vec_ele_t>(query_vecs_path);
        _gt_vecs = new VectorArray<vecs_num_t, vec_id_t>(gt_vecs_path);

        logger.success(
            std::format(
                "Successfully loaded {} base vectors ({} dims), {} query vectors ({} dims), and {} ground truth vectors ({} dims).", 
                _base_vecs->get_num_vecs(), 
                _base_vecs->get_vec_dim(), 
                _query_vecs->get_num_vecs(), 
                _query_vecs->get_vec_dim(),
                _gt_vecs->get_num_vecs(), 
                _gt_vecs->get_vec_dim()
            )
        );
    }
};

}
}