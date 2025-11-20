/*
 * @FilePath: /Artea/include/artea/cpu/vector_dataset.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Refactored to hold VectorArray members directly, leveraging RAII
 *               for automatic memory management.
 */

#pragma once

#include <fstream>
#include <filesystem>
#include <utility> // For std::move, though it's implicitly used in assignment

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <artea/definitions.hpp>
#include <artea/cpu/vector_array.hpp>
#include <artea/logger.hpp>
#include <artea/config.hpp>

namespace artea {
namespace cpu {

template <
    typename vecs_num_t,
    typename vec_ele_t
>
class VectorDataset {

    // Typedefs for clarity
    using vec_id_t = vecs_num_t;
    using BaseQueryArray = VectorArray<vecs_num_t, vec_ele_t>;
    using GroundTruthArray = VectorArray<vecs_num_t, vec_id_t>;

public:

    VectorDataset() = default;
    VectorDataset(const std::string& config_path, const std::string& dataset_name) {
        from_config(config_path, dataset_name);
    }

    ~VectorDataset() = default;
    VectorDataset(const VectorDataset&) = delete;
    VectorDataset& operator=(const VectorDataset&) = delete;

    /**
     * @brief Moving a dataset is efficient and is the preferred way to transfer ownership.
     *        This is inherited from the movable nature of VectorArray.
     */
    VectorDataset(VectorDataset&&) noexcept = default;
    VectorDataset& operator=(VectorDataset&&) noexcept = default;


    void from_config(const std::string& config_path, const std::string& dataset_name) {
        _load_config(config_path);
        _load_datasets(dataset_name);
    }

    // --- Accessors ---
    __attribute__((always_inline))
    auto get_base_vecs() -> BaseQueryArray* {
        return &_base_vecs;
    }

    __attribute__((always_inline))
    auto get_base_vecs() const -> const BaseQueryArray* {
        return &_base_vecs;
    }

    __attribute__((always_inline))
    auto get_query_vecs() -> BaseQueryArray* {
        return &_query_vecs;
    }

    __attribute__((always_inline))
    auto get_query_vecs() const -> const BaseQueryArray* {
        return &_query_vecs;
    }

    __attribute__((always_inline))
    auto get_gt_vecs() -> GroundTruthArray* {
        return &_gt_vecs;
    }

    __attribute__((always_inline))
    auto get_gt_vecs() const -> const GroundTruthArray* {
        return &_gt_vecs;
    }

private:
    nlohmann::json _config;

    BaseQueryArray _base_vecs;
    BaseQueryArray _query_vecs;
    GroundTruthArray _gt_vecs;

    auto _load_config(const std::string& config_path) -> void {
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            throw std::runtime_error("Failed to open dataset config file.");
        }
        _config = nlohmann::json::parse(config_file);
        config_file.close();

        ArteaLogger logger("VectorDataset::load_config", system_log_level);
        logger.success(fmt::format("Successfully loaded dataset config from {}", config_path));
    }

    auto _load_datasets(const std::string& dataset_name) -> void {
        auto dataset_config = _config["datasets"][dataset_name];

        std::filesystem::path root_dir = _config["root_dir"];
        std::filesystem::path dataset_dir = root_dir / dataset_config["dataset_dir"];
        std::filesystem::path base_vecs_path = dataset_dir / dataset_config["base_path"];
        std::filesystem::path query_vecs_path = dataset_dir / dataset_config["query_path"];
        std::filesystem::path gt_vecs_path = dataset_dir / dataset_config["gt_path"];

        ArteaLogger logger("VectorDataset::load_datasets", system_log_level);
        logger.info(fmt::format("Loading dataset {} from {} ...", dataset_name, dataset_dir.string()));

        _base_vecs = BaseQueryArray(base_vecs_path.string());
        _query_vecs = BaseQueryArray(query_vecs_path.string());
        _gt_vecs = GroundTruthArray(gt_vecs_path.string());

        logger.success(
            fmt::format(
                "Successfully loaded {} base vectors ({} dims), {} query vectors ({} dims), and {} ground truth vectors ({} dims).",
                _base_vecs.get_num_vecs(),
                _base_vecs.get_vec_dim(),
                _query_vecs.get_num_vecs(),
                _query_vecs.get_vec_dim(),
                _gt_vecs.get_num_vecs(),
                _gt_vecs.get_vec_dim()
            )
        );
    }
};  // class VectorDataset

} // namespace cpu
} // namespace artea