/*
 * @FilePath: /Artea/include/artea/cpu/containers/vector_dataset.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Refactored to hold VectorArray members directly, leveraging RAII
 *               for automatic memory management.
 */

#pragma once

#include <fstream>
#include <filesystem>
#include <utility> // For std::move, though it's implicitly used in assignment
#include <unordered_map>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <artea/common/logger.hpp>
#include <artea/cpu/utils/random_seq_nr.hpp>

namespace artea {
namespace cpu {

template <typename BaseTraitsT>
class VectorDataset {

    // Typedefs for clarity
    using vec_id_t = typename BaseTraitsT::vec_id_t;
    using vec_num_t = typename BaseTraitsT::vec_num_t;
    using vec_ele_t = typename BaseTraitsT::vec_ele_t;
    using base_vecs_t = typename BaseTraitsT::base_vecs_t;
    using query_vecs_t = typename BaseTraitsT::query_vecs_t;
    using ground_truth_t = typename BaseTraitsT::ground_truth_t;
    using random_seq_nr_t = RandomSeqNR<BaseTraitsT>;

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
    auto get_base_vecs() -> base_vecs_t& {
        return _base_vecs;
    }

    __attribute__((always_inline))
    auto get_base_vecs() const -> const base_vecs_t& {
        return _base_vecs;
    }

    __attribute__((always_inline))
    auto get_query_vecs() -> query_vecs_t& {
        return _query_vecs;
    }

    __attribute__((always_inline))
    auto get_query_vecs() const -> const query_vecs_t& {
        return _query_vecs;
    }

    __attribute__((always_inline))
    auto get_gt_vecs() -> ground_truth_t& {
        return _gt_vecs;
    }

    __attribute__((always_inline))
    auto get_gt_vecs() const -> const ground_truth_t& {
        return _gt_vecs;
    }

    __attribute__((always_inline))
    auto get_num_base_vecs() const -> vec_num_t {
        return static_cast<vec_num_t>(_base_vecs.get_num_vecs());
    }

    __attribute__((always_inline))
    auto get_num_query_vecs() const -> vec_num_t {
        return static_cast<vec_num_t>(_query_vecs.get_num_vecs());
    }

    __attribute__((always_inline))
    auto get_num_gt_vecs() const -> vec_num_t {
        return static_cast<vec_num_t>(_gt_vecs.get_num_vecs());
    }

    __attribute__((always_inline))
    auto get_vec_dim() const -> vec_num_t {
        return static_cast<vec_num_t>(_base_vecs.get_vec_dim());
    }

    /**
     * @brief Shuffle the dataset in-place using Fisher-Yates algorithm.
     *        Updates base vectors and ground truth IDs accordingly.
     * @param seed Random seed for reproducibility (default: random_device).
     *
     * TODO: Optimize memory usage by implementing true in-place shuffle.
     *       Current implementation uses extract_subset which creates a full copy,
     *       resulting in 2x peak memory usage. A better approach would be:
     *       1. Use Fisher-Yates shuffle with direct vector swapping via get()
     *       2. Parallelize vector element swaps with TBB for large dimensions
     *       3. Parallelize ground truth updates with TBB
     *       This would reduce peak memory from 2N to N bytes.
     */
    auto shuffle_in_place(uint32_t seed = std::random_device{}()) -> uint32_t {
        const vec_num_t num_base_vecs = _base_vecs.get_num_vecs();
        if (num_base_vecs == 0) {
            logger.warn("Cannot shuffle empty dataset");
            return seed;
        }

        logger.info(fmt::format("Shuffling dataset with {} base vectors (seed={})...", num_base_vecs, seed));

        // Generate shuffle indices using RandomSeqNR
        random_seq_nr_t shuffle_gen(num_base_vecs, seed);

        // Create old_id -> new_id mapping for ground truth update
        std::vector<vec_id_t> old_to_new(num_base_vecs);
        std::vector<vec_id_t> shuffle_ids(num_base_vecs);
        for (vec_num_t new_id = 0; new_id < num_base_vecs; ++new_id) {
            vec_id_t old_id = shuffle_gen[new_id];
            old_to_new[old_id] = static_cast<vec_id_t>(new_id);
            shuffle_ids[new_id] = old_id;
        }

        // Shuffle base vectors in-place using extract_subset and move assignment
        _base_vecs = std::move(_base_vecs.extract_subset(shuffle_ids));

        // Update ground truth IDs
        const vec_num_t num_gt_vecs = _gt_vecs.get_num_vecs();
        const vec_num_t gt_vec_dim = _gt_vecs.get_vec_dim();

        for (vec_num_t i = 0; i < num_gt_vecs; ++i) {
            vec_id_t* gt_row = _gt_vecs.get(i);
            for (vec_num_t j = 0; j < gt_vec_dim; ++j) {
                vec_id_t old_id = gt_row[j];
                if (old_id < num_base_vecs) {
                    gt_row[j] = old_to_new[old_id];
                }
            }
        }

        logger.success("Dataset shuffled successfully");
        return seed;
    }

private:
    nlohmann::json _config;

    base_vecs_t _base_vecs;
    query_vecs_t _query_vecs;
    ground_truth_t _gt_vecs;

    auto _load_config(const std::string& config_path) -> void {
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            throw std::runtime_error("Failed to open dataset config file.");
        }
        _config = nlohmann::json::parse(config_file);
        config_file.close();

        logger.success(fmt::format("Successfully loaded dataset config from {}", config_path));
    }

    auto _load_datasets(const std::string& dataset_name) -> void {
        auto dataset_config = _config["datasets"][dataset_name];

        std::filesystem::path root_dir = _config["root_dir"];
        std::filesystem::path dataset_dir = root_dir / dataset_config["dataset_dir"];
        std::filesystem::path base_vecs_path = dataset_dir / dataset_config["base_path"];
        std::filesystem::path query_vecs_path = dataset_dir / dataset_config["query_path"];
        std::filesystem::path gt_vecs_path = dataset_dir / dataset_config["gt_path"];

        logger.info(fmt::format("Loading dataset {} from {} ...", dataset_name, dataset_dir.string()));

        _base_vecs = base_vecs_t(base_vecs_path.string());
        _query_vecs = query_vecs_t(query_vecs_path.string());
        _gt_vecs = ground_truth_t(gt_vecs_path.string());

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