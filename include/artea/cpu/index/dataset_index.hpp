// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

namespace artea::cpu {

/** @brief Common data ownership for graph indexes; graph state belongs to derived classes.
 *
 * Loaded datasets are immutable. Incremental indexes append to an initially empty owned dataset.
 * Legacy vector-reference constructors explicitly borrow data that must outlive the whole index,
 * including its compact graph. Derived graph members are destroyed before this data-owning base.
 * Moving a derived index must transfer its graph together with this base; dataset addresses stay stable.
 */
template <typename IndexTraitsT>
class DatasetIndex {
public:
    using vector_dataset_t = typename IndexTraitsT::vector_dataset_t;
    using vector_array_t = typename IndexTraitsT::vector_array_t;

    DatasetIndex(const DatasetIndex&) = delete;
    DatasetIndex& operator=(const DatasetIndex&) = delete;

    /** @brief Whether this index owns a dataset rather than borrowing external base vectors. */
    auto owns_dataset() const noexcept -> bool { return static_cast<bool>(_dataset); }

    /** @brief Immutable dataset, available in loaded and incremental owning modes. */
    auto get_dataset() const -> const vector_dataset_t& {
        if (!_dataset) throw std::logic_error("This index does not own a dataset");
        return *_dataset;
    }

    /** @brief Base vectors shared by construction and queries in all storage modes. */
    auto get_base_vecs() const -> const vector_array_t& {
        return _borrowed_vectors ? *_borrowed_vectors : get_dataset().get_base_vecs();
    }

    /** @brief Compatibility accessor for read-only construction code. */
    auto get_vecs_storage() const -> const vector_array_t& { return get_base_vecs(); }

    /** @brief Mutable storage is restricted to incremental indexes between graph operations. */
    auto get_vecs_storage() -> vector_array_t& {
        if (!_allows_append || !_dataset) throw std::logic_error("Index vectors are immutable");
        return _dataset->get_base_vecs();
    }

protected:
    /** @brief Create an incremental index with one initially empty owned dataset. */
    DatasetIndex() : _dataset(std::make_unique<vector_dataset_t>()), _allows_append(true) {}

    /** @brief Take exclusive ownership without moving or copying vector arrays. */
    explicit DatasetIndex(std::unique_ptr<vector_dataset_t> dataset) : _dataset(std::move(dataset)) {
        if (!_dataset) throw std::invalid_argument("DatasetIndex requires a dataset");
    }

    /** @brief Borrow immutable vectors; the caller must keep their storage alive and unchanged. */
    explicit DatasetIndex(const vector_array_t& vectors) : _borrowed_vectors(&vectors) {}
    DatasetIndex(vector_array_t&&) = delete;
    DatasetIndex(const vector_array_t&&) = delete;

    DatasetIndex(DatasetIndex&& other) noexcept
        : _dataset(std::move(other._dataset)),
          _borrowed_vectors(std::exchange(other._borrowed_vectors, nullptr)),
          _allows_append(std::exchange(other._allows_append, false)) {}

    /** @brief Derived move assignment must destroy its old graphs before moving the data owner. */
    auto operator=(DatasetIndex&& other) noexcept -> DatasetIndex& {
        if (this != &other) {
            _dataset = std::move(other._dataset);
            _borrowed_vectors = std::exchange(other._borrowed_vectors, nullptr);
            _allows_append = std::exchange(other._allows_append, false);
        }
        return *this;
    }

    /** @brief Destroy only through the concrete index type; this base has no virtual dispatch. */
    ~DatasetIndex() = default;

    /** @brief Append after the derived index has checked that its building graph is available. */
    auto append_base_vecs(vector_array_t&& vectors) -> void {
        get_vecs_storage().append_batch(std::move(vectors));
    }

private:
    /** @brief Sole owner of base vectors, queries, and ground truth; never replaced during a rebuild. */
    std::unique_ptr<vector_dataset_t> _dataset;

    /** @brief External immutable vectors in the legacy borrowing mode; otherwise null. */
    const vector_array_t* _borrowed_vectors = nullptr;

    /** @brief Only incremental construction may append to the owned base vectors. */
    bool _allows_append = false;
};

}  // namespace artea::cpu
