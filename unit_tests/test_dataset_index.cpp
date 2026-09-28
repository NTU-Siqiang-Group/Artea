// Copyright 2026 Weitang Ye
// SPDX-License-Identifier: Apache-2.0

#include <artea/cpu/framework/type_context/default_context.hpp>
#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include <algorithm>
#include <bit>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>

using namespace artea::cpu;

namespace {
constexpr vec_dim_t dimension = 128;
constexpr auto metric = DistanceMetricsT::EUCLIDEAN;
using factory_t = artea_graph::factory_t<metric, dimension>;
using building_index_t = artea_graph::index_t<metric, dimension>;
using rgraph_config_t = artea_graph::rgraph_config_t<metric, dimension>;
using propagate_config_t = artea_graph::propagate_config_t<metric, dimension>;
using pruning_config_t = artea_graph::pruning_config_t<metric, dimension>;

auto make_dataset(vertex_num_t count) -> std::unique_ptr<vector_dataset_t> {
    auto dataset = std::make_unique<vector_dataset_t>();
    dataset->get_base_vecs() = vector_array_t(count, dimension);
    dataset->get_query_vecs() = vector_array_t(4, dimension);
    dataset->get_gt_vecs() = ground_truth_t(4, 1);
    std::mt19937 generator(4103);
    std::normal_distribution<float> coordinates(0.0f, 10.0f);
    for (vertex_id_t vid = 0; vid < count; ++vid) {
        for (vec_dim_t coordinate = 0; coordinate < dimension; ++coordinate) {
            dataset->get_base_vecs().get(vid)[coordinate] = coordinates(generator);
        }
    }
    for (vertex_id_t query_id = 0; query_id < 4 && count; ++query_id) {
        const vertex_id_t target = query_id % count;
        std::copy_n(dataset->get_base_vecs().get(target), dimension, dataset->get_query_vecs().get(query_id));
        dataset->get_gt_vecs().get(query_id)[0] = target;
    }
    return dataset;
}

auto rgraph_config(vertex_num_t bottom_capacity = 16) -> rgraph_config_t {
    return rgraph_config_t(2.0f, 0, 0.0f, 3.0f, 16, 32, 32, 8, bottom_capacity);
}

void expect_same_graph(const compact::hierarchical_graph_t& left,
                       const compact::hierarchical_graph_t& right) {
    ASSERT_EQ(left.get_num_vertices(), right.get_num_vertices());
    ASSERT_EQ(left.top_occupied_level_id(), right.top_occupied_level_id());
    ASSERT_EQ(left.entry_point_vid(), right.entry_point_vid());
    for (vertex_id_t vid = 0; vid < left.get_num_vertices(); ++vid) {
        const auto highest_level = left.get_highest_level_id(vid);
        ASSERT_EQ(highest_level, right.get_highest_level_id(vid));
        for (layer_id_t level = 0; level <= highest_level; ++level) {
            const auto left_neighbors = left.fetch_level_nbrs(vid, level);
            const auto right_neighbors = right.fetch_level_nbrs(vid, level);
            ASSERT_EQ(left_neighbors.size(), right_neighbors.size());
            EXPECT_TRUE(std::equal(left_neighbors.begin(), left_neighbors.end(), right_neighbors.begin()));
        }
    }
}

TEST(DatasetIndex, OwnershipSurvivesCompactionAndRebuildsWithoutCopying) {
    auto dataset = make_dataset(96);
    const auto* dataset_address = dataset.get();
    const float* base_address = dataset->get_base_vecs().get_all();
    const float* query_address = dataset->get_query_vecs().get_all();
    const auto* gt_address = dataset->get_gt_vecs().get_all();
    const auto original_vectors = dataset->get_base_vecs().extract_subset(0, 96);
    index_traits_t::artea_graph::index_t owner(std::move(dataset));
    EXPECT_EQ(dataset, nullptr);
    EXPECT_EQ(&owner.get_dataset(), dataset_address);
    const propagate_config_t propagate(1, 1, 0.6f);
    const pruning_config_t pruning(1.05f, 0.0f);
    dist_func_t<metric, dimension> build_distance;

    for (bool shuffle : {false, true}) {
        const auto config = rgraph_config(shuffle ? 24 : 16);
        auto& building = owner.prepare_build(config, propagate, pruning);
        EXPECT_THROW(owner.get_compact_graph(), std::logic_error);
        EXPECT_EQ(std::as_const(building).get_vecs_storage().get_all(), base_address);
        factory_t::build_from_vectors(building, build_distance, false, shuffle);
        EXPECT_THROW(factory_t::build_from_vectors(building, build_distance), std::logic_error);

        // Compare against the exact construction graph; random refinement changes independent builds.
        auto expected = hierarchical_graph_compactor_t::compact_graph(
            building.get_hierarchical_graph(), original_vectors, build_distance);
        owner.compact(build_distance);
        EXPECT_THROW(owner.get_hierarchical_graph(), std::logic_error);
        EXPECT_EQ(owner.get_dataset().get_base_vecs().get_all(), base_address);
        EXPECT_EQ(owner.get_dataset().get_query_vecs().get_all(), query_address);
        EXPECT_EQ(owner.get_dataset().get_gt_vecs().get_all(), gt_address);
        EXPECT_TRUE(std::equal(original_vectors.begin(), original_vectors.end(),
            owner.get_dataset().get_base_vecs().begin(), [](const float* left, const float* right) {
                return std::equal(left, left + dimension, right);
            }));
        expect_same_graph(owner.get_compact_graph(), expected);

        dist_func_t<DistanceMetricsT::EUCLIDEAN_SQR, dimension> search_distance;
        hierarchical_graph_router_t<DistanceMetricsT::EUCLIDEAN_SQR, dimension> router(
            owner.get_dataset().get_base_vecs(), search_distance, 5, 96);
        router.initialize();
        const auto batch = router.template batch_query<false, false>(
            owner.get_dataset().get_query_vecs(), owner.get_compact_graph());
        ASSERT_EQ(batch.size(), 20);
        for (vertex_id_t query_id = 0; query_id < 4; ++query_id) {
            const auto single = router.template query<false, false>(
                query_address + query_id * dimension, expected);
            ASSERT_EQ(single.size(), 5);
            for (std::size_t rank = 0; rank < single.size(); ++rank) {
                EXPECT_EQ(batch[query_id * 5 + rank].get_vid(), single[rank].get_vid());
                EXPECT_EQ(std::bit_cast<uint32_t>(batch[query_id * 5 + rank].get_distance()),
                          std::bit_cast<uint32_t>(single[rank].get_distance()));
            }
        }
    }
}

TEST(DatasetIndex, StaticAndOwnedInsertionProduceTheSameGraph) {
    // Disable parallel scheduling and use a small graph that skips randomized refinement.
    tbb::global_control deterministic_build(tbb::global_control::max_allowed_parallelism, 1);
    auto dataset = make_dataset(24);
    const auto& vectors = dataset->get_base_vecs();
    const rgraph_config_t config(2.0f, 0, 0.0f, 3.0f, 64, 64, 64, 32, 32);
    const propagate_config_t propagate(1, 1, 0.6f);
    const pruning_config_t pruning(1.05f, 0.0f);
    dist_func_t<metric, dimension> distance;
    for (bool shuffle : {false, true}) {
        building_index_t borrowed(vectors, config, propagate, pruning);
        building_index_t owned(24, config, propagate, pruning);
        factory_t::build_from_vectors(borrowed, distance, true, shuffle);
        factory_t::add_vertices(owned, vectors.extract_subset(0, 24), distance, true, shuffle);
        const auto static_graph = hierarchical_graph_compactor_t::compact_graph(
            borrowed.get_hierarchical_graph(), vectors, distance);
        const auto owned_graph = hierarchical_graph_compactor_t::compact_graph(
            owned.get_hierarchical_graph(), vectors, distance);
        expect_same_graph(static_graph, owned_graph);
    }
}

TEST(DatasetIndex, BorrowedStorageRejectsMutationAndTemporarySources) {
    static_assert(!std::is_constructible_v<building_index_t, vector_array_t&&,
                  const rgraph_config_t&, propagate_config_t, pruning_config_t>);
    static_assert(!std::is_constructible_v<building_index_t, const vector_array_t&&,
                  const rgraph_config_t&, propagate_config_t, pruning_config_t>);
    auto dataset = make_dataset(8);
    const auto* source_address = dataset->get_base_vecs().get_all();
    index_traits_t::artea_graph::index_t owner(std::move(dataset));
    auto& building = owner.prepare_build(rgraph_config(), propagate_config_t(1, 1, 0.6f),
                                         pruning_config_t(1.05f, 0.0f));
    EXPECT_THROW(building.get_vecs_storage(), std::logic_error);
    auto batch = vector_array_t(2, dimension);
    const auto* batch_address = batch.get_all();
    dist_func_t<metric, dimension> distance;
    EXPECT_THROW(factory_t::add_vertices(building, std::move(batch), distance), std::logic_error);
    EXPECT_EQ(batch.get_all(), batch_address);
    EXPECT_EQ(building.get_num_vertices(), 0);
    EXPECT_EQ(std::as_const(building).get_vecs_storage().get_all(), source_address);
}

TEST(DatasetIndex, CompactionFailureRetainsDatasetAndAllowsRetry) {
    index_traits_t::artea_graph::index_t owner(make_dataset(8));
    const auto* base_address = owner.get_dataset().get_base_vecs().get_all();
    auto& building = owner.prepare_build(rgraph_config(), propagate_config_t(1, 1, 0.6f),
                                         pruning_config_t(1.05f, 0.0f));
    dist_func_t<metric, dimension> distance;
    factory_t::build_from_vectors(building, distance, true);
    const auto failing_distance = [](const float*, const float*) -> float {
        throw std::runtime_error("simulated compaction failure");
    };
    EXPECT_THROW(owner.compact(failing_distance), std::runtime_error);
    EXPECT_EQ(owner.get_dataset().get_base_vecs().get_all(), base_address);
    EXPECT_EQ(owner.get_hierarchical_graph().get_num_vertices(), 8);
    EXPECT_THROW(owner.get_compact_graph(), std::logic_error);
    owner.compact(distance);
    EXPECT_EQ(owner.get_compact_graph().get_num_vertices(), 8);
}

TEST(DatasetIndex, EmptyDatasetAndStageAccessors) {
    EXPECT_THROW(index_traits_t::artea_graph::index_t(nullptr), std::invalid_argument);
    index_traits_t::artea_graph::index_t owner(make_dataset(0));
    EXPECT_THROW(owner.get_hierarchical_graph(), std::logic_error);
    EXPECT_THROW(owner.get_compact_graph(), std::logic_error);
    auto& building = owner.prepare_build(rgraph_config(), propagate_config_t(1, 1, 0.6f),
                                         pruning_config_t(1.05f, 0.0f));
    dist_func_t<metric, dimension> distance;
    const auto timing = factory_t::build_from_vectors(building, distance);
    EXPECT_EQ(timing.bottom_layer_time_ms, 0.0);
    owner.compact(distance);
    EXPECT_EQ(owner.get_compact_graph().get_num_vertices(), 0);
}

TEST(DatasetIndex, IncrementalOwnedBatchesPreserveVectorIdsAndContents) {
    auto dataset = make_dataset(48);
    const auto& original = dataset->get_base_vecs();
    building_index_t index(48, rgraph_config(), propagate_config_t(1, 1, 0.6f),
                           pruning_config_t(1.05f, 0.0f));
    dist_func_t<metric, dimension> distance;
    for (vertex_num_t offset : {0u, 16u, 32u}) {
        {
            auto batch = original.extract_subset(offset, 16);
            factory_t::add_vertices(index, std::move(batch), distance, true);
        }
        EXPECT_EQ(index.get_num_vertices(), offset + 16);
        const auto& stored = std::as_const(index).get_vecs_storage();
        ASSERT_EQ(stored.get_num_vecs(), offset + 16);
        EXPECT_TRUE(std::equal(original.get_all(), original.get_all() + (offset + 16) * dimension,
                               stored.get_all()));
    }
    auto empty_batch = vector_array_t(0, dimension);
    EXPECT_NO_THROW(factory_t::add_vertices(index, std::move(empty_batch), distance));
    EXPECT_EQ(index.get_num_vertices(), 48);
    index.release();
    EXPECT_EQ(std::as_const(index).get_vecs_storage().get_num_vecs(), 48);
}

using knn_index_t = index_traits_t::knn_graph::index_t;
using conv_index_t = index_traits_t::conv_graph::index_t;
using stacked_index_t = index_traits_t::stacked_rgraph::index_t;
using hier_index_t = index_traits_t::hier_conv_graph::index_t;
using flat_propagate_t = index_traits_t::knn_graph::propagate_config_t;
using flat_pruning_t = index_traits_t::conv_graph::pruning_config_t;
using hierarchy_config_t = index_traits_t::hier_conv_graph::hierarchy_config_t;
using layer_config_t = index_traits_t::layer_config_t;

static_assert(std::is_base_of_v<DatasetIndex<index_traits_t>, building_index_t>);
static_assert(std::is_base_of_v<DatasetIndex<index_traits_t>, stacked_index_t>);
static_assert(std::is_base_of_v<DatasetIndex<index_traits_t>, hier_index_t>);
static_assert(std::is_base_of_v<DatasetIndex<index_traits_t>, knn_index_t>);
static_assert(std::is_base_of_v<DatasetIndex<index_traits_t>, conv_index_t>);
static_assert(std::is_same_v<knn_index_t, index_traits_t::symmetric_knn_graph::index_t>);
static_assert(!std::is_destructible_v<DatasetIndex<index_traits_t>>);
static_assert(!std::is_move_constructible_v<DatasetIndex<index_traits_t>>);
static_assert(!std::is_constructible_v<knn_index_t, vector_array_t&&, layer_config_t, flat_propagate_t>);
static_assert(!std::is_constructible_v<conv_index_t, const vector_array_t&&,
              layer_config_t, flat_pruning_t, flat_propagate_t>);

struct KnnCase {
    using index_t = knn_index_t;
    using factory_t = knn_graph::factory_t<metric, dimension>;
    static void prepare(index_t& index, vertex_num_t degree) {
        index.prepare_build(layer_config_t(degree), flat_propagate_t(1, 1, 0.5f));
    }
    static auto construct(std::unique_ptr<vector_dataset_t> dataset) {
        return factory_t::construct_graph(std::move(dataset), layer_config_t(16),
                                         flat_propagate_t(1, 1, 0.5f));
    }
};
struct SymmetricKnnCase : KnnCase {
    using factory_t = symmetric_knn_graph::factory_t<metric, dimension>;
    static auto construct(std::unique_ptr<vector_dataset_t> dataset) {
        return factory_t::construct_graph(std::move(dataset), layer_config_t(16),
                                         flat_propagate_t(1, 1, 0.5f));
    }
};
struct ConvCase {
    using index_t = conv_index_t;
    using factory_t = conv_graph::factory_t<metric, dimension>;
    static void prepare(index_t& index, vertex_num_t degree) {
        index.prepare_build(layer_config_t(degree), flat_pruning_t(1.1f, 0),
                            flat_propagate_t(1, 1, 0.5f));
    }
    static auto construct(std::unique_ptr<vector_dataset_t> dataset) {
        return factory_t::construct_graph(std::move(dataset), layer_config_t(16),
                                         flat_pruning_t(1.1f, 0), flat_propagate_t(1, 1, 0.5f));
    }
};

template <typename CaseT>
class FlatDatasetIndex : public ::testing::Test {};
using FlatCases = ::testing::Types<KnnCase, SymmetricKnnCase, ConvCase>;
TYPED_TEST_SUITE(FlatDatasetIndex, FlatCases);

TYPED_TEST(FlatDatasetIndex, OwnsBuildsCompactsRebuildsAndMovesWithoutCopying) {
    using index_t = typename TypeParam::index_t;
    using build_factory_t = typename TypeParam::factory_t;
    auto dataset = make_dataset(96);
    const auto* dataset_address = dataset.get();
    const auto* base_address = dataset->get_base_vecs().get_all();
    const auto* query_address = dataset->get_query_vecs().get_all();
    const auto* gt_address = dataset->get_gt_vecs().get_all();
    index_t index(std::move(dataset));
    EXPECT_THROW(index.get_refining_graph(), std::logic_error);
    EXPECT_THROW(index.get_compact_graph(), std::logic_error);
    EXPECT_THROW(index.get_vecs_storage(), std::logic_error);
    dist_func_t<metric, dimension> distance;
    for (vertex_num_t degree : {16u, 24u}) {
        TypeParam::prepare(index, degree);
        EXPECT_THROW(index.get_compact_graph(), std::logic_error);
        build_factory_t::build_from_vectors(index, distance);
        auto reference = refining_graph_compactor_t::compact_graph(index.get_refining_graph(), 8);
        index.compact(8);
        EXPECT_THROW(index.get_refining_graph(), std::logic_error);
        EXPECT_EQ(index.get_compact_graph().get_csr_nbrs(), reference.get_csr_nbrs());
        EXPECT_EQ(&index.get_compact_graph().get_vecs_data(), &index.get_dataset().get_base_vecs());

        // Exercise construction and assignment of both compact and building states.
        index_t moved(std::move(index));
        EXPECT_FALSE(index.owns_dataset());
        EXPECT_THROW(index.get_base_vecs(), std::logic_error);
        index_t target(make_dataset(48));
        TypeParam::prepare(target, 16);
        build_factory_t::build_from_vectors(target, distance);
        if (degree == 24) target.compact(8);
        target = std::move(moved);
        EXPECT_THROW(moved.get_compact_graph(), std::logic_error);
        index = std::move(target);
        index = std::move(index);
        EXPECT_EQ(&index.get_dataset(), dataset_address);
        EXPECT_EQ(index.get_base_vecs().get_all(), base_address);
        EXPECT_EQ(index.get_dataset().get_query_vecs().get_all(), query_address);
        EXPECT_EQ(index.get_dataset().get_gt_vecs().get_all(), gt_address);
        EXPECT_EQ(index.get_compact_graph().get_vecs_data().get_all(), base_address);

        single_layer_router_t<metric, dimension> router(index.get_base_vecs(), distance, 5, 96);
        router.initialize();
        const auto actual = router.batch_query(
            index.get_dataset().get_query_vecs(), index.get_compact_graph());
        const auto expected = router.batch_query(index.get_dataset().get_query_vecs(), reference);
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t rank = 0; rank < actual.size(); ++rank) {
            EXPECT_EQ(actual[rank].get_vid(), expected[rank].get_vid());
            EXPECT_EQ(actual[rank].get_distance(), expected[rank].get_distance());
        }
    }
}

TYPED_TEST(FlatDatasetIndex, OwningFactoryAndEmptyDataset) {
    using index_t = typename TypeParam::index_t;
    EXPECT_THROW(index_t(nullptr), std::invalid_argument);
    for (vertex_num_t count : {0u, 96u}) {
        auto dataset = make_dataset(count);
        const auto* original = dataset.get();
        auto result = TypeParam::construct(std::move(dataset));
        EXPECT_EQ(dataset, nullptr);
        EXPECT_EQ(&result.graph.get_dataset(), original);
        EXPECT_EQ(result.graph.get_num_vertices(), count);
        result.graph.compact(8);
        EXPECT_EQ(result.graph.get_compact_graph().get_num_vertices(), count);
        EXPECT_EQ(&result.graph.get_compact_graph().get_vecs_data(), &original->get_base_vecs());
    }
}

TEST(DatasetIndex, FlatBorrowingRemainsExplicitAndMoveAssignmentReplacesReferences) {
    auto first_dataset = make_dataset(96);
    auto second_dataset = make_dataset(48);
    knn_index_t first(first_dataset->get_base_vecs(), layer_config_t(16), flat_propagate_t(1, 1));
    knn_index_t second(second_dataset->get_base_vecs(), layer_config_t(16), flat_propagate_t(1, 1));
    EXPECT_FALSE(first.owns_dataset());
    EXPECT_THROW(first.get_dataset(), std::logic_error);
    EXPECT_THROW(first.get_vecs_storage(), std::logic_error);
    second = std::move(first);
    EXPECT_THROW(first.get_base_vecs(), std::logic_error);
    EXPECT_EQ(&second.get_refining_graph().get_vecs_data(), &first_dataset->get_base_vecs());
    auto owned = KnnCase::construct(make_dataset(64));
    owned.graph = std::move(second);
    EXPECT_FALSE(owned.graph.owns_dataset());
    EXPECT_NE(&owned.graph.get_base_vecs(), &second_dataset->get_base_vecs());
}

TEST(DatasetIndex, KnnToSymmetricToConvTransfersDatasetAlongWithGraph) {
    auto dataset = make_dataset(96);
    const auto* original = dataset.get();
    const auto* vectors = dataset->get_base_vecs().get_all();
    auto converted = [&]() {
        auto knn = KnnCase::construct(std::move(dataset));
        auto symmetric = symmetric_knn_graph::factory_t<metric, dimension>::construct_graph(
            std::move(knn.graph));
        EXPECT_FALSE(knn.graph.owns_dataset());
        auto conv = conv_graph::factory_t<metric, dimension>::construct_graph(
            std::move(symmetric.graph), flat_pruning_t(1.1f, 0));
        EXPECT_FALSE(symmetric.graph.owns_dataset());
        EXPECT_THROW(symmetric.graph.get_refining_graph(), std::logic_error);
        return conv;
    }();
    EXPECT_EQ(&converted.graph.get_dataset(), original);
    EXPECT_EQ(converted.graph.get_refining_graph().get_vecs_data().get_all(), vectors);
    converted.graph.compact(8);
    EXPECT_EQ(converted.graph.get_compact_graph().get_vecs_data().get_all(), vectors);
    dist_func_t<metric, dimension> distance;
    single_layer_router_t<metric, dimension> router(converted.graph.get_base_vecs(), distance, 5, 96);
    router.initialize();
    const auto results = router.batch_query(original->get_query_vecs(), converted.graph.get_compact_graph());
    ASSERT_EQ(results.size(), 20);
    for (const auto& result : results) EXPECT_LT(result.get_vid(), 96);
}

TEST(DatasetIndex, ConversionRejectsCompactedSourceWithoutMovingItsDataset) {
    auto source = KnnCase::construct(make_dataset(96));
    const auto* original = &source.graph.get_dataset();
    source.graph.compact(8);
    EXPECT_THROW((conv_graph::factory_t<metric, dimension>::construct_graph(
        std::move(source.graph), flat_pruning_t(1.1f, 0))), std::logic_error);
    EXPECT_THROW((symmetric_knn_graph::factory_t<metric, dimension>::construct_graph(
        std::move(source.graph))), std::logic_error);
    EXPECT_EQ(&source.graph.get_dataset(), original);
    EXPECT_EQ(source.graph.get_compact_graph().get_num_vertices(), 96);
}

struct StackedCase {
    using index_t = stacked_index_t;
    using factory_t = stacked_rgraph::factory_t<metric, dimension>;
    static void prepare(index_t& index, vertex_num_t degree) {
        index.prepare_build(rgraph_config(degree), flat_pruning_t(1.1f, 0));
    }
};
struct HierConvCase {
    using index_t = hier_index_t;
    using factory_t = hier_conv_graph::factory_t<metric, dimension>;
    static void prepare(index_t& index, vertex_num_t degree) {
        index.prepare_build(hierarchy_config_t(8, degree, 0.2f), flat_propagate_t(1, 1, 0.5f),
                            flat_pruning_t(1.1f, 0));
    }
};
template <typename CaseT>
class HierarchicalDatasetIndex : public ::testing::Test {};
using HierarchicalCases = ::testing::Types<StackedCase, HierConvCase>;
TYPED_TEST_SUITE(HierarchicalDatasetIndex, HierarchicalCases);

TYPED_TEST(HierarchicalDatasetIndex, DatasetSurvivesEmptyBuildCompactionAndRebuild) {
    using index_t = typename TypeParam::index_t;
    using build_factory_t = typename TypeParam::factory_t;
    EXPECT_THROW(index_t(nullptr), std::invalid_argument);
    for (vertex_num_t count : {0u, 96u}) {
        auto dataset = make_dataset(count);
        const auto* original = dataset.get();
        const auto* vectors = dataset->get_base_vecs().get_all();
        index_t index(std::move(dataset));
        EXPECT_THROW(index.get_hierarchical_graph(), std::logic_error);
        dist_func_t<metric, dimension> distance;
        for (vertex_num_t degree : {16u, 24u}) {
            TypeParam::prepare(index, degree);
            build_factory_t::build_from_vectors(index, distance);
            auto reference = hierarchical_graph_compactor_t::compact_graph(
                index.get_hierarchical_graph(), index.get_base_vecs(), distance);
            index.compact(distance);
            EXPECT_THROW(index.get_hierarchical_graph(), std::logic_error);
            EXPECT_THROW(index.fetch_level_nbrs(0, 0), std::logic_error);
            EXPECT_EQ(&index.get_dataset(), original);
            EXPECT_EQ(index.get_base_vecs().get_all(), vectors);
            expect_same_graph(reference, index.get_compact_graph());
        }
    }
}

TYPED_TEST(HierarchicalDatasetIndex, RebuildIndexesExistingVectorsBeforeAcceptingAnotherBatch) {
    using index_t = typename TypeParam::index_t;
    using build_factory_t = typename TypeParam::factory_t;
    auto index = []() {
        if constexpr (std::is_same_v<index_t, stacked_index_t>) {
            return std::make_unique<index_t>(48, rgraph_config(), flat_pruning_t(1.1f, 0));
        } else {
            return std::make_unique<index_t>(48, hierarchy_config_t(8, 16, 0.2f),
                                             flat_propagate_t(1, 1, 0.5f), flat_pruning_t(1.1f, 0));
        }
    }();
    auto dataset = make_dataset(48);
    dist_func_t<metric, dimension> distance;
    build_factory_t::add_vertices(*index, dataset->get_base_vecs().extract_subset(0, 16), distance);
    TypeParam::prepare(*index, 16);
    auto next_batch = dataset->get_base_vecs().extract_subset(16, 16);
    const auto* batch_address = next_batch.get_all();
    EXPECT_THROW(build_factory_t::add_vertices(*index, std::move(next_batch), distance), std::logic_error);
    EXPECT_EQ(index->get_num_vertices(), 0);
    EXPECT_EQ(index->get_base_vecs().get_num_vecs(), 16);
    EXPECT_EQ(next_batch.get_all(), batch_address);
    build_factory_t::build_from_vectors(*index, distance);
    build_factory_t::add_vertices(*index, std::move(next_batch), distance);
    ASSERT_EQ(index->get_num_vertices(), 32);
    EXPECT_TRUE(std::equal(
        index->get_base_vecs().get_all(), index->get_base_vecs().get_all() + 32 * dimension,
        dataset->get_base_vecs().get_all()));
}

TEST(DatasetIndex, HierConvIncrementalUsesDatasetStorageAndRetainsIds) {
    auto reference = make_dataset(96);
    hier_index_t index(96, hierarchy_config_t(8, 16, 0.2f), flat_propagate_t(1, 1, 0.5f),
                       flat_pruning_t(1.1f, 0));
    dist_func_t<metric, dimension> distance;
    for (vertex_num_t offset : {0u, 48u}) {
        auto batch = reference->get_base_vecs().extract_subset(offset, 48);
        hier_conv_graph::factory_t<metric, dimension>::add_vertices(index, std::move(batch), distance);
        EXPECT_EQ(index.get_num_vertices(), offset + 48);
        EXPECT_EQ(&index.get_base_vecs(), &index.get_dataset().get_base_vecs());
        EXPECT_TRUE(std::equal(index.get_base_vecs().get_all(),
            index.get_base_vecs().get_all() + (offset + 48) * dimension,
            reference->get_base_vecs().get_all()));
    }
    index.compact(distance);
    auto batch = vector_array_t(1, dimension);
    EXPECT_THROW((hier_conv_graph::factory_t<metric, dimension>::add_vertices(
        index, std::move(batch), distance)), std::logic_error);
    EXPECT_EQ(batch.get_num_vecs(), 1);
    EXPECT_EQ(index.get_dataset().get_num_base_vecs(), 96);
}
}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
