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
 * @FilePath: /Artea/unit_tests/test_internal_graph_compactor.cpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @Description: Tests for InternalGraphCompactor (single-layer compaction).
 */

#include <memory>
#include <random>
#include <vector>
#include <gtest/gtest.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

class InternalGraphCompactorTest : public ::testing::Test {
protected:
    static constexpr vertex_num_t max_num_vertices = 10000;
    static constexpr vertex_num_t src_max_nbr_size = 63;

    void SetUp() override {
        src_graph_ = std::make_unique<internal_graph_t>(max_num_vertices, src_max_nbr_size);
    }

    /**
     * @brief Populate the source graph with @p num_vertices vertices, each with
     *        deterministic neighbors of count [0, src_max_nbr_size].
     *        Vertex i gets (i % (src_max_nbr_size + 1)) valid neighbors,
     *        where neighbor j has base_vid = i*1000 + j and layer_vid = j.
     */
    void populate_src_graph(const vertex_num_t num_vertices) {
        for (vertex_num_t i = 0; i < num_vertices; ++i) {
            const vertex_id_t v = src_graph_->add_vertex(i + 7);  // arbitrary lower_layer_vid

            const uint64_t valid_count = i % (src_max_nbr_size + 1);
            auto block = src_graph_->fetch_nbrs(v);
            // block[0] is header, block[1..src_max_nbr_size] are slots
            for (uint64_t j = 0; j < valid_count; ++j) {
                block[1 + j] = lnbr_t(i * 1000 + static_cast<vertex_id_t>(j),
                                      static_cast<vertex_id_t>(j));
            }
            src_graph_->num_valid_nbrs(v, valid_count);
        }
    }

    std::unique_ptr<internal_graph_t> src_graph_;
};

TEST_F(InternalGraphCompactorTest, EmptyGraph) {
    // No vertices added.
    auto compact = internal_graph_compactor_t::from_internal_graph(*src_graph_, src_max_nbr_size);

    EXPECT_EQ(compact.get_num_vertices(), 0u);
    EXPECT_EQ(compact.max_nbr_size(), src_max_nbr_size);
}

TEST_F(InternalGraphCompactorTest, BasicConversionEqualSize) {
    const vertex_num_t num = 1000;
    populate_src_graph(num);

    auto compact = internal_graph_compactor_t::from_internal_graph(*src_graph_, src_max_nbr_size);

    EXPECT_EQ(compact.get_num_vertices(), num);
    EXPECT_EQ(compact.max_nbr_size(), src_max_nbr_size);

    for (vertex_num_t i = 0; i < num; ++i) {
        const vertex_id_t v = static_cast<vertex_id_t>(i);
        const uint64_t valid_count = i % (src_max_nbr_size + 1);

        auto compact_nbrs = compact.fetch_nbrs(v);
        EXPECT_EQ(compact_nbrs.size(), src_max_nbr_size);

        // First valid_count slots match
        for (uint64_t j = 0; j < valid_count; ++j) {
            EXPECT_EQ(compact_nbrs[j].base_vid, i * 1000 + static_cast<vertex_id_t>(j));
            EXPECT_EQ(compact_nbrs[j].layer_vid, static_cast<vertex_id_t>(j));
        }
        // Remaining slots are sentinel
        for (uint64_t j = valid_count; j < src_max_nbr_size; ++j) {
            EXPECT_EQ(compact_nbrs[j], base_traits_t::invalid_lnbr);
        }

        // Inter-layer link preserved
        EXPECT_EQ(compact.get_inter_layer_link(v), i + 7);
    }
}

TEST_F(InternalGraphCompactorTest, ExtractedNbrSizeSmaller) {
    const vertex_num_t num = 500;
    populate_src_graph(num);

    const vertex_num_t extracted = 16;
    auto compact = internal_graph_compactor_t::from_internal_graph(*src_graph_, extracted);

    EXPECT_EQ(compact.get_num_vertices(), num);
    EXPECT_EQ(compact.max_nbr_size(), extracted);

    for (vertex_num_t i = 0; i < num; ++i) {
        const vertex_id_t v = static_cast<vertex_id_t>(i);
        const uint64_t valid_count = i % (src_max_nbr_size + 1);
        const uint64_t expected_copied = std::min(valid_count, static_cast<uint64_t>(extracted));

        auto compact_nbrs = compact.fetch_nbrs(v);
        EXPECT_EQ(compact_nbrs.size(), extracted);

        for (uint64_t j = 0; j < expected_copied; ++j) {
            EXPECT_EQ(compact_nbrs[j].base_vid, i * 1000 + static_cast<vertex_id_t>(j));
            EXPECT_EQ(compact_nbrs[j].layer_vid, static_cast<vertex_id_t>(j));
        }
        for (uint64_t j = expected_copied; j < extracted; ++j) {
            EXPECT_EQ(compact_nbrs[j], base_traits_t::invalid_lnbr);
        }
    }
}

TEST_F(InternalGraphCompactorTest, ExtractedNbrSizeLargerThrows) {
    populate_src_graph(10);
    EXPECT_THROW({
        internal_graph_compactor_t::from_internal_graph(*src_graph_, src_max_nbr_size + 1);
    }, std::runtime_error);
}

TEST_F(InternalGraphCompactorTest, AllVerticesFull) {
    // Every vertex has exactly src_max_nbr_size valid neighbors.
    const vertex_num_t num = 1000;
    for (vertex_num_t i = 0; i < num; ++i) {
        const vertex_id_t v = src_graph_->add_vertex(i);
        auto block = src_graph_->fetch_nbrs(v);
        for (uint64_t j = 0; j < src_max_nbr_size; ++j) {
            block[1 + j] = lnbr_t(i + static_cast<vertex_id_t>(j),
                                  static_cast<vertex_id_t>(j));
        }
        src_graph_->num_valid_nbrs(v, src_max_nbr_size);
    }

    auto compact = internal_graph_compactor_t::from_internal_graph(*src_graph_, src_max_nbr_size);
    EXPECT_EQ(compact.get_num_vertices(), num);

    for (vertex_id_t v = 0; v < num; ++v) {
        auto compact_nbrs = compact.fetch_nbrs(v);
        for (uint64_t j = 0; j < src_max_nbr_size; ++j) {
            EXPECT_EQ(compact_nbrs[j].base_vid, v + static_cast<vertex_id_t>(j));
            EXPECT_EQ(compact_nbrs[j].layer_vid, static_cast<vertex_id_t>(j));
        }
    }
}

TEST_F(InternalGraphCompactorTest, AllVerticesEmpty) {
    // Add vertices but never write any neighbors → all sentinel after compaction.
    const vertex_num_t num = 1000;
    for (vertex_num_t i = 0; i < num; ++i) {
        src_graph_->add_vertex(i);
    }

    auto compact = internal_graph_compactor_t::from_internal_graph(*src_graph_, src_max_nbr_size);
    EXPECT_EQ(compact.get_num_vertices(), num);

    for (vertex_id_t v = 0; v < num; ++v) {
        auto compact_nbrs = compact.fetch_nbrs(v);
        for (uint64_t j = 0; j < src_max_nbr_size; ++j) {
            EXPECT_EQ(compact_nbrs[j], base_traits_t::invalid_lnbr);
        }
    }
}

TEST_F(InternalGraphCompactorTest, LargeScale) {
    // 100k vertices, parallel correctness check.
    const vertex_num_t num = 100'000;
    populate_src_graph(num);

    auto compact = internal_graph_compactor_t::from_internal_graph(*src_graph_, src_max_nbr_size);
    EXPECT_EQ(compact.get_num_vertices(), num);

    std::atomic<uint32_t> errors{0};
    tbb::parallel_for(
        tbb::blocked_range<vertex_num_t>(0, num),
        [&](const tbb::blocked_range<vertex_num_t>& range) {
            for (vertex_num_t i = range.begin(); i < range.end(); ++i) {
                const vertex_id_t v = static_cast<vertex_id_t>(i);
                const uint64_t valid_count = i % (src_max_nbr_size + 1);

                auto compact_nbrs = compact.fetch_nbrs(v);

                for (uint64_t j = 0; j < valid_count; ++j) {
                    if (compact_nbrs[j].base_vid != i * 1000 + static_cast<vertex_id_t>(j) ||
                        compact_nbrs[j].layer_vid != static_cast<vertex_id_t>(j)) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                for (uint64_t j = valid_count; j < src_max_nbr_size; ++j) {
                    if (compact_nbrs[j] != base_traits_t::invalid_lnbr) {
                        errors.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                if (compact.get_inter_layer_link(v) != i + 7) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    );
    EXPECT_EQ(errors.load(), 0u);
}

TEST_F(InternalGraphCompactorTest, SourceUnchangedAfterCompaction) {
    // Verify that compaction does not mutate the source graph.
    const vertex_num_t num = 200;
    populate_src_graph(num);

    auto compact = internal_graph_compactor_t::from_internal_graph(*src_graph_, src_max_nbr_size);

    EXPECT_EQ(src_graph_->get_num_vertices(), num);
    for (vertex_num_t i = 0; i < num; ++i) {
        const vertex_id_t v = static_cast<vertex_id_t>(i);
        const uint64_t valid_count = i % (src_max_nbr_size + 1);

        EXPECT_EQ(src_graph_->num_valid_nbrs(v), valid_count);
        EXPECT_EQ(src_graph_->get_inter_layer_link(v), i + 7);

        auto src_block = src_graph_->fetch_nbrs(v);
        for (uint64_t j = 0; j < valid_count; ++j) {
            EXPECT_EQ(src_block[1 + j].base_vid, i * 1000 + static_cast<vertex_id_t>(j));
            EXPECT_EQ(src_block[1 + j].layer_vid, static_cast<vertex_id_t>(j));
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
