#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea::cpu;

namespace {

using traits_t = computer_traits_t<DistanceMetricsT::EUCLIDEAN, 96>;
using vectors_t = traits_t::vector_array_t;
using queries_t = traits_t::query_vecs_t;

vectors_t make_base(uint32_t count = 1200) {
    vectors_t base(count, 96);
    for (uint32_t i = 0; i < count; ++i) {
        std::fill_n(base.get(i), 96, 0.0f);
        base.get(i)[0] = static_cast<float>(i + 1);
    }
    return base;
}

queries_t make_queries(uint32_t count = 500, uint32_t dim = 96) {
    queries_t queries(count, dim);
    for (uint32_t i = 0; i < count; ++i) {
        std::fill_n(queries.get(i), dim, 0.0f);
    }
    return queries;
}

double reference_lid(double offset) {
    // Equally spaced anchors have equal gaps; RV(J=2) reduces to this ratio.
    return std::log(2.0) / std::log((1000.0 + offset) / (500.0 + offset));
}

// Record which query IDs were evaluated, independently of the sampling implementation.
struct RecordingTraits : traits_t {
    struct dist_func_t {
        std::vector<std::atomic<uint32_t>>& calls;
        float operator()(const float* query, const float* base) const {
            calls[static_cast<uint32_t>(query[1])].fetch_add(1, std::memory_order_relaxed);
            return std::abs(base[0] - query[0]);
        }
    };
};

} // namespace

TEST(FixedRveLid, ArithmeticMeanOfQueryEstimatesUsingExactly1000NearestNeighbors) {
    auto base = make_base();
    auto queries = make_queries();
    // Two query populations have different estimates: the result must be the
    // mean of their individual RVE-LIDs, not an estimate from pooled distances.
    for (uint32_t i = 250; i < 500; ++i) queries.get(i)[0] = -1000.0f;
    traits_t::dist_func_t distance;
    DatasetProber<traits_t> prober(base, distance);
    const double expected = (reference_lid(0.0) + reference_lid(1000.0)) / 2.0;
    EXPECT_NEAR(prober.probe_lid(queries), expected, 1e-12);
}

TEST(FixedRveLid, Samples500DistinctQueryIdsReproduciblyAndScansFullBase) {
    auto base = make_base();
    auto queries = make_queries(750);
    for (uint32_t i = 0; i < 750; ++i) queries.get(i)[1] = static_cast<float>(i);
    std::vector<std::atomic<uint32_t>> calls(750);
    for (auto& count : calls) count.store(0);
    RecordingTraits::dist_func_t distance{calls};
    DatasetProber<RecordingTraits> prober(base, distance);
    std::vector<uint32_t> previous_ids;
    for (int repeat = 0; repeat < 2; ++repeat) {
        EXPECT_NEAR(prober.probe_lid(queries), reference_lid(0.0), 1e-12);
        std::vector<uint32_t> ids;
        for (uint32_t i = 0; i < 750; ++i) {
            const auto count = calls[i].exchange(0);
            if (count != 0) {
                EXPECT_EQ(count, 1200u); // No resampling and no query/base ID exclusion.
                ids.push_back(i);
            }
        }
        EXPECT_EQ(ids.size(), 500u);
        if (repeat != 0) EXPECT_EQ(ids, previous_ids);
        previous_ids = ids;
    }
}

TEST(FixedRveLid, RejectsInsufficientQueriesBaseAndMismatchedDimensions) {
    auto base = make_base(1000);
    auto queries = make_queries(499);
    traits_t::dist_func_t distance;
    DatasetProber<traits_t> prober(base, distance);
    EXPECT_THROW(prober.probe_lid(queries), std::runtime_error);
    queries = make_queries(500, 112);
    EXPECT_THROW(prober.probe_lid(queries), std::runtime_error);
    queries = make_queries();
    auto small_base = make_base(999);
    DatasetProber<traits_t> small_prober(small_base, distance);
    EXPECT_THROW(small_prober.probe_lid(queries), std::runtime_error);
    EXPECT_NEAR(prober.probe_lid(queries), reference_lid(0.0), 1e-12);
}

TEST(FixedRveLid, RejectsUndefinedEstimatesInsteadOfDroppingQueriesFromMean) {
    auto base = make_base(1000);
    auto queries = make_queries();
    traits_t::dist_func_t distance;
    DatasetProber<traits_t> prober(base, distance);
    for (uint32_t i = 0; i < 1000; ++i) base.get(i)[0] = 1.0f;
    EXPECT_THROW(prober.probe_lid(queries), std::runtime_error);
    for (uint32_t i = 0; i < 1000; ++i) base.get(i)[0] = 0.0f;
    EXPECT_THROW(prober.probe_lid(queries), std::runtime_error);
}

TEST(FixedRveLid, UsesOnlyRanks5007501000IncludingZeroDistances) {
    auto base = make_base();
    auto queries = make_queries();
    traits_t::dist_func_t distance;
    DatasetProber<traits_t> prober(base, distance);
    // Alter every distance below the first anchor without changing any anchor.
    // Zero distances must not shift the fixed 1-based ranks.
    for (uint32_t i = 0; i < 499; ++i) base.get(i)[0] = 0.0f;
    EXPECT_NEAR(prober.probe_lid(queries), 1.0, 1e-12);
    base.get(499)[0] = 0.0f;
    EXPECT_THROW(prober.probe_lid(queries), std::runtime_error);
}

TEST(FixedRveLid, RecoversPowerLawDimensionAndPreservesDistanceScale) {
    auto base = make_base();
    auto queries = make_queries();
    traits_t::dist_func_t distance;
    DatasetProber<traits_t> prober(base, distance);
    for (const double scale : {1e-12, 1.0, 1e10}) {
        for (uint32_t i = 0; i < 1200; ++i) {
            base.get(i)[0] = static_cast<float>(scale * (i + 1));
        }
        EXPECT_NEAR(prober.probe_lid(queries), 1.0, 2e-6);
    }
    // F(r) proportional to sqrt(r) has intrinsic dimension 1/2.
    for (uint32_t i = 0; i < 1200; ++i) {
        base.get(i)[0] = static_cast<float>((i + 1) * (i + 1));
    }
    EXPECT_NEAR(prober.probe_lid(queries), 0.5, 2e-6);
}

TEST(FixedRveLid, HandlesTheFiniteLimitWhenOneAdjacentAnchorPairTies) {
    auto base = make_base(1000);
    auto queries = make_queries();
    traits_t::dist_func_t distance;
    DatasetProber<traits_t> prober(base, distance);
    for (uint32_t i = 0; i < 1000; ++i) base.get(i)[0] = i < 750 ? 100.0f : 200.0f;
    EXPECT_NEAR(prober.probe_lid(queries), std::log(4.0 / 3.0) / std::log(2.0), 1e-12);
    for (uint32_t i = 0; i < 1000; ++i) base.get(i)[0] = i < 500 ? 100.0f : 200.0f;
    EXPECT_NEAR(prober.probe_lid(queries), std::log(1.5) / std::log(2.0), 1e-12);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
