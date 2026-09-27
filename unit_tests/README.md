# Test Documentation

## Compaction Capacity Baseline (Step 01)

`test_hierarchical_graph_compactor` uses synthetic vectors and ordered neighbor
lists, with no dataset download. It covers empty and L0-only graphs, retained
layers (including the exact apex threshold and an empty intermediate group),
consecutive top-layer trimming, demotion order, sentinel tails, and a unique
centroid entry point. It also checks that the source remains usable after
compaction. The reusable input lives in `compaction_baseline_fixture.hpp`.

The segmented fixture uses the production 2,048-slot blocks and 8-slot TLS
reservations. A worker consumes five reserved slots; the main thread then
crosses a block boundary. These two allocation contexts make holes deterministic
and are a correctness fixture, not a performance thread setting. The source
L1 apex bucket contains 2,056 vertices, its reservation high-water mark is 2,064,
and its allocated capacity is 4,096. Two L2 vertices are demoted to L1.
The current compactor copies the existing offsets and appends demoted vertices
after capacity. The test records this behavior; it does **not** fix it.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_hierarchical_graph_compactor -j
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all ./build/unit_tests/test_hierarchical_graph_compactor
```

To retain pre-repair version-1 snapshots and JSON reports of coordinates,
source/final buckets, ordered neighbors, capacities, reservation high-water
marks, vertex layers, offsets, and entry points, set
`ARTEA_COMPACTION_BASELINE_DIR` to a fresh directory under
`temp/validation/compact-capacity/<experiment>/<run>/` for the test process.
Without this variable the artifact-capture test is skipped. Save the source Git
revision, launch command, CPU affinity, NUMA policy, thread setting, test logs,
and snapshot checksums alongside these artifacts before changing the compactor.
Reports record source neighbor rows; final rows are those prefixes at retained
levels, checked independently by the topology assertions.

The future dense-layout requirement is deliberately disabled in normal runs.
Run it separately to reproduce the expected failure on the old compactor:

```sh
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all ./build/unit_tests/test_hierarchical_graph_compactor \
  --gtest_also_run_disabled_tests \
  --gtest_filter=CompactionBaseline.DISABLED_DenseSlotsFollowFinalBucketOrder
```

This command must fail before the step-02 repair. Do not count it as a passing
test. Step 02 should enable the dense test and replace the transitional assertions
in `SegmentedSourceRetainsCapacityAndReservationHoles`; logical topology assertions
remain valid regardless of physical offsets.

- To test `BruteforceRouter`, run:

```bash
cd Artea
# ./build/tests/test_bruteforce_router -c <datasets config> -d <dataset name>
./build/tests/test_bruteforce_router -c ./configs/datasets.json -d sift-1m
```

- To test `RandomSeq`, run:

```bash
cd Artea
./build/tests/test_random_seq -n 1024 -s 1000000
```

- To test `SIMD Distance Functions`, run:

```bash
cd Artea
./build/tests/test_simd_distance -c ./configs/datasets.json -d sift-1m
```

## About Metrics

Graph tests use Euclidean distance for construction and squared Euclidean distance
for queries after compaction. Build-and-search tests use separate
`build_infra_dispatch` and `search_infra_dispatch` calls, passing the compact graph
between them. Cosine and inner-product tasks keep their original metric.
Dynamic construction-router comparisons stay in build-distance units; their
compact-graph counterparts use the search metric. Distance-kernel tests and
standalone prober tests keep explicit control over the metric.

`test_stage_distance` is self-contained: it checks all Euclidean aliases and
supported dimensions, then builds a graph, verifies its stored L2 edge weights,
compacts it, and verifies query IDs and squared result distances. Run it with:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_stage_distance -j
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all ./build/unit_tests/test_stage_distance
```

## R-net Geometry

Graph construction sets its radius geometry using `rnet_beta > 1`
(default `2.0`) and `num_skipped_levels >= 0` (integer, default `0`):
`R0 = l0_min_distance`, and
`Rh = R0 * rnet_beta^(num_skipped_levels + h)` for `h >= 1`.
`tau = shifted_coeffs >= 0` (default `0`) controls ARTEA's L0 refinement shift
and does not affect these radii.
The L0 distance scale stays unchanged; `scale_coeffs` does not affect radii. ARTEA and stacked r-nets share `stacked_rgraph::RGraphConfig`.
All distances use build-distance units; the same L0 scale controls ARTEA's
refinement shift.

`test_artea_graph` reads these values from the workload's `indexes-config.artea`
entry. CLI tests `test_stacked_rgraph` and `test_hierarchical_graph_persistence`
accept `--beta`, `--num-skipped-levels`, `--shifted-coeffs`, `--scale-coeffs`,
and `--l0-min-distance`.
Their distance probe can supply `l0_min_distance` when a negative value is given.
`test_stage_distance` checks skipped levels, fractional beta, geometric radius
growth, tau scaling, invalid parameters, L1-radius overflow, actual L1 membership, and the
shared ARTEA/stacked-rnet configuration type.
