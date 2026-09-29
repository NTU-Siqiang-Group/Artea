# Test Documentation

## Grouped CSR Compaction

`test_hierarchical_graph_compactor` uses synthetic vectors and ordered neighbor lists without dataset downloads.
It covers empty and L0-only graphs, retained layers, the apex threshold, empty intermediate groups, consecutive
trimming, demotion order, and a unique centroid entry point. The reusable input is `compaction_baseline_fixture.hpp`.

Vertices are grouped by their final highest level `h`. `VertexInfo` contains two 32-bit fields:
`highest_level` and `local_vid`. Group `h` stores `N_h * (h + 1) + 1` global 64-bit offsets. A vertex's rows are
ordered `h, h-1, ..., 0`, with row index `local_vid * (h + 1) + (h - level_id)`. The global neighbor array contains
only valid 32-bit vertex IDs. Empty rows share consecutive offsets; compact spans contain no sentinel tails.
`allocated_storage_bytes()` reports vector allocation capacities, excluding the graph object and allocator overhead.

The segmented fixture uses production 2,048-slot blocks and 8-slot TLS reservations. A worker consumes five
reserved slots; the main thread then crosses a block boundary. These allocation contexts create deterministic
holes and are a correctness fixture, not a performance thread setting. The source L1 apex bucket has 2,056
vertices, a reservation high-water mark of 2,064 and capacity 4,096. Two L2 vertices are demoted. CSR assigns the
final 2,058 L1 vertices consecutive local IDs and stores only their valid neighbors. Source offsets and capacity
never determine compact storage.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_hierarchical_graph_compactor test_compact_csr_query -j
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all ./build/unit_tests/test_hierarchical_graph_compactor
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all ./build/unit_tests/test_compact_csr_query
```

Every synthetic case checks exact CSR row counts, monotonic offsets, disjoint row spans, group membership,
neighbor order, unchanged source topology, and version-1 snapshot round trips. Additional tests exercise
zero-neighbor arrays, unassigned records, allocation overflow, malformed/truncated files, and cleanup after an
injected centroid-distance exception. Run with ASan/UBSan to verify memory access and cleanup as well.

`test_compact_csr_query` compares greedy-upper / L0-beam top-10 results against independent exact sorting,
including tied distances. It checks IDs and distances across dynamic/compact graphs, restore, and batch queries.
Prefetch cases cover batch sizes 4/8/16, traversal order, duplicates, previsited neighbors, empty lists,
partial batches, sentinel termination, and scalar visited-mark timing for noncontiguous adapters.
Round-trip test artifacts stay under the ignored `temp/validation/compact-csr/` directory.

Set `ARTEA_COMPACTION_LEGACY_DIR` to the retained step-01 sample directory to restore all eight pre-CSR version-1
files and check their topology and byte-identical reserialization. Set `ARTEA_COMPACTION_BASELINE_DIR` to a fresh
validation directory to capture current snapshots and JSON layout reports. Both artifact tests otherwise skip.
Keep historical samples intact, and record source revision, launch command, CPU affinity, NUMA policy, thread
setting and checksums with each experiment. Snapshot version 1 and its ordered logical records are unchanged.

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
(default `2.0`) and `tau_k >= 0` (floating point, default `0.0`):
`R0 = l0_min_distance`, and
`Rh = R0 * (1 + tau_k) * rnet_beta^(h - 1)` for `h >= 1`.
`tau = shifted_coeffs >= 0` (default `0`) controls ARTEA's L0 refinement shift
and does not affect these radii.
The L0 distance scale stays unchanged; `scale_coeffs` does not affect radii. ARTEA and stacked r-nets share `stacked_rgraph::RGraphConfig`.
All distances use build-distance units; the same L0 scale controls ARTEA's
refinement shift.

`test_artea_graph` reads these values from the workload's `indexes-config.artea`
entry. CLI tests `test_stacked_rgraph` and `test_hierarchical_graph_persistence`
accept `--beta`, `--tau-k`, `--shifted-coeffs`, `--scale-coeffs`,
and `--l0-min-distance`.
Their distance probe can supply `l0_min_distance` when a negative value is given.
`test_stage_distance` checks fractional tau_k and beta, geometric radius
growth, independence from the pruning shift, invalid parameters, L1-radius overflow, actual L1 membership, and the
shared ARTEA/stacked-rnet configuration type.
