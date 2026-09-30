# Test Documentation

## Construction regressions

`test_graph_build_regressions` uses synthetic graphs without external datasets.
It checks that unordered hierarchical neighbor rows are sorted by `(distance, ID)`
before refinement, including tied distances, sparse upper-layer ID mapping, log
merging without duplicate edges, and truncation to a smaller destination capacity.
An incremental insertion fixture also verifies that widening the construction
queue from 30 to 100 revisits a previously rejected candidate and discovers the
true nearest neighbor behind it. The fixture has a unique apex and inserts one
vertex through the production factory, so it does not depend on scheduling.

Layer-growth regressions synchronize two insertions after both have captured
the same old top. They check L2/L3 growth with and without L0 insertion: the
losing creator must retry, connect at the new layer, or lower its insertion
level if the new layer covers it. A test-only index substitution controls this
interleaving with a preallocated batch and two explicit caller threads, so the
barrier does not depend on TBB splitting a tiny range. Production code has no
test hooks. The tests also check that new
tops are published after their lower-layer edges and that captured entry views
remain stable across publication. The race cases require at least two threads;
the test executable honors `OMP_NUM_THREADS` for its TBB concurrency limit.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_graph_build_regressions -j
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all ./build/unit_tests/test_graph_build_regressions
```

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

`test_artea_graph` and `test_stacked_rgraph` read these values from the workload's
`indexes-config.artea` entry. Their distance probe supplies `l0_min_distance` when
that field is omitted. `test_hierarchical_graph_persistence` accepts `--beta`,
`--tau-k`, `--shifted-coeffs`, `--scale-coeffs`, and `--l0-min-distance`, with a
negative `--l0-min-distance` enabling its distance probe.
`test_stage_distance` checks fractional tau_k and beta, geometric radius
growth, independence from the pruning shift, invalid parameters, L1-radius overflow, actual L1 membership, and the
shared ARTEA/stacked-rnet configuration type.

To inspect the stacked r-net's level distribution, run from the Artea repository root:

```sh
cmake --build build --target test_stacked_rgraph -j "$(nproc)"
OMP_NUM_THREADS="$(nproc)" OMP_DYNAMIC=FALSE numactl --interleave=all \
    ./build/unit_tests/test_stacked_rgraph --workload ./workloads/sift1m-bench.jsonc
```

`test_stacked_rgraph` accepts only `-w`/`--workload` for configuration; the former individual build flags
are no longer supported. JSON and JSONC comments are supported. Dataset paths are resolved relative to the
working directory, as in the benchmark. An object-valued `indexes-config.artea` is used directly; an array
must contain exactly one untagged base entry, with tagged variants ignored. No search sweep is required.
Only one graph is built, honoring `insert_on_L0` and `shuffle_insertion_order` (both default to `false`).
The output includes build time, cumulative vertex counts at each level, and a compactor trim preview.
This builds the stacked r-net backbone; ARTEA's L0 refinement and queries are not run.
