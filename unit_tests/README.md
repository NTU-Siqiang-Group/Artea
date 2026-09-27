# Test Documentation

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
