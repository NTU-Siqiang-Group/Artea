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

Graph construction uses `R1 = l0_min_distance * rnet_beta^(num_skip_levels + 1)`,
followed by `Rh = l0_min_distance * rnet_beta^(num_skip_levels + h)` for `h >= 1`.
`num_skip_levels` is a nonnegative integer defaulting to `0` in workload and
CLI runners. `l0_min_distance` is expressed in build-distance
units and also scales ARTEA's L0 refinement shift. `radius_at(0)` returns this
base distance scale; L0 itself is not an r-net.

`test_artea_graph` reads these values from the workload's `indexes-config.artea`
entry. CLI tests `test_stacked_rgraph` and `test_hierarchical_graph_persistence`
accept `--num-skip-levels`, `--l0-min-distance`, and `--beta`. Their distance probe
can supply `l0_min_distance` when a negative value is given.
`test_stage_distance` also checks the radius progression and verifies that
L1 membership changes with `num_skip_levels` and `rnet_beta`, including the
zero-skip case, non-integer growth factors, and L1-radius overflow rejection.
