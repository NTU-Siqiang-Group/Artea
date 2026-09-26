# bin/

This directory contains symbolic links to compiled executables from `build/apps/`.

The symlinks are automatically created by CMake during the build process, allowing you to run applications directly from the project root:

```bash
./bin/probe_dataset
./bin/build_conv_graph
```

instead of:

```bash
./build/apps/probe_dataset
./build/apps/build_conv_graph
```

## Dataset probing

Build and run `probe_dataset` from the ARTEA root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target probe_dataset -j
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all ./bin/probe_dataset \
    --dataset sift-1m --metric euclidean --num-samples 1000
```

Use `--config path/to/datasets.json` to override the dataset configuration.
The defaults are `--dataset sift-1m --metric euclidean_sqr --num-samples 1000 --mode all`.
Both `euclidean` / `l2` and `euclidean_sqr` / `l2_sqr` are supported, as are
`inner_product` and `cosine`.

For modes that report distance quantiles, the application follows
`unit_tests/test_dataset_prober.cpp` and prints the same 13 quantiles, from 0.0001
to 0.9999. Select statistics with `--mode`:

| Mode | Output |
| --- | --- |
| `lid` | Load base and query vectors, and report only fixed query-based mean RVE-LID without computing distance quantiles or other statistics. |
| `nearest` | Sample base vertices, scan all other base vectors for their top 128 neighbors, and report a rank-by-quantile table plus fixed query-based mean RVE-LID. |
| `query` | Recompute distances to the configured ground-truth IDs for every query, and report a rank-by-quantile table. |
| `farthest` | Sample base vertices, scan for each vertex's farthest distance, and report its quantiles and sample minimum/maximum. |
| `aspect-ratio` | Report fixed query-based mean RVE-LID, max farthest / 0.0001-quantile nearest, and median farthest / median nearest. |
| `all` | Run all statistics, reusing the nearest/farthest results for the aspect ratios. |

Base nearest-neighbor probing requires at least 129 base vectors and excludes the
sampled vertex itself. Sampling is with replacement; nearest and farthest probes
sample independently. `--num-samples` controls each base probe, while query probing
always uses all queries and requires ground truth generated for the selected
distance ordering. The `lid` mode loads base and query vectors and does not require
`gt_path` or a ground-truth file. Other modes load base, query, and ground-truth files.

Every mode that reports LID (`lid`, `nearest`, `aspect-ratio`, and `all`) uses the
same **FIXED** configuration, defined in `DatasetProber`:

- Sample exactly **500 distinct query IDs**, without replacement, with seed **42**.
- Find each sampled query's exact **1000 nearest neighbors in the full base set**.
  Query IDs and base IDs are independent; matching IDs do not exclude a neighbor.
- Compute **RVE-LID (RV, J=2)** using the sorted distances at 1-based ranks
  **500, 750, 1000**, and take the arithmetic mean of all **500** query estimates.
- Fewer than 500 query vectors or 1000 base vectors is an error. Undefined
  per-query estimates also cause an error rather than being dropped from the mean.
  Distances must be finite and non-negative, with `r_500 > 0` and
  `r_1000 > r_500`. Zero distances remain part of the 1000 neighbors and do not
  shift the fixed ranks. There is no epsilon clamp.

The estimator follows the three-anchor RV variant used by
[ELKI RVEstimator](https://github.com/elki-project/elki/blob/main/elki-core-math/src/main/java/elki/math/statistics/intrinsicdimensionality/RVEstimator.java),
based on [Amsaleg et al., KDD 2015, Sections 4.4 and 5.1](https://doi.org/10.1145/2783258.2783405).
For `i=500`, `j=750`, `k=1000`, define `a=r_j-r_i`, `b=r_k-r_j`:

```text
RVE-LID(q) = [b*log(k/j) + a*log(j/i)] / [b*log(r_k/r_j) + a*log(r_j/r_i)]
```

The implementation uses this stabilized form with both gaps scaled by `r_k`,
avoiding division by a second difference or an adjacent distance gap. It also
handles equally spaced anchors and the finite limit when one adjacent pair ties.

These counts and the seed have no runtime overrides. `--mode lid` rejects
`--num-samples` / `-n`; in other modes that option controls only the base distance
probes and never changes the LID configuration.

To compute only LID, using Euclidean distances:

```sh
OMP_NUM_THREADS="$(nproc)" numactl --interleave=all ./bin/probe_dataset \
    --dataset sift-1m --metric euclidean --mode lid
```

From the **benchmark repository root**, the Python batch wrapper runs the same
executable with NUMA interleave and all available OpenMP threads:

```sh
python3 scripts/compute_lid.py --datasets sift-1m gist-1m --output lid.json
```

The wrapper defaults to Euclidean distances, supports `--config` and `--metric`,
and has no estimator, sample-count, neighbor-count, or seed overrides.

Distances, LID, and aspect ratios use the selected distance values directly,
matching `DatasetProber`; squared distances are not automatically square-rooted.
The extreme aspect ratio uses a low quantile as a closest-pair estimate, not an
exact minimum over the entire dataset. A non-positive nearest-distance denominator
is reported as an infinite ratio.
