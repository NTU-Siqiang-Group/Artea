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

The application follows `unit_tests/test_dataset_prober.cpp` and prints the same
13 quantiles, from 0.0001 to 0.9999. Select statistics with `--mode`:

| Mode | Output |
| --- | --- |
| `nearest` | Sample base vertices, scan all other base vectors for their top 128 neighbors, and report a rank-by-quantile table plus LID. |
| `query` | Recompute distances to the configured ground-truth IDs for every query, and report a rank-by-quantile table. |
| `farthest` | Sample base vertices, scan for each vertex's farthest distance, and report its quantiles and sample minimum/maximum. |
| `aspect-ratio` | Report max farthest / 0.0001-quantile nearest and median farthest / median nearest. |
| `all` | Run all statistics, reusing the nearest/farthest results for the aspect ratios. |

Nearest-neighbor probing requires at least 129 base vectors and excludes the
sampled vertex itself. Sampling is with replacement; nearest and farthest probes
sample independently. `--num-samples` controls each base probe, while query probing
always uses all queries and requires ground truth generated for the selected
distance ordering. The existing dataset loader loads base, query, and ground-truth
files in every mode.

Distances, LID, and aspect ratios use the selected distance values directly,
matching `DatasetProber`; squared distances are not automatically square-rooted.
The extreme aspect ratio uses a low quantile as a closest-pair estimate, not an
exact minimum over the entire dataset. A non-positive nearest-distance denominator
is reported as an infinite ratio.
