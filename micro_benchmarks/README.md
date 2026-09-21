# Artea Micro Benchmarks

This directory contains micro-benchmarks for various components of the Artea library.

## Available Benchmarks

### 1. bench_random_eg
Benchmarks the RandomEG (Random Edge Generator) for graph initialization.

**Purpose**: Measures the time to randomly initialize a graph's edges.

**Parameters**:
- `-c, --config`: Path to dataset configuration file (default: `./configs/datasets.json`)
- `-d, --dataset`: Dataset name (default: `sift-1m`)
- `--init-nbrs`: Number of random neighbors to generate for each vertex (default: 32)
- `-i, --iterations`: Number of benchmark iterations (default: 10)

**Example**:
```bash
./build/micro_benchmarks/bench_random_eg -c ./configs/datasets.json --dataset sift-1m --init-nbrs 32 -i 10
```

---

### 2. bench_propagate_engine
Benchmarks the PropagateEngine with TriangleUpdater for RNG pruning.

**Purpose**: Measures the propagation time for running multiple iterations of edge generation with triangle inequality pruning.

**Parameters**:
- `-c, --config`: Path to dataset configuration file (default: `./configs/datasets.json`)
- `-d, --dataset`: Dataset name (default: `sift-1m`)
- `--init-nbrs`: Number of random neighbors for initial graph (default: 32)
- `--max-nbrs`: Maximum neighbor size after pruning (default: 16)
- `--num-iters`: Number of propagation iterations to run (default: 5)
- `--scale-coeffs`: Scale coefficient for triangle inequality pruning (default: 1.0)
- `--shifted-coeffs`: Shifted coefficient for triangle inequality pruning (default: 0.0)
- `-i, --iterations`: Number of benchmark iterations (default: 10)

**Example**:
```bash
./build/micro_benchmarks/bench_propagate_engine -c ./configs/datasets.json --dataset sift-1m --init-nbrs 32 --max-nbrs 64 --num-iters 5 -i 10
```

**Note**: Each benchmark iteration resets the graph to its initial random state, ensuring consistent and reproducible measurements. Within each iteration, `propagate_engine.run()` performs the specified number of propagation iterations on the same graph.

---

### 3. bench_reverse_updater
Benchmarks the PropagateEngine with ReverseUpdater for bidirectional edge generation.

**Purpose**: Measures the time to add reverse edges to a sparse directed graph, converting it into a bidirectional graph.

**Parameters**:
- `-c, --config`: Path to dataset configuration file (default: `./configs/datasets.json`)
- `-d, --dataset`: Dataset name (default: `sift-1m`)
- `--init-nbrs`: Number of random neighbors for initial sparse graph (default: 32)
- `--num-iters`: Number of propagation iterations to run (default: 1)
- `-i, --iterations`: Number of benchmark iterations (default: 10)

**Example**:
```bash
./build/micro_benchmarks/bench_reverse_updater -c ./configs/datasets.json --dataset sift-1m --init-nbrs 32 --num-iters 1 -i 10
```

**Note**: This benchmark starts with a sparse random graph and adds reverse edges to make it bidirectional. Each benchmark iteration resets the graph to its initial sparse state before running the reverse updater.

---

### 4. bench_radius_prober
Benchmarks the radius probing functionality.

**Purpose**: Measures the performance of radius-based neighbor probing.

---

### 5. bench_simd_distance
Benchmarks SIMD-optimized distance calculations.

**Purpose**: Measures the performance of vectorized Euclidean, squared Euclidean, inner-product, and cosine distance computations.

Select the distance with `--metric euclidean` (alias `l2`),
`--metric euclidean_sqr` (alias `l2_sqr`),
`--metric inner_product`, or `--metric cosine`. Result names contain the canonical
metric, for example `Artea_euclidean_U1`, `Artea_euclidean_sqr_U1`, `Artea_inner_product_U1`, and
`Artea_cosine_U1`; parallel cases add the `Par_` prefix. The scalar and standard
SIMD reference kernels are registered only for `euclidean_sqr`, which is the
distance they implement. Google Benchmark options such as
`--benchmark_filter` and `--benchmark_list_tests=true` can be combined with
the dataset and metric options.

---

### 6. bench_vertex_generator
Benchmarks vertex generation strategies.

**Purpose**: Measures the performance of various vertex generator implementations.

---

## Running Benchmarks

### Quick Run Script

Use the provided `run_bench.sh` script to quickly run any benchmark with default parameters:

```bash
# Run a benchmark with default parameters
./run_bench.sh bench_random_eg

# Run with custom parameters
./run_bench.sh bench_propagate_engine --num-iters 10 --max-nbrs 20
```

### Manual Execution

All benchmarks are built in the `build/micro_benchmarks/` directory. You can run them directly:

```bash
cd build/micro_benchmarks/
./bench_random_eg --help          # Show help message
./bench_random_eg                 # Run with default parameters
```

## Common Parameters

Most benchmarks share these common parameters:

- `-c, --config`: Path to dataset configuration JSON file
- `-d, --dataset`: Dataset name (defined in the config file)
- `-i, --iterations`: Number of benchmark iterations
- `-h, --help`: Display help message

## Dataset Configuration

Benchmarks expect a `datasets.json` file in the working directory. Example format:

```json
{
  "sift-1m": {
    "base": "path/to/sift_base.fvecs",
    "query": "path/to/sift_query.fvecs",
    "groundtruth": "path/to/sift_groundtruth.ivecs"
  }
}
```

## Build Requirements

- CMake >= 3.16
- C++20 compiler (GCC 10+, Clang 12+)
- Google Benchmark library
- Intel TBB (Threading Building Blocks)
- AVX-512 support (for SIMD benchmarks)

## Notes

- All benchmarks use Google Benchmark framework
- Results are displayed in milliseconds by default
- Each benchmark includes throughput metrics (items processed per second)
- Use `--benchmark_filter=<pattern>` to run specific benchmark cases
- Use `--benchmark_format=<json|csv|console>` to change output format
