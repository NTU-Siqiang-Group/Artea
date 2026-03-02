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
- `--reserved-nbrs`: Reserved neighbor array size for the graph (default: 32)
- `-i, --iterations`: Number of benchmark iterations (default: 10)

**Example**:
```bash
./build/micro_benchmarks/bench_random_eg -c ./configs/datasets.json --dataset sift-1m --init-nbrs 32 --reserved-nbrs 32 -i 10
```

---

### 2. bench_propagate_engine
Benchmarks the PropagateEngine with TriangleUpdater for RNG pruning.

**Purpose**: Measures the propagation time for running multiple iterations of edge generation with triangle inequality pruning.

**Parameters**:
- `-c, --config`: Path to dataset configuration file (default: `./configs/datasets.json`)
- `-d, --dataset`: Dataset name (default: `sift-1m`)
- `--init-nbrs`: Number of random neighbors for initial graph (default: 32)
- `--reserved-nbrs`: Reserved neighbor array size (default: 32)
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
- `--reserved-nbrs`: Reserved neighbor array size (default: 32)
- `--num-iters`: Number of propagation iterations to run (default: 1)
- `-i, --iterations`: Number of benchmark iterations (default: 10)

**Example**:
```bash
./build/micro_benchmarks/bench_reverse_updater -c ./configs/datasets.json --dataset sift-1m --init-nbrs 32 --num-iters 1 -i 10
```

**Note**: This benchmark starts with a sparse random graph and adds reverse edges to make it bidirectional. Each benchmark iteration resets the graph to its initial sparse state before running the reverse updater.

---

### 4. bench_lsh_table
Benchmarks the LSH (Locality-Sensitive Hashing) table operations.

**Purpose**: Measures the performance of LSH hash computation and table operations.

---

### 4. bench_radius_prober
Benchmarks the radius probing functionality.

**Purpose**: Measures the performance of radius-based neighbor probing.

---

### 5. bench_simd_distance
Benchmarks SIMD-optimized distance calculations.

**Purpose**: Measures the performance of vectorized distance computations (Euclidean, Inner Product, etc.).

---

### 6. bench_simd_fma
Benchmarks SIMD FMA (Fused Multiply-Add) operations.

**Purpose**: Measures the performance of FMA operations using SIMD instructions.

---

### 7. bench_simd_linear
Benchmarks SIMD linear operations.

**Purpose**: Measures the performance of linear algebra operations using SIMD.

---

### 8. bench_vertex_generator
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