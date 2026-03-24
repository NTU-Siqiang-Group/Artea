# Workloads Directory

This directory contains search workload configuration files for benchmarking.

## File Format

Workload files use JSON format with the following structure:

```json
[
  {
    "algorithm": "conv_graph",
    "file_path": "graph_index_repo/artea/conv_graph.sift-1m/mns32_ens20_sc1.00_sh0.00_noi4_nii14.index",
    "params": {
      "dataset": "sift-1m",
      "extracted_nbr_size": 20,
      "max_nbr_size": 32,
      "num_triu_iters": 14,
      "num_outer_iters": 4,
      "scale_coeffs": 1.0,
      "shifted_coeffs": 0.0
    }
  }
]
```

**Important**: The `file_path` field should be a relative path from the project root directory.

## Usage

Use workload files with benchmark programs:

```bash
./build/micro_benchmarks/bench_proximity_graph_router \
  --workload workloads/search_workload.json \
  --topk 10 \
  --candidate-queue-sizes 32 64
```

The benchmark will automatically load all indices specified in the workload file and test them on their corresponding datasets.
