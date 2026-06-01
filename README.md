# Artea: High performance graph based approximate nearest neighbors index construction and search library.

## Installation and Tests

1. Step 1: Initialize intel oneapi environment variables:

```sh
# source /path/to/your/oneapi/setvars.sh
source $HOME/intel/oneapi/setvars.sh
```

2. Step 2: Build and run tests:

```sh
rm -rf build    # Optional: clean previous builds
# Assuming you are using intel oneapi compiler
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_COMPILER=$HOME/intel/oneapi/compiler/latest/bin/icpx \
      -D CMAKE_C_COMPILER=$HOME/intel/oneapi/compiler/latest/bin/icx

# debug mode
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
      -D CMAKE_CXX_COMPILER=$HOME/intel/oneapi/compiler/latest/bin/icpx \
      -D CMAKE_C_COMPILER=$HOME/intel/oneapi/compiler/latest/bin/icx

# If you do not have intel oneapi compiler, you can use gcc/g++ alternatively
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_COMPILER=/usr/local/gcc-14/bin/g++ \
      -D CMAKE_C_COMPILER=/usr/local/gcc-14/bin/gcc

cmake --build build -j64
./build/unit_tests/test_simd_distance -c datasets.json -d sift-1m    # e.g. test the SIMD distance implementation
```

---

## Benchmarking tip: NUMA page-cache saturation can silently slow runs

On a multi-socket (NUMA) machine the benchmark runs under `numactl --interleave=all`,
which assumes every node has balanced **free** memory. Reading the large dataset files
fills the OS page cache; if one NUMA node's free memory drops to near zero (almost all
of it reclaimable `FilePages`), interleaved allocations destined for that node force
synchronous page reclaim. This uniformly slows **both** index build (allocation-heavy —
observed +40%) and search throughput (memory-latency-bound — ~15% lower) while producing
**byte-identical** results (same recall, same index size) — so it looks like a
performance regression but isn't.

**Symptom:** the current run is uniformly slower than a stored/legacy baseline at every
operating point *and* the build is slower too, yet recall and index size are unchanged.

**Check** per-node free memory (look for a node with very little free):

```bash
numactl --hardware
grep -E 'MemFree|FilePages' /sys/devices/system/node/node*/meminfo
```

**Fix** — drop the page cache so every node regains real free memory, then re-run:

```bash
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

The cache refills harmlessly as the datasets are re-read.



