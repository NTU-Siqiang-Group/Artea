# Artea

High performance graph based approximate nearest negihbors index construction and search library on CPUs/GPUs.

## How to test

```sh
rm -rf build    # Optional: clean previous builds
cmake -B build
cmake --build build -j
./build/tests/test_simd_distance    # e.g. test the SIMD distance implementation
```
