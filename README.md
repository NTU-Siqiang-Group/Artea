# Artea

ANN Search on CPU/GPU

## How to test

```sh
rm -rf build    # Optional: clean previous builds
cmake -B build
cmake --build build -j
./build/tests/test_simd_distance    # e.g. test the SIMD distance implementation
```
