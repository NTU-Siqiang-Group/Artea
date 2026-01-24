# Test Documentation

- To test `BruteforceRouter`, run:

```bash
cd Artea
# ./build/tests/test_bruteforce_router -c <datasets config> -d <dataset name>
./build/tests/test_bruteforce_router -c ./datasets.json -d sift-1m
```

- To test `RandomSeq`, run:

```bash
cd Artea
./build/tests/test_random_seq -n 1024 -s 1000000
```

- To test `SIMD Distance Functions`, run:

```bash
cd Artea
./build/tests/test_simd_distance -c ./datasets.json -d sift-1m
```

- To test `SIMD FMA Functions`, run:

```bash
cd Artea
./build/tests/test_simd_fma -c ./datasets.json -d sift-1m
```