# Artea

High performance graph based approximate nearest negihbors index construction and search library on CPUs/GPUs.

## How to test

initialize intel oneapi:

```sh
# source /path/to/your/oneapi/setvars.sh
source $HOME/intel/oneapi/setvars.sh
```


```sh
rm -rf build    # Optional: clean previous builds
# Assuming you are using intel oneapi compiler
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_COMPILER=$HOME/intel/oneapi/compiler/latest/bin/icpx \
      -D CMAKE_C_COMPILER=$HOME/intel/oneapi/compiler/latest/bin/icx

# If you do not have intel oneapi compiler, you can use gcc/g++ instead
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -D CMAKE_CXX_COMPILER=/usr/local/gcc-14/bin/g++ \
      -D CMAKE_C_COMPILER=/usr/local/gcc-14/bin/gcc

cmake --build build -j16
./build/tests/test_simd_distance    # e.g. test the SIMD distance implementation
```
