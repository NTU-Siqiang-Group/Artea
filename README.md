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
# activate oneapi support first
bash ./scripts/install.sh
./build/unit_tests/test_artea_graph --help
```


