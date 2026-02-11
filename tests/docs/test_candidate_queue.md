# Candidate Queue Test Documentation

## Overview

Test suite for three candidate queue implementations: `StdCandidateQueue`, `LinearCandidateQueue`, and `FHCandidateQueue`.

**File**: `tests/test_candidate_queue.cpp`

## Test Sections

### 1. Initialize Function Tests (3 tests)
- **Tests**: `Initialize_{Std,Linear,FH}Queue`
- **Logic**: Create K shuffled candidates → call `initialize()` → verify size and sorted order
- **Validates**: Initialization correctly sorts and sets up queue state

### 2. Random Initialize Function Tests (3 tests)
- **Tests**: `RandomInitialize_{Std,Linear,FH}Queue`
- **Logic**: Create 1000 random vectors → call `random_initialize()` → verify size and sorted order
- **Validates**: Integration with distance computation and random sampling

### 3. Basic Sort Tests (3 tests)
- **Tests**: `BasicSort_{Std,Linear,FH}Queue`
- **Logic**: Insert N shuffled candidates → drain via `pop_best_unexplored()` → verify ascending order
- **Validates**: Core sorting functionality

### 4. Capacity & Eviction Tests (3 tests)
- **Tests**: `CapacityEviction_{Std,Linear,FH}Queue`
- **Logic**: Fill to capacity → insert worse (rejected) → insert better (accepted, worst evicted)
- **Validates**: Top-K maintenance and eviction logic

### 5. Extract Results/IDs Tests (9 tests)
- **Extract Results** (3 tests): `ExtractResults_{Std,Linear,FH}Queue`
  - Insert K shuffled → `extract_results()` → verify ascending order and correct distances
- **Extract IDs** (3 tests): `ExtractResultIds_{Std,Linear,FH}Queue`
  - Insert K with ID=distance → `extract_result_ids()` → verify ascending order
- **Consistency** (3 tests): `ExtractConsistency_{Std,Linear,FH}Queue`
  - Two identical queues → compare `extract_results()` vs `extract_result_ids()` → verify IDs match
- **Validates**: Both extraction methods return correctly sorted results

### 6. Cursor Regression Tests (4 tests)
- **Tests**: `CursorRegression_{Std,Linear,FH}Queue`, `CursorRegressionDeep_LinearQueue`
- **Logic**: Pop first → insert better candidate before cursor → verify it's returned next
- **Validates**: LinearQueue cursor correctly regresses when better candidates inserted

### 7. Threshold Logic Tests (3 tests)
- **Tests**: `ThresholdLogic_{Std,Linear,FH}Queue`
- **Logic**: Fill to capacity → attempt to insert equal/worse candidates → verify all rejected
- **Validates**: O(1) fast rejection for candidates that can't improve top-K

### 8. Stress Tests (3 tests)
- **Tests**: `StressTest_{Std,Linear,FH}Queue`
- **Logic**: Insert 256*scale random candidates into capacity-64*scale queue → verify sorted output
- **Validates**: Correctness under high load

## Helper Functions

- `make_entry(id, dist)`: Creates candidate entry
- `drain_unexplored(queue)`: Extracts all candidates via `pop_best_unexplored()`

## Running Tests

```bash
# Build and run
cmake --build build --target test_candidate_queue
./build/tests/test_candidate_queue

# With custom parameters
./build/tests/test_candidate_queue --scale 10 --seed 123
```

## Key Optimizations Tested

1. **Reverse vs Sort**: Extract methods use `std::reverse` (O(n)) instead of `std::sort` (O(n log n))
2. **Fast Rejection**: Threshold checks prevent unnecessary heap operations
3. **Cursor Optimization**: LinearQueue avoids redundant scans

## Implementation Notes

- **StdCandidateQueue**: Uses `std::priority_queue` (dual-heap)
- **LinearCandidateQueue**: Sorted array with lazy deletion, optimized for N ≤ 64
- **FHCandidateQueue**: Uses custom `FourAryHeap` for better cache locality
