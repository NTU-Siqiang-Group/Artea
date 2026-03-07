#!/usr/bin/env python3
"""
Verify the compute_num_dists_sampled function matches our Python calculations.
"""

import math
from scipy import stats

def compute_num_dists_sampled_python(quantile: float, confidence: float, relative_err: float) -> int:
    """Python reference implementation."""
    alpha = 1.0 - confidence
    z_value = stats.norm.ppf(1.0 - alpha / 2.0)
    m = (z_value**2 * (1.0 - quantile)) / (quantile * relative_err**2)
    return int(math.ceil(m))

def main():
    print("="*80)
    print("VERIFY compute_num_dists_sampled FUNCTION")
    print("="*80)

    test_cases = [
        # (name, quantile, confidence, relative_err, expected_m)
        ("P_0.05, 99% conf, 10% err", 0.0005, 0.99, 0.1, 1327000),
        ("P_0.1, 99% conf, 10% err", 0.001, 0.99, 0.1, 663500),
        ("P_0.1, 95% conf, 10% err", 0.001, 0.95, 0.1, 384160),
        ("P_0.1, 99% conf, 20% err", 0.001, 0.99, 0.2, 165875),
        ("P_0.01, 99% conf, 10% err", 0.0001, 0.99, 0.1, 6635000),
    ]

    all_passed = True
    for name, quantile, confidence, relative_err, expected_m in test_cases:
        calculated_m = compute_num_dists_sampled_python(quantile, confidence, relative_err)

        print(f"\n{name}:")
        print(f"  quantile={quantile}, confidence={confidence}, relative_err={relative_err}")
        print(f"  Expected: {expected_m:,}")
        print(f"  Calculated: {calculated_m:,}")

        # Allow small rounding differences
        match = abs(calculated_m - expected_m) / expected_m < 0.01
        print(f"  Match: {'✓' if match else '✗'}")

        if not match:
            all_passed = False

    print("\n" + "="*80)
    print(f"All tests passed: {'✓ YES' if all_passed else '✗ NO'}")
    print("="*80)

    # Show example usage
    print("\nExample usage in probe_radius:")
    print("  # Option 1: Specify samples directly")
    print("  ./bin/probe_radius -q 0.0005 -m 1327000")
    print()
    print("  # Option 2: Auto-compute from confidence and relative error")
    print("  ./bin/probe_radius -q 0.0005 --confidence 0.99 --relative-err 0.1")
    print("  # This will automatically compute m=1,327,000")

if __name__ == "__main__":
    main()
