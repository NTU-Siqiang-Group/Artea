#!/usr/bin/env python3
"""
Verify numerical calculations for the new independent sampling approach.
With independent sampling, we directly sample m distances, so the formulas are simpler.
"""

import math
from scipy import stats

def calculate_required_distances(p: float, confidence: float, delta: float) -> int:
    """
    Calculate required distance samples using the formula:
    m = Z^2 * (1-p) / (p * delta^2)

    Args:
        p: Target quantile probability (e.g., 0.001 for P_0.1)
        confidence: Confidence level (e.g., 0.95 or 0.99)
        delta: Relative error (e.g., 0.1 for 10%)

    Returns:
        Required number of distance samples
    """
    alpha = 1 - confidence
    z_value = stats.norm.ppf(1 - alpha/2)

    m = (z_value**2 * (1 - p)) / (p * delta**2)
    return int(math.ceil(m))

def verify_calculation(name: str, p: float, confidence: float, delta: float, expected_m: int):
    """Verify a single calculation from the documentation."""
    print(f"\n{'='*80}")
    print(f"Verifying: {name}")
    print(f"{'='*80}")
    print(f"Parameters:")
    print(f"  p (quantile): {p}")
    print(f"  Confidence: {confidence*100}%")
    print(f"  Relative error δ: {delta*100}%")

    # Calculate Z-value
    alpha = 1 - confidence
    z_value = stats.norm.ppf(1 - alpha/2)
    print(f"  Z-value (α/2={alpha/2}): {z_value:.4f}")

    # Calculate required distance samples
    calculated_m = calculate_required_distances(p, confidence, delta)
    print(f"\nDistance samples (m):")
    print(f"  Expected: {expected_m:,}")
    print(f"  Calculated: {calculated_m:,}")

    # Allow small rounding differences
    match = abs(calculated_m - expected_m) / expected_m < 0.01
    print(f"  Match: {'✓' if match else '✗'}")

    if not match:
        print(f"  Difference: {abs(calculated_m - expected_m):,} ({abs(calculated_m - expected_m)/expected_m*100:.2f}%)")

    return match

def main():
    print("="*80)
    print("RADIUS PROBER VERIFICATION (Independent Sampling)")
    print("="*80)

    # Test cases from documentation
    test_cases = [
        # (name, p, confidence, delta, expected_m)
        ("P_0.1, 95% conf, 10% error", 0.001, 0.95, 0.1, 384160),
        ("P_0.1, 99% conf, 10% error", 0.001, 0.99, 0.1, 663500),
        ("P_0.1, 99% conf, 20% error", 0.001, 0.99, 0.2, 165875),
        ("P_0.05, 95% conf, 10% error", 0.0005, 0.95, 0.1, 768320),
        ("P_0.05, 99% conf, 10% error", 0.0005, 0.99, 0.1, 1327000),
        ("P_0.01, 95% conf, 10% error", 0.0001, 0.95, 0.1, 3841600),
        ("P_0.01, 99% conf, 10% error", 0.0001, 0.99, 0.1, 6635000),
    ]

    all_passed = True
    for test_case in test_cases:
        passed = verify_calculation(*test_case)
        all_passed = all_passed and passed

    # Verify probe_radius.cpp defaults
    print("\n" + "="*80)
    print("PROBE_RADIUS.CPP DEFAULT VERIFICATION")
    print("="*80)

    print("\nDefault configuration:")
    print("  quantile = 0.0005 (P_0.05)")
    print("  num_distances = 1,327,000")
    print("  Expected: 99% confidence, 10% relative error")

    verify_calculation(
        "probe_radius.cpp defaults",
        p=0.0005,
        confidence=0.99,
        delta=0.1,
        expected_m=1327000
    )

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"All tests passed: {'✓ YES' if all_passed else '✗ NO'}")
    print("\nKey advantages of independent sampling:")
    print("  1. Distance samples are truly i.i.d., satisfying statistical assumptions")
    print("  2. No need to convert between vector samples and distance samples")
    print("  3. Linear time complexity O(m·d) instead of quadratic O(s²·d)")
    print("  4. Linear space complexity O(m) instead of quadratic O(s²)")
    print("="*80)

if __name__ == "__main__":
    main()
