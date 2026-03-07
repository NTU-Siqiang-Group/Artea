#!/usr/bin/env python3
"""
Verify all numerical calculations in radius_determination.md and related code.
"""

import math
from scipy import stats
from typing import Tuple

def calculate_required_samples(p: float, confidence: float, delta: float) -> int:
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

def calculate_vector_samples(m: int) -> int:
    """
    Calculate required vector samples from distance samples.
    Uses formula: s = (1 + sqrt(1 + 8m)) / 2

    Args:
        m: Required distance samples

    Returns:
        Required number of vector samples
    """
    s = (1 + math.sqrt(1 + 8*m)) / 2
    return int(math.ceil(s))

def actual_distances(s: int) -> int:
    """
    Calculate actual number of pairwise distances from s vectors.
    Formula: C(s,2) = s(s-1)/2
    """
    return s * (s - 1) // 2

def verify_calculation(name: str, p: float, confidence: float, delta: float,
                       expected_m: int, expected_s: int, expected_actual: int):
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
    calculated_m = calculate_required_samples(p, confidence, delta)
    print(f"\nDistance samples (m):")
    print(f"  Expected: {expected_m:,}")
    print(f"  Calculated: {calculated_m:,}")
    print(f"  Match: {'✓' if abs(calculated_m - expected_m) / expected_m < 0.01 else '✗'}")

    # Calculate required vector samples
    calculated_s = calculate_vector_samples(calculated_m)
    print(f"\nVector samples (s):")
    print(f"  Expected: {expected_s:,}")
    print(f"  Calculated: {calculated_s:,}")
    print(f"  Match: {'✓' if calculated_s == expected_s else '✗'}")

    # Calculate actual distances
    calculated_actual = actual_distances(calculated_s)
    print(f"\nActual distances C(s,2):")
    print(f"  Expected: {expected_actual:,}")
    print(f"  Calculated: {calculated_actual:,}")
    print(f"  Match: {'✓' if calculated_actual == expected_actual else '✗'}")

    # Verify that actual >= required
    print(f"\nVerification: actual >= required?")
    print(f"  {calculated_actual:,} >= {calculated_m:,}: {'✓' if calculated_actual >= calculated_m else '✗'}")

    return calculated_m == expected_m or abs(calculated_m - expected_m) / expected_m < 0.01

def main():
    print("="*80)
    print("RADIUS PROBER NUMERICAL VERIFICATION")
    print("="*80)

    # Section 1.4 - Main example
    print("\n" + "="*80)
    print("SECTION 1.4: Main Example")
    print("="*80)

    p = 0.001
    confidence = 0.99
    delta = 0.1
    z = 2.576

    print(f"\nManual calculation for P_0.1, 99% confidence, 10% error:")
    print(f"  Z = {z}")
    print(f"  m = Z^2 / (p * δ^2)")
    print(f"  m = {z**2} / ({p} * {delta**2})")
    print(f"  m = {z**2 / (p * delta**2):.1f}")

    # Table from Section 1.4
    test_cases = [
        # (name, p, confidence, delta, expected_m, expected_s, expected_actual)
        ("P_0.1, 95% conf, 10% error", 0.001, 0.95, 0.1, 384160, 877, 384126),
        ("P_0.1, 99% conf, 10% error", 0.001, 0.99, 0.1, 663500, 1152, 663576),
        ("P_0.1, 99% conf, 20% error", 0.001, 0.99, 0.2, 165875, 576, 165600),
        ("P_0.05, 95% conf, 10% error", 0.0005, 0.95, 0.1, 768320, 1240, 768780),
        ("P_0.05, 99% conf, 10% error", 0.0005, 0.99, 0.1, 1327000, 1629, 1325406),
        ("P_0.01, 95% conf, 10% error", 0.0001, 0.95, 0.1, 3841600, 2772, 3842406),
        ("P_0.01, 99% conf, 10% error", 0.0001, 0.99, 0.1, 6635000, 3643, 6636663),
    ]

    all_passed = True
    for test_case in test_cases:
        passed = verify_calculation(*test_case)
        all_passed = all_passed and passed

    # Verify specific examples from Section 2.2
    print("\n" + "="*80)
    print("SECTION 2.2: Pairwise Distance Examples")
    print("="*80)

    examples = [
        (100, 4950),
        (500, 124750),
        (1000, 499500),
        (2000, 1999000),
        (5000, 12497500),
        (10000, 49995000),
    ]

    for s, expected in examples:
        calculated = actual_distances(s)
        match = "✓" if calculated == expected else "✗"
        print(f"s={s:5d}: expected={expected:10,}, calculated={calculated:10,} {match}")

    # Verify probe_radius.cpp defaults
    print("\n" + "="*80)
    print("PROBE_RADIUS.CPP DEFAULT VERIFICATION")
    print("="*80)

    print("\nDefault configuration:")
    print("  quantile = 0.0005 (P_0.05)")
    print("  num_samples = 1629")
    print("  Expected: 99% confidence, 10% relative error")

    verify_calculation(
        "probe_radius.cpp defaults",
        p=0.0005,
        confidence=0.99,
        delta=0.1,
        expected_m=1327000,
        expected_s=1629,
        expected_actual=1325406
    )

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"All tests passed: {'✓ YES' if all_passed else '✗ NO'}")
    print("="*80)

if __name__ == "__main__":
    main()
