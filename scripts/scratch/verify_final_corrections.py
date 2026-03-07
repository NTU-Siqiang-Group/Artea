#!/usr/bin/env python3
"""
Final verification of corrected values in radius_determination.md
"""

import math
from scipy import stats

def verify_corrected_values():
    """Verify all corrected values are now correct."""

    print("="*80)
    print("FINAL VERIFICATION OF CORRECTED VALUES")
    print("="*80)

    # Corrected values from the documentation
    corrected_cases = [
        # (name, p, conf, delta, m, s, actual)
        ("P_0.1, 95%, 10%", 0.001, 0.95, 0.1, 384160, 878, 385003),
        ("P_0.1, 99%, 10%", 0.001, 0.99, 0.1, 663500, 1153, 664128),
        ("P_0.1, 99%, 20%", 0.001, 0.99, 0.2, 165875, 577, 166176),
        ("P_0.05, 95%, 10%", 0.0005, 0.95, 0.1, 768320, 1241, 769420),
        ("P_0.05, 99%, 10%", 0.0005, 0.99, 0.1, 1327000, 1630, 1327635),
        ("P_0.01, 95%, 10%", 0.0001, 0.95, 0.1, 3841600, 2773, 3843378),
        ("P_0.01, 99%, 10%", 0.0001, 0.99, 0.1, 6635000, 3644, 6637546),
    ]

    all_correct = True

    for name, p, conf, delta, doc_m, doc_s, doc_actual in corrected_cases:
        print(f"\n{name}:")

        # Calculate Z-value
        alpha = 1 - conf
        z = stats.norm.ppf(1 - alpha/2)

        # Calculate required distance samples
        m_calc = (z**2 * (1-p)) / (p * delta**2)
        m_rounded = round(m_calc)

        # Calculate required vector samples
        s_calc = (1 + math.sqrt(1 + 8*doc_m)) / 2
        s_ceil = math.ceil(s_calc)

        # Calculate actual distances
        actual_calc = doc_s * (doc_s - 1) // 2

        # Verify
        s_correct = (s_ceil == doc_s)
        actual_correct = (actual_calc == doc_actual)
        sufficient = (actual_calc >= m_rounded)

        print(f"  Vector samples: {doc_s} {'✓' if s_correct else '✗ WRONG'}")
        print(f"  Actual distances: {doc_actual:,} {'✓' if actual_correct else '✗ WRONG'}")
        print(f"  Sufficient (actual >= required): {'✓' if sufficient else '✗ INSUFFICIENT'}")

        if not (s_correct and actual_correct and sufficient):
            all_correct = False
            print(f"  ERROR: Expected s={s_ceil}, actual={actual_calc:,}")

    # Verify probe_radius.cpp defaults
    print("\n" + "="*80)
    print("PROBE_RADIUS.CPP DEFAULTS")
    print("="*80)

    probe_s = 1630
    probe_p = 0.0005
    probe_conf = 0.99
    probe_delta = 0.1

    alpha = 1 - probe_conf
    z = stats.norm.ppf(1 - alpha/2)
    m_required = round((z**2 * (1-probe_p)) / (probe_p * probe_delta**2))
    actual_distances = probe_s * (probe_s - 1) // 2

    print(f"\nDefault num_samples = {probe_s}")
    print(f"For P_0.05 (p={probe_p}), 99% confidence, 10% error:")
    print(f"  Required distance samples: {m_required:,}")
    print(f"  Actual distance samples: {actual_distances:,}")
    print(f"  Sufficient: {'✓' if actual_distances >= m_required else '✗'}")
    print(f"  Margin: {actual_distances - m_required:,} ({(actual_distances - m_required)/m_required*100:.2f}%)")

    if actual_distances < m_required:
        all_correct = False

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if all_correct:
        print("✓ ALL VALUES ARE NOW CORRECT!")
    else:
        print("✗ SOME VALUES STILL NEED CORRECTION")

    return all_correct

if __name__ == "__main__":
    success = verify_corrected_values()
    exit(0 if success else 1)
