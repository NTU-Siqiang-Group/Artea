#!/usr/bin/env python3
"""
Detailed analysis of discrepancies in radius_determination.md
"""

import math
from scipy import stats

def analyze_case(name: str, p: float, confidence: float, delta: float,
                 doc_m: int, doc_s: int, doc_actual: int):
    """Analyze a single case and show all intermediate values."""

    print(f"\n{'='*80}")
    print(f"{name}")
    print(f"{'='*80}")

    # Step 1: Calculate Z-value
    alpha = 1 - confidence
    z_value = stats.norm.ppf(1 - alpha/2)

    print(f"\nStep 1: Z-value calculation")
    print(f"  Confidence: {confidence*100}%")
    print(f"  α = 1 - confidence = {alpha}")
    print(f"  α/2 = {alpha/2}")
    print(f"  Z_(α/2) = {z_value:.6f}")

    # Step 2: Calculate required distance samples
    print(f"\nStep 2: Required distance samples (m)")
    print(f"  Formula: m = Z^2 * (1-p) / (p * δ^2)")
    print(f"  p = {p}")
    print(f"  δ = {delta}")
    print(f"  1-p = {1-p}")

    m_exact = (z_value**2 * (1-p)) / (p * delta**2)
    m_rounded = round(m_exact)

    print(f"  m (exact) = {m_exact:.2f}")
    print(f"  m (rounded) = {m_rounded:,}")
    print(f"  Documentation says: {doc_m:,}")
    print(f"  Difference: {abs(m_rounded - doc_m):,} ({abs(m_rounded - doc_m)/doc_m*100:.2f}%)")

    # Step 3: Calculate required vector samples
    print(f"\nStep 3: Required vector samples (s)")
    print(f"  Formula: s = (1 + sqrt(1 + 8m)) / 2")

    # Try with doc_m
    s_from_doc_m = (1 + math.sqrt(1 + 8*doc_m)) / 2
    s_from_doc_m_ceil = math.ceil(s_from_doc_m)

    print(f"  Using doc m={doc_m:,}:")
    print(f"    s (exact) = {s_from_doc_m:.6f}")
    print(f"    s (ceil) = {s_from_doc_m_ceil}")
    print(f"    Documentation says: {doc_s}")
    print(f"    Match: {'✓' if s_from_doc_m_ceil == doc_s else '✗'}")

    # Try with calculated m
    s_from_calc_m = (1 + math.sqrt(1 + 8*m_rounded)) / 2
    s_from_calc_m_ceil = math.ceil(s_from_calc_m)

    print(f"  Using calculated m={m_rounded:,}:")
    print(f"    s (exact) = {s_from_calc_m:.6f}")
    print(f"    s (ceil) = {s_from_calc_m_ceil}")

    # Step 4: Calculate actual distances
    print(f"\nStep 4: Actual pairwise distances")
    print(f"  Formula: C(s,2) = s(s-1)/2")

    actual_from_doc_s = doc_s * (doc_s - 1) // 2
    print(f"  Using doc s={doc_s}:")
    print(f"    Actual distances = {actual_from_doc_s:,}")
    print(f"    Documentation says: {doc_actual:,}")
    print(f"    Match: {'✓' if actual_from_doc_s == doc_actual else '✗'}")
    print(f"    Difference: {abs(actual_from_doc_s - doc_actual):,}")

    # Step 5: Verify sufficiency
    print(f"\nStep 5: Verification")
    print(f"  Required distance samples: {m_rounded:,}")
    print(f"  Actual distance samples: {actual_from_doc_s:,}")
    print(f"  Sufficient: {'✓' if actual_from_doc_s >= m_rounded else '✗'}")
    print(f"  Margin: {actual_from_doc_s - m_rounded:,} ({(actual_from_doc_s - m_rounded)/m_rounded*100:.2f}%)")

    # Recommendation
    print(f"\n{'='*80}")
    if actual_from_doc_s == doc_actual and s_from_doc_m_ceil == doc_s:
        print("✓ All values in documentation are CORRECT")
    else:
        print("✗ ISSUES FOUND:")
        if actual_from_doc_s != doc_actual:
            print(f"  - Actual distances should be {actual_from_doc_s:,}, not {doc_actual:,}")
        if s_from_doc_m_ceil != doc_s:
            print(f"  - Vector samples should be {s_from_doc_m_ceil}, not {doc_s}")

    return {
        'z': z_value,
        'm_calc': m_rounded,
        'm_doc': doc_m,
        's_calc': s_from_doc_m_ceil,
        's_doc': doc_s,
        'actual_calc': actual_from_doc_s,
        'actual_doc': doc_actual,
    }

def main():
    print("="*80)
    print("DETAILED ANALYSIS OF RADIUS_DETERMINATION.MD")
    print("="*80)

    cases = [
        ("P_0.1, 95% conf, 10% error", 0.001, 0.95, 0.1, 384160, 877, 384126),
        ("P_0.1, 99% conf, 10% error", 0.001, 0.99, 0.1, 663500, 1152, 663576),
        ("P_0.1, 99% conf, 20% error", 0.001, 0.99, 0.2, 165875, 576, 165600),
        ("P_0.05, 95% conf, 10% error", 0.0005, 0.95, 0.1, 768320, 1240, 768780),
        ("P_0.05, 99% conf, 10% error", 0.0005, 0.99, 0.1, 1327000, 1629, 1325406),
        ("P_0.01, 95% conf, 10% error", 0.0001, 0.95, 0.1, 3841600, 2772, 3842406),
        ("P_0.01, 99% conf, 10% error", 0.0001, 0.99, 0.1, 6635000, 3643, 6636663),
    ]

    results = []
    for case in cases:
        result = analyze_case(*case)
        results.append((case[0], result))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL CORRECTIONS NEEDED")
    print("="*80)

    corrections = []
    for name, result in results:
        if result['actual_calc'] != result['actual_doc'] or result['s_calc'] != result['s_doc']:
            corrections.append({
                'name': name,
                's_old': result['s_doc'],
                's_new': result['s_calc'],
                'actual_old': result['actual_doc'],
                'actual_new': result['actual_calc'],
            })

    if corrections:
        print("\nThe following rows need correction in the documentation:\n")
        for corr in corrections:
            print(f"{corr['name']}:")
            if corr['s_old'] != corr['s_new']:
                print(f"  Vector samples: {corr['s_old']} → {corr['s_new']}")
            if corr['actual_old'] != corr['actual_new']:
                print(f"  Actual distances: {corr['actual_old']:,} → {corr['actual_new']:,}")
    else:
        print("\n✓ All values are correct!")

if __name__ == "__main__":
    main()
