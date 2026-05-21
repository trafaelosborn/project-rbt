"""
Diagnose why hungarian_alignment_diagnostics(inv, inv) != 1.0 for identical
inventories. Build the Latin family inventory twice, compare, run alignment,
and inspect the cost matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequester.guard import load_sequestered, lock_sequestration, unlock_sequestration
from src.validation.hungarian_alignment import (
    FamilyAlignmentConfig,
    _family_cost,
    extract_family_inventory,
    hungarian_alignment_diagnostics,
    load_latin_family_reference,
)


def main() -> None:
    cfg = FamilyAlignmentConfig()

    # Inventory A: via production loader
    inv_a = load_latin_family_reference(cfg)

    # Inventory B: rebuild from same Latin[:50000] slice manually
    unlock_sequestration("Diagnostic for self-similarity family-alignment ceiling check.")
    try:
        corpus = load_sequestered("latin")
    finally:
        lock_sequestration()
    seqs = corpus["sequences"][: cfg.max_latin_sequences]
    inv_b = extract_family_inventory("latin", seqs, cfg)

    print(f"A families: {len(inv_a.families)}; B families: {len(inv_b.families)}")
    print(f"A total_tokens: {inv_a.total_tokens}; B total_tokens: {inv_b.total_tokens}")

    # Compare family by family.
    print()
    print("Per-family field equality (A vs B):")
    for i, (fa, fb) in enumerate(zip(inv_a.families, inv_b.families)):
        same_id = fa.family_id == fb.family_id
        same_kind = fa.kind == fb.kind
        same_mass = fa.mass == fb.mass
        same_occ = fa.total_occurrences == fb.total_occurrences
        same_member = fa.member_token_count == fb.member_token_count
        same_length = fa.mean_token_length == fb.mean_token_length
        same_samples = fa.sample_tokens == fb.sample_tokens
        same_tg = fa.char_trigram_profile == fb.char_trigram_profile
        same_sfx = fa.suffix_profile == fb.suffix_profile
        all_same = all([same_id, same_kind, same_mass, same_occ, same_member,
                        same_length, same_samples, same_tg, same_sfx])
        cost_self_a = _family_cost(fa, fa, cfg)
        cost_a_b = _family_cost(fa, fb, cfg)
        if not all_same or cost_self_a != 0.0 or cost_a_b != 0.0:
            print(f"  [{i}] id={fa.family_id} kind={fa.kind} all_same={all_same} self_cost={cost_self_a} a_b_cost={cost_a_b}")
            if not same_tg:
                print(f"       tg differ: |A|={len(fa.char_trigram_profile)} |B|={len(fb.char_trigram_profile)}")

    # Alignment of inv_a against itself.
    diag = hungarian_alignment_diagnostics(inv_a, inv_a, cfg)
    print()
    print(f"alignment(A, A): score={diag['family_alignment_score']:.6f} cost={diag['family_alignment_cost']:.6f}")
    print(f"  matched={diag['matched_family_count']} unmatched_bridge={diag['unmatched_bridge_families']} unmatched_ref={diag['unmatched_reference_families']}")
    print(f"  bridge_n={diag['bridge_family_count']} ref_n={diag['reference_family_count']}")

    # Inspect cost matrix diagonal.
    n = len(inv_a.families)
    diagonal_costs = [_family_cost(inv_a.families[i], inv_a.families[i], cfg) for i in range(n)]
    off_diag_min = []
    for i in range(n):
        row = [_family_cost(inv_a.families[i], inv_a.families[j], cfg) for j in range(n)]
        row[i] = float("inf")  # exclude diagonal
        off_diag_min.append(min(row))
    print()
    print("Cost diagonal (self-pair) vs cheapest off-diagonal for each row:")
    for i in range(n):
        fa = inv_a.families[i]
        print(f"  [{i:2d}] {fa.family_id:30s} self={diagonal_costs[i]:.4f}  min_off_diag={off_diag_min[i]:.4f}")

    # Now look at the matched_pairs the algorithm actually picked.
    print()
    print("Hungarian-picked pairs (bridge_id -> ref_id):")
    for pair in diag["best_pairs"][:50] + diag.get("worst_pairs", [])[:5]:
        print(f"  cost={pair['pair_cost']:.4f}  bridge={pair['bridge_family_id']}  ref={pair['reference_family_id']}")


if __name__ == "__main__":
    main()
