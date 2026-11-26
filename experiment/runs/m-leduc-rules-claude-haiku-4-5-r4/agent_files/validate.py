#!/usr/bin/env python3
"""Validate and analyze the learned strategy"""
import json
import sys

def validate_strategy(filename: str):
    """Validate strategy.json format and content"""
    with open(filename) as f:
        data = json.load(f)

    # Check required fields
    assert data.get("format") == "bluff-strategy-v1", "Invalid format"
    assert data.get("game") == "leduc", "Invalid game name"
    assert "probs" in data, "Missing probs"

    probs = data["probs"]

    # Validate each information set
    total_isets = 0
    for iset_name, actions in probs.items():
        total_isets += 1

        # Sum probabilities
        prob_sum = sum(float(p) for p in actions.values())
        assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities don't sum to 1 for {iset_name}: {prob_sum}"

        # All probabilities positive
        for action, prob in actions.items():
            p = float(prob)
            assert p >= 0 and p <= 1, f"Invalid probability for {iset_name}/{action}: {p}"

    print(f"✓ Strategy is valid")
    print(f"✓ Total information sets: {total_isets}")
    print(f"\nFirst 10 information sets:")
    for i, (iset_name, actions) in enumerate(list(probs.items())[:10]):
        print(f"  {iset_name}: {actions}")

    return data

if __name__ == "__main__":
    try:
        validate_strategy("strategy.json")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        sys.exit(1)
