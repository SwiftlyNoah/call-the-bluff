#!/usr/bin/env python3
"""Analyze the learned strategy for convergence"""
import json
import statistics

def analyze_strategy(filename: str):
    """Analyze strategy convergence"""
    with open(filename) as f:
        data = json.load(f)

    probs = data["probs"]

    # Collect statistics
    entropy_values = []
    max_probs = []
    num_actions_per_iset = []

    for iset_name, actions in probs.items():
        probs_list = [float(p) for p in actions.values()]

        # Compute entropy (measure of mixed-ness)
        entropy = -sum(p * math.log(p + 1e-10) for p in probs_list if p > 0)
        entropy_values.append(entropy)

        # Max probability (measure of purity)
        max_probs.append(max(probs_list))

        # Number of actions
        num_actions_per_iset.append(len(actions))

    print(f"Strategy Analysis:")
    print(f"  Total information sets: {len(probs)}")
    print(f"  Average actions per infoset: {statistics.mean(num_actions_per_iset):.2f}")
    print(f"\nEntropy (how mixed strategies are):")
    print(f"  Average entropy: {statistics.mean(entropy_values):.4f}")
    print(f"  Min entropy: {min(entropy_values):.4f}")
    print(f"  Max entropy: {max(entropy_values):.4f}")
    print(f"\nAction probability max values (convergence measure):")
    print(f"  Average max prob: {statistics.mean(max_probs):.4f}")
    print(f"  Min max prob: {min(max_probs):.4f}")
    print(f"  Max max prob: {max(max_probs):.4f}")

    # Count pure vs mixed strategies
    pure_count = sum(1 for p in max_probs if p > 0.99)
    mixed_count = sum(1 for p in max_probs if 0.4 <= p <= 0.6)
    print(f"\nStrategy purity:")
    print(f"  Pure strategies (max prob > 0.99): {pure_count} ({100*pure_count/len(probs):.1f}%)")
    print(f"  Mixed strategies (0.4 <= max prob <= 0.6): {mixed_count} ({100*mixed_count/len(probs):.1f}%)")

    return data

if __name__ == "__main__":
    import math
    analyze_strategy("strategy.json")
