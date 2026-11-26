#!/usr/bin/env python3
"""Quick verification of strategy.json"""

import json

# Load and validate strategy file
with open('strategy.json', 'r') as f:
    strategy = json.load(f)

print(f"Format: {strategy['format']}")
print(f"Game: {strategy['game']}")
print(f"Producer: {strategy['producer']}")
print(f"Number of infosets: {len(strategy['probs'])}")

# Check that probabilities sum to 1 for each infoset
max_error = 0.0
for infoset, probs in strategy['probs'].items():
    total = sum(float(p) for p in probs.values())
    error = abs(total - 1.0)
    max_error = max(max_error, error)
    if error > 0.01:
        print(f"Warning: {infoset} has probability sum {total}")

print(f"Max probability sum error: {max_error:.2e}")
print("Strategy file is valid!" if max_error < 0.01 else "Strategy has probability errors!")
