#!/usr/bin/env python3
"""Debug CFR to see what's happening."""

import json
import solve

# Load game
with open('game.json', 'r') as f:
    game_data = json.load(f)

solver = solve.CFRPlusSolver(game_data)

# Run a few iterations
for i in range(5):
    solver.cfr(solver.root_id, 1.0, 1.0)
    print(f"\nAfter iteration {i+1}:")

    # Print a few key information sets
    for infoset in ['J:', 'Q:b', 'K:b']:
        print(f"  {infoset}:")
        print(f"    Regrets: {dict(solver.regret_sum[infoset])}")
        print(f"    Strategy: {solver.get_strategy(infoset)}")
