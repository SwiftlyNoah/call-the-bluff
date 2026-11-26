#!/usr/bin/env python3
"""Quick test to verify solver logic is working correctly."""

import json
from cfr_plus_solver import CFRPlusSolver

# Create a simple test case
solver = CFRPlusSolver('game.json')

print("Game statistics:")
print(f"  Total nodes: {len(solver.nodes)}")

chance_nodes = sum(1 for n in solver.nodes.values() if n.kind == 'chance')
decision_nodes = sum(1 for n in solver.nodes.values() if n.kind == 'decision')
terminal_nodes = sum(1 for n in solver.nodes.values() if n.kind == 'terminal')

print(f"  Chance nodes: {chance_nodes}")
print(f"  Decision nodes: {decision_nodes}")
print(f"  Terminal nodes: {terminal_nodes}")

# Count information sets
infosets_p0 = set()
infosets_p1 = set()
for node in solver.nodes.values():
    if node.kind == 'decision':
        if node.player == 0:
            infosets_p0.add(node.infoset)
        else:
            infosets_p1.add(node.infoset)

print(f"  Player 0 information sets: {len(infosets_p0)}")
print(f"  Player 1 information sets: {len(infosets_p1)}")

# Run a few iterations to test
print("\nRunning 100 iterations as a quick test...")
solver.train(100, report_every=100)

# Check a strategy
avg_strategy = solver.get_average_strategy()
print(f"\nGenerated strategy for {len(avg_strategy)} information sets")

# Compute value
exploitability, game_value = solver.compute_exploitability()
print(f"After 100 iterations:")
print(f"  Game value (P0): {game_value:.6f}")
print(f"  Exploitability: {exploitability:.6f}")
