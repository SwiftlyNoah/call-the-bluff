#!/usr/bin/env python3
"""Debug value calculations."""

import json
from cfr_plus_solver import CFRPlusSolver

solver = CFRPlusSolver('game.json')

# Run just a few iterations
solver.train(1000, report_every=1000)

avg_strategy = solver.get_average_strategy()

# Manual check: compute game value by hand
print("Computing game value manually...")
game_value = solver._strategy_value(avg_strategy, solver.root)
print(f"Game value (P0 perspective): {game_value:.6f}")

# Check best responses
print("\nComputing best responses...")
br0 = solver._best_response_value(0, avg_strategy, solver.root)
br1 = solver._best_response_value(1, avg_strategy, solver.root)
print(f"BR0 (P0 best response value): {br0:.6f}")
print(f"BR1 (P1 best response value, from P1 perspective): {br1:.6f}")

print(f"\nP0's gain from best-responding: {br0 - game_value:.6f}")
print(f"P1's gain from best-responding: {br1 - (-game_value):.6f} = {br1 + game_value:.6f}")

exploitability = (br0 - game_value + br1 + game_value) / 2
print(f"\nExploitability: {exploitability:.6f}")
print(f"Alternative formula: (br0 + br1) / 2 = {(br0 + br1) / 2:.6f}")

# Check some terminal payoffs
print("\nSample terminal payoffs:")
terminals = [n for n in solver.nodes.values() if n.kind == 'terminal']
for i, term in enumerate(terminals[:10]):
    print(f"  Node {term.id}: payoff_p0 = {term.payoff_p0:.2f}")

# Check what the minimum and maximum terminal payoffs are
payoffs = [n.payoff_p0 for n in terminals]
print(f"\nTerminal payoff range: [{min(payoffs):.2f}, {max(payoffs):.2f}]")
print(f"Average terminal payoff: {sum(payoffs) / len(payoffs):.2f}")
