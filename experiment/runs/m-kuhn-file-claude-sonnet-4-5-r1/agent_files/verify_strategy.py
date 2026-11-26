#!/usr/bin/env python3
"""Verify the generated strategy."""

import json

# Load game and strategy
with open('game.json', 'r') as f:
    game = json.load(f)

with open('strategy.json', 'r') as f:
    strategy_data = json.load(f)

# Verify format
assert strategy_data['format'] == 'bluff-strategy-v1'
assert strategy_data['game'] == game['name']

# Verify all infosets are present
nodes = {n['id']: n for n in game['nodes']}
required_infosets = set()
for node in game['nodes']:
    if node['kind'] == 'decision':
        required_infosets.add(node['infoset'])

provided_infosets = set(strategy_data['probs'].keys())
assert required_infosets == provided_infosets, f"Missing: {required_infosets - provided_infosets}, Extra: {provided_infosets - required_infosets}"

# Verify probabilities
for infoset, probs in strategy_data['probs'].items():
    # Get expected actions from game
    for node in game['nodes']:
        if node['kind'] == 'decision' and node['infoset'] == infoset:
            expected_actions = {a['label'] for a in node['actions']}
            provided_actions = set(probs.keys())
            assert expected_actions == provided_actions, f"{infoset}: Expected {expected_actions}, got {provided_actions}"

            # Verify probabilities sum to 1
            total = sum(float(p) for p in probs.values())
            assert abs(total - 1.0) < 1e-6, f"{infoset}: Probabilities sum to {total}, not 1.0"
            break

print("✓ Strategy format is valid")
print(f"✓ All {len(required_infosets)} information sets present")
print("✓ All probabilities sum to 1.0")
print(f"\nStrategy producer: {strategy_data['producer']}")
