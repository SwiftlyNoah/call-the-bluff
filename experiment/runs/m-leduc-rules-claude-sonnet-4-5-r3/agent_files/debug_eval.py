#!/usr/bin/env python3
"""Debug evaluation."""

import json
from evaluate import LeducEvaluator

with open('strategy.json', 'r') as f:
    data = json.load(f)

strategy = data['probs']
evaluator = LeducEvaluator(strategy)

# Test a few specific scenarios
test_cases = [
    (['Js', 'Qh'], "Both fold scenario"),
    (['Ks', 'Qh'], "K vs Q"),
    (['Jh', 'Js'], "Pair of Jacks possible"),
]

print("Testing specific card combinations:")
for cards, desc in test_cases:
    value = evaluator.ev_strategy(cards, '')
    print(f"{desc} {cards}: {value:.4f}")

# Check a simple manual case
print("\n Manual check - both players always check:")
cards = ['Js', 'Qh']
# Expected: check-check in round 1, then board is dealt
# Then check-check in round 2, then showdown
# Pot should be [1, 1]
# Q beats J, so P0 loses 1
# But we average over all boards...
remaining = [c for c in evaluator.CARDS if c not in cards]
print(f"Remaining cards for board: {remaining}")

total = 0
for board in remaining:
    full_cards = cards + [board]
    payoff = evaluator.get_payoff(full_cards, 'cc/cc')
    print(f"  Board {board}: payoff = {payoff}")
    total += payoff

avg = total / len(remaining)
print(f"Average payoff for cc/cc: {avg}")
