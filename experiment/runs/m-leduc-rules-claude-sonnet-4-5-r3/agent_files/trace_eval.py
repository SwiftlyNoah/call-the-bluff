#!/usr/bin/env python3
"""Trace through evaluation."""

import json
from evaluate import LeducEvaluator

with open('strategy.json', 'r') as f:
    data = json.load(f)

strategy = data['probs']
evaluator = LeducEvaluator(strategy)

# Trace through 'c' action
cards = ['Js', 'Qh']
history = ''

print(f"Start: cards={cards}, history='{history}'")
print(f"  is_terminal: {evaluator.is_terminal(history)}")
print(f"  should_deal_board: {evaluator.should_deal_board(history)}")
print(f"  next_to_act: {evaluator.next_to_act(history)}")

infoset = evaluator.get_infoset(cards[0], None, history)
print(f"  infoset: {infoset}")
if infoset in strategy:
    print(f"  strategy: {strategy[infoset]}")

# After first check
history = 'c'
print(f"\nAfter P0 checks: history='{history}'")
print(f"  is_terminal: {evaluator.is_terminal(history)}")
print(f"  should_deal_board: {evaluator.should_deal_board(history)}")
print(f"  next_to_act: {evaluator.next_to_act(history)}")

infoset = evaluator.get_infoset(cards[1], None, history)
print(f"  infoset: {infoset}")
if infoset in strategy:
    print(f"  strategy: {strategy[infoset]}")

# After both check
history = 'cc'
print(f"\nAfter both check: history='{history}'")
print(f"  is_terminal: {evaluator.is_terminal(history)}")
print(f"  should_deal_board: {evaluator.should_deal_board(history)}")

# Should now deal board and add '/'
history = 'cc/'
print(f"\nRound 2 starts: history='{history}'")
print(f"  is_terminal: {evaluator.is_terminal(history)}")
print(f"  should_deal_board: {evaluator.should_deal_board(history)}")
print(f"  next_to_act: {evaluator.next_to_act(history)}")

# After P0 checks in round 2
history = 'cc/c'
print(f"\nP0 checks in round 2: history='{history}'")
print(f"  is_terminal: {evaluator.is_terminal(history)}")
print(f"  next_to_act: {evaluator.next_to_act(history)}")

# After both check in round 2
history = 'cc/cc'
print(f"\nBoth check in round 2: history='{history}'")
print(f"  is_terminal: {evaluator.is_terminal(history)}")
