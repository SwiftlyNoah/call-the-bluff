#!/usr/bin/env python3
"""Test that we're reading the game correctly."""

import json

def parse_rational(s: str) -> float:
    if '/' in s:
        num, den = s.split('/')
        return float(num) / float(den)
    return float(s)

with open('game.json', 'r') as f:
    game = json.load(f)

nodes = {n['id']: n for n in game['nodes']}

# Test: What happens if both players always check?
def evaluate_always_check(node_id):
    node = nodes[node_id]

    if node['kind'] == 'terminal':
        return parse_rational(node['payoff_p0'])

    if node['kind'] == 'chance':
        total = 0.0
        for outcome in node['outcomes']:
            prob = parse_rational(outcome['prob'])
            total += prob * evaluate_always_check(outcome['child'])
        return total

    # Decision: always check (first action)
    return evaluate_always_check(node['actions'][0]['child'])

# Test: What happens if player 0 always bets, player 1 always calls?
def evaluate_p0_bet_p1_call(node_id):
    node = nodes[node_id]

    if node['kind'] == 'terminal':
        return parse_rational(node['payoff_p0'])

    if node['kind'] == 'chance':
        total = 0.0
        for outcome in node['outcomes']:
            prob = parse_rational(outcome['prob'])
            total += prob * evaluate_p0_bet_p1_call(outcome['child'])
        return total

    # Player 0: always bet (try second action if available, else first)
    if node['player'] == 0:
        if len(node['actions']) > 1 and node['actions'][1]['label'] == 'bet':
            return evaluate_p0_bet_p1_call(node['actions'][1]['child'])
        elif node['actions'][0]['label'] == 'bet':
            return evaluate_p0_bet_p1_call(node['actions'][0]['child'])
        else:
            return evaluate_p0_bet_p1_call(node['actions'][0]['child'])
    else:
        # Player 1: always call (try second action if available, else first)
        if len(node['actions']) > 1 and node['actions'][1]['label'] == 'call':
            return evaluate_p0_bet_p1_call(node['actions'][1]['child'])
        elif node['actions'][0]['label'] == 'call':
            return evaluate_p0_bet_p1_call(node['actions'][0]['child'])
        else:
            return evaluate_p0_bet_p1_call(node['actions'][0]['child'])

print("Test 1: Both players always check")
print(f"  Expected value for P0: {evaluate_always_check(game['root'])}")
print()

print("Test 2: P0 always bets, P1 always calls")
print(f"  Expected value for P0: {evaluate_p0_bet_p1_call(game['root'])}")
print()

# Check information set structure
infosets = {}
for node in game['nodes']:
    if node['kind'] == 'decision':
        infoset = node['infoset']
        player = node['player']
        actions = tuple(a['label'] for a in node['actions'])

        if infoset not in infosets:
            infosets[infoset] = {'player': player, 'actions': actions, 'nodes': []}
        infosets[infoset]['nodes'].append(node['id'])

print(f"Information sets found: {len(infosets)}")
for name, data in sorted(infosets.items()):
    print(f"  {name}: player {data['player']}, actions {data['actions']}, nodes {data['nodes']}")
