#!/usr/bin/env python3
"""Compute final metrics for the strategy."""

import json

def parse_rational(s: str) -> float:
    if '/' in s:
        num, den = s.split('/')
        return float(num) / float(den)
    return float(s)

# Load game and strategy
with open('game.json', 'r') as f:
    game = json.load(f)

with open('strategy.json', 'r') as f:
    strategy_data = json.load(f)

nodes = {n['id']: n for n in game['nodes']}
strategy = {k: {a: float(p) for a, p in v.items()} for k, v in strategy_data['probs'].items()}

def evaluate_strategy(node_id):
    """Compute expected utility for P0 when both players follow strategy."""
    node = nodes[node_id]

    if node['kind'] == 'terminal':
        return parse_rational(node['payoff_p0'])

    if node['kind'] == 'chance':
        util = 0.0
        for outcome in node['outcomes']:
            prob = parse_rational(outcome['prob'])
            util += prob * evaluate_strategy(outcome['child'])
        return util

    # Decision node
    infoset = node['infoset']
    strat = strategy[infoset]

    util = 0.0
    for action_node in node['actions']:
        action = action_node['label']
        util += strat[action] * evaluate_strategy(action_node['child'])

    return util

def best_response(player, node_id):
    """Compute best-response utility for given player."""
    node = nodes[node_id]

    if node['kind'] == 'terminal':
        payoff_p0 = parse_rational(node['payoff_p0'])
        return payoff_p0 if player == 0 else -payoff_p0

    if node['kind'] == 'chance':
        util = 0.0
        for outcome in node['outcomes']:
            prob = parse_rational(outcome['prob'])
            util += prob * best_response(player, outcome['child'])
        return util

    # Decision node
    acting_player = node['player']

    if acting_player == player:
        # Best respond: maximize
        best_util = float('-inf')
        for action_node in node['actions']:
            util = best_response(player, action_node['child'])
            best_util = max(best_util, util)
        return best_util
    else:
        # Opponent follows their strategy
        infoset = node['infoset']
        strat = strategy[infoset]

        util = 0.0
        for action_node in node['actions']:
            action = action_node['label']
            util += strat[action] * best_response(player, action_node['child'])
        return util

# Compute metrics
game_value = evaluate_strategy(game['root'])
br0_value = best_response(0, game['root'])
br1_value = best_response(1, game['root'])

# Exploitability
p0_loss_from_p1_br = game_value - (-br1_value)
p0_gain_from_p0_br = br0_value - game_value
exploitability = (p0_gain_from_p0_br + p0_loss_from_p1_br) / 2

print(f"Game value (P0): {game_value:.10f}")
print(f"P0 best-response value: {br0_value:.10f}")
print(f"P1 best-response value: {br1_value:.10f}")
print(f"Exploitability: {exploitability:.10f}")
print(f"Converged (exp < 0.001): {exploitability < 0.001}")

# Output JSON
result = {
    "claim_converged": exploitability < 0.001,
    "claimed_exploitability": round(exploitability, 10),
    "claimed_value_p0": round(game_value, 10),
    "confidence": "medium" if exploitability < 0.01 else "low",
    "notes": f"Vanilla CFR with 1M iterations. Known Nash value is -1/18 ≈ -0.0556. Achieved exploitability {exploitability:.4f}."
}

print("\nFinal JSON:")
print(json.dumps(result))
