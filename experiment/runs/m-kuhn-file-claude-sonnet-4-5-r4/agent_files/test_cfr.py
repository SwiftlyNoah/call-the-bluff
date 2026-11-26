#!/usr/bin/env python3
"""Test CFR with detailed logging."""

import json
from collections import defaultdict

def parse_rational(s: str) -> float:
    if '/' in s:
        num, denom = s.split('/')
        return float(num) / float(denom)
    return float(s)

# Load game
with open('game.json', 'r') as f:
    game_data = json.load(f)

nodes = {node['id']: node for node in game_data['nodes']}
root_id = game_data['root']

# CFR data
regret_sum = defaultdict(lambda: defaultdict(float))
strategy_sum = defaultdict(lambda: defaultdict(float))

# Infosets
infosets = {}
for node in nodes.values():
    if node['kind'] == 'decision':
        infoset = node['infoset']
        if infoset not in infosets:
            actions = [a['label'] for a in node['actions']]
            infosets[infoset] = (node['player'], actions)

def get_strategy(infoset: str):
    _, actions = infosets[infoset]
    pos_regrets = {a: max(0, regret_sum[infoset][a]) for a in actions}
    total = sum(pos_regrets.values())

    if total > 0:
        return {a: pos_regrets[a] / total for a in actions}
    else:
        return {a: 1.0 / len(actions) for a in actions}

def cfr(node_id: int, traversing_player: int, p0: float, p1: float, indent=0) -> float:
    node = nodes[node_id]
    kind = node['kind']

    prefix = "  " * indent

    if kind == 'terminal':
        val = parse_rational(node['payoff_p0'])
        return val

    if kind == 'chance':
        util = 0.0
        for outcome in node['outcomes']:
            prob = parse_rational(outcome['prob'])
            util += prob * cfr(outcome['child'], traversing_player, p0, p1, indent+1)
        return util

    # Decision node
    player = node['player']
    infoset = node['infoset']
    actions = [a['label'] for a in node['actions']]

    strategy = get_strategy(infoset)

    # Compute utility for each action
    utils = {}
    for action_dict in node['actions']:
        action = action_dict['label']
        prob = strategy[action]

        if player == 0:
            utils[action] = cfr(action_dict['child'], traversing_player, p0 * prob, p1, indent+1)
        else:
            utils[action] = cfr(action_dict['child'], traversing_player, p0, p1 * prob, indent+1)

    # Node utility
    node_util = sum(strategy[a] * utils[a] for a in actions)

    # Only update regrets for the traversing player
    if player == traversing_player:
        cf_reach = p1 if player == 0 else p0

        for action in actions:
            regret = utils[action] - node_util
            regret_sum[infoset][action] += cf_reach * regret

        own_reach = p0 if player == 0 else p1
        for action in actions:
            strategy_sum[infoset][action] += own_reach * strategy[action]

    return node_util

# Run a few iterations and log
print("Running 3 iterations with logging:\n")
for iteration in range(3):
    print(f"=== Iteration {iteration + 1} ===")
    for player in [0, 1]:
        print(f"Traversing for player {player}")
        val = cfr(root_id, player, 1.0, 1.0)
        print(f"  Returned value (for P0): {val:.4f}")

    print("\nRegrets after iteration:")
    for infoset in sorted(infosets.keys()):
        if regret_sum[infoset]:
            print(f"  {infoset}:")
            for action, regret in regret_sum[infoset].items():
                print(f"    {action}: {regret:.6f}")
    print()

# Now let's manually trace one path
print("\n=== Manual trace of JQ deal (P0=J, P1=Q) ===")
print("Uniform strategy (first iteration):")
print("Node 0 (chance) -> Node 1 (P0 with J)")
print("  P0 can check or bet (uniform: 0.5 each)")
print("  If check -> Node 2 (P1 with Q after check)")
print("    P1 can check or bet (uniform: 0.5 each)")
print("    If check -> Node 3 (terminal, payoff=-1)")
print("    If bet -> Node 4 (P0 must fold/call)")
print("      If fold -> Node 5 (payoff=-1)")
print("      If call -> Node 6 (payoff=-2)")
print("  If bet -> Node 7 (P1 with Q after bet)")
print("    P1 can fold or call (uniform: 0.5 each)")
print("    If fold -> Node 8 (payoff=1)")
print("    If call -> Node 9 (payoff=-2)")
print("\nExpected payoffs with uniform play:")
print("  Check path: 0.5*(-1) + 0.5*(0.5*(-1) + 0.5*(-2)) = -0.5 + 0.5*(-1.5) = -1.25")
print("  Bet path: 0.5*(1) + 0.5*(-2) = 0.5 - 1.0 = -0.5")
print("  Overall for JQ: 0.5*(-1.25) + 0.5*(-0.5) = -0.625 - 0.25 = -0.875")
print("\nSo with J against Q, betting (-0.5) is better than checking (-1.25)!")
print("This explains why P0 learns to bet with J!")
print("\nBUT: This assumes P1 plays uniformly. In equilibrium, P1 with Q should:")
print("  - After P0 bets: mostly fold (since Q is not a strong hand)")
print("  - After P0 checks: sometimes bluff")
