#!/usr/bin/env python3
"""Debug CFR to see what's happening."""

import json
from cfr_clean import GameCFR, parse_num

with open('game.json') as f:
    game = json.load(f)

print(f"Running CFR on {game['name']}...")
cfr = GameCFR(game)

# Run for fewer iterations to see progression
for iteration in [100, 500, 1000, 5000, 10000]:
    cfr.cum_regret = {iset: [0.0] * len(cfr.infosets[iset]['actions']) for iset in cfr.infosets}
    cfr.strat_sum = {iset: [0.0] * len(cfr.infosets[iset]['actions']) for iset in cfr.infosets}

    cfr.run_cfr(iteration)
    strategy = cfr.get_avg_strategy()

    # Compute game value
    def compute_value(node_id, strat):
        node = cfr.nodes_by_id[node_id]
        if node['kind'] == 'terminal':
            return parse_num(node['payoff_p0'])
        if node['kind'] == 'chance':
            val = 0.0
            for outcome in node['outcomes']:
                prob = parse_num(outcome['prob'])
                val += prob * compute_value(outcome['child'], strat)
            return val

        player = node['player']
        iset = node['infoset']
        actions = cfr.infosets[iset]['actions']
        s = strat[iset]

        val = 0.0
        for i, action_label in enumerate(actions):
            action_node = [a for a in node['actions'] if a['label'] == action_label][0]
            child_id = action_node['child']
            child_val = compute_value(child_id, strat)
            val += s[action_label] * child_val
        return val

    value = compute_value(cfr.root, strategy)

    print(f"\nIteration {iteration}: game value = {value}")
    print(f"  J:  check={strategy['J:']['check']:.6f}, bet={strategy['J:']['bet']:.6f}")
    print(f"  K:  check={strategy['K:']['check']:.6f}, bet={strategy['K:']['bet']:.6f}")
    print(f"  Q:  check={strategy['Q:']['check']:.6f}, bet={strategy['Q:']['bet']:.6f}")
