#!/usr/bin/env python3
"""Clean CFR implementation for Kuhn poker."""

import json
from collections import defaultdict


def parse_num(s):
    if '/' in s:
        a, b = s.split('/')
        return float(a) / float(b)
    return float(s)


class GameCFR:
    def __init__(self, game_json):
        self.nodes_by_id = {n['id']: n for n in game_json['nodes']}
        self.root = game_json['root']

        # Collect all info sets
        self.infosets = {}
        for node in game_json['nodes']:
            if node['kind'] == 'decision':
                iset = node['infoset']
                if iset not in self.infosets:
                    self.infosets[iset] = {
                        'player': node['player'],
                        'actions': [a['label'] for a in node['actions']],
                        'nodes': []
                    }
                self.infosets[iset]['nodes'].append(node['id'])

        # Initialize cumulative regrets
        self.cum_regret = {}
        for iset in self.infosets:
            num_actions = len(self.infosets[iset]['actions'])
            self.cum_regret[iset] = [0.0] * num_actions

        # Strategy sum for averaging
        self.strat_sum = {}
        for iset in self.infosets:
            num_actions = len(self.infosets[iset]['actions'])
            self.strat_sum[iset] = [0.0] * num_actions

    def get_strategy(self, iset):
        """Get regret-matched strategy."""
        regrets = self.cum_regret[iset]
        pos_regrets = [max(0, r) for r in regrets]
        total = sum(pos_regrets)

        if total > 0:
            return [r / total for r in pos_regrets]
        return [1.0 / len(regrets)] * len(regrets)

    def node_cfr(self, node_id, reach_p0, reach_p1):
        """
        CFR recursive node evaluation.
        Returns value for player 0 from this node.
        """
        node = self.nodes_by_id[node_id]

        if node['kind'] == 'terminal':
            return parse_num(node['payoff_p0'])

        if node['kind'] == 'chance':
            val = 0.0
            for outcome in node['outcomes']:
                prob = parse_num(outcome['prob'])
                val += prob * self.node_cfr(outcome['child'], reach_p0, reach_p1)
            return val

        # Decision node
        player = node['player']
        iset = node['infoset']
        actions = self.infosets[iset]['actions']

        # Get current strategy for this infoset
        strat = self.get_strategy(iset)

        # Evaluate each action
        values = []
        for i, action_label in enumerate(actions):
            # Find the child node for this action
            action_node = [a for a in node['actions'] if a['label'] == action_label][0]
            child_id = action_node['child']

            # Recurse with updated reach probabilities
            if player == 0:
                child_val = self.node_cfr(child_id, reach_p0 * strat[i], reach_p1)
            else:
                child_val = self.node_cfr(child_id, reach_p0, reach_p1 * strat[i])

            values.append(child_val)

        # Node value under current strategy
        node_val = sum(strat[i] * values[i] for i in range(len(actions)))

        # Compute regrets
        if player == 0:
            reach_opp = reach_p1
            reach_self = reach_p0
        else:
            reach_opp = reach_p0
            reach_self = reach_p1

        # Update cumulative regrets and strategy sum
        for i in range(len(actions)):
            regret = reach_opp * (values[i] - node_val)
            self.cum_regret[iset][i] += regret
            self.strat_sum[iset][i] += reach_self * strat[i]

        return node_val

    def run_cfr(self, iterations):
        for _ in range(iterations):
            self.node_cfr(self.root, 1.0, 1.0)

    def get_avg_strategy(self):
        result = {}
        for iset, info in self.infosets.items():
            actions = info['actions']
            total = sum(self.strat_sum[iset])
            if total > 0:
                probs = [self.strat_sum[iset][i] / total for i in range(len(actions))]
            else:
                probs = [1.0 / len(actions)] * len(actions)
            result[iset] = dict(zip(actions, probs))
        return result


with open('game.json') as f:
    game = json.load(f)

print(f"Running CFR on {game['name']}...")
cfr = GameCFR(game)
cfr.run_cfr(100000)

strategy = cfr.get_avg_strategy()

output = {
    'format': 'bluff-strategy-v1',
    'game': game['name'],
    'producer': 'CFR (vanilla)',
    'probs': strategy
}

with open('strategy.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Done! Strategy:")
for iset in sorted(strategy.keys()):
    print(f"  {iset}: {strategy[iset]}")
