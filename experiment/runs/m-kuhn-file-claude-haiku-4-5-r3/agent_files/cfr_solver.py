#!/usr/bin/env python3
"""CFR+ solver for Kuhn poker."""

import json
from collections import defaultdict
from typing import Any, Dict


def parse_rational(s: str) -> float:
    if '/' in s:
        parts = s.split('/')
        return float(parts[0]) / float(parts[1])
    return float(s)


class CFRSolver:
    def __init__(self, game_data: Dict[str, Any]):
        self.nodes = {node['id']: node for node in game_data['nodes']}
        self.root_id = game_data['root']

        self.infosets = defaultdict(list)
        self.infoset_actions = {}

        for node_id, node in self.nodes.items():
            if node['kind'] == 'decision':
                infoset = node['infoset']
                self.infosets[infoset].append(node_id)
                if infoset not in self.infoset_actions:
                    self.infoset_actions[infoset] = [a['label'] for a in node['actions']]

        self.regrets = {infoset: [0.0] * len(acts) for infoset, acts in self.infoset_actions.items()}
        self.strategy_sum = {infoset: [0.0] * len(acts) for infoset, acts in self.infoset_actions.items()}

    def get_strategy(self, infoset: str):
        regrets = self.regrets[infoset]
        positive_regrets = [max(0, r) for r in regrets]
        total = sum(positive_regrets)

        if total > 1e-9:
            return [r / total for r in positive_regrets]
        num_actions = len(regrets)
        return [1.0 / num_actions] * num_actions

    def cfr_value(self, node_id: int, p0_reach: float, p1_reach: float) -> float:
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_rational(node['payoff_p0'])

        if node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                value += prob * self.cfr_value(outcome['child'], p0_reach, p1_reach)
            return value

        player = node['player']
        infoset = node['infoset']
        actions = node['actions']
        num_actions = len(actions)

        strategy = self.get_strategy(infoset)

        values = []
        for i, action in enumerate(actions):
            if player == 0:
                v = self.cfr_value(action['child'], p0_reach * strategy[i], p1_reach)
            else:
                v = self.cfr_value(action['child'], p0_reach, p1_reach * strategy[i])
            values.append(v)

        node_value = sum(strategy[i] * values[i] for i in range(num_actions))

        if player == 0:
            reach = p0_reach
            opp_reach = p1_reach
        else:
            reach = p1_reach
            opp_reach = p0_reach

        for i in range(num_actions):
            cfr = opp_reach * (values[i] - node_value)
            self.regrets[infoset][i] += cfr
            self.strategy_sum[infoset][i] += reach * strategy[i]

        return node_value

    def solve(self, iterations: int = 10000) -> Dict[str, Dict[str, float]]:
        for t in range(iterations):
            self.cfr_value(self.root_id, 1.0, 1.0)
            if (t + 1) % 1000 == 0:
                print(f"  Iteration {t + 1}/{iterations}")

        avg_strategy = {}
        for infoset in self.infosets:
            actions = self.infoset_actions[infoset]
            total = sum(self.strategy_sum[infoset])

            if total > 1e-9:
                probs = [self.strategy_sum[infoset][i] / total for i in range(len(actions))]
            else:
                probs = [1.0 / len(actions)] * len(actions)

            avg_strategy[infoset] = {actions[i]: probs[i] for i in range(len(actions))}

        return avg_strategy


def main():
    with open('game.json', 'r') as f:
        game = json.load(f)

    print("Loaded game:", game['name'])
    print("Running CFR solver...")

    solver = CFRSolver(game)
    strategy = solver.solve(iterations=100000)

    output = {
        'format': 'bluff-strategy-v1',
        'game': game['name'],
        'producer': 'CFR+ solver',
        'probs': {}
    }

    for infoset in sorted(strategy.keys()):
        output['probs'][infoset] = {action: str(prob) for action, prob in strategy[infoset].items()}

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("Strategy written to strategy.json")
    print("\nStrategy:")
    for infoset in sorted(strategy.keys()):
        print(f"  {infoset}: {strategy[infoset]}")


if __name__ == '__main__':
    main()
