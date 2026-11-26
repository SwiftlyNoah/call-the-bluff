#!/usr/bin/env python3
import json
from collections import defaultdict

def parse_number(s):
    if '/' in s:
        return float(s.split('/')[0]) / float(s.split('/')[1])
    return float(s)

class GameSolver:
    def __init__(self, game_file):
        with open(game_file, 'r') as f:
            self.game = json.load(f)

        self.nodes_by_id = {node['id']: node for node in self.game['nodes']}
        self.root_id = self.game['root']

        # Build infoset structure
        self.infosets = {}
        self.infoset_actions = {}
        for node in self.game['nodes']:
            if node['kind'] == 'decision':
                infoset = node['infoset']
                if infoset not in self.infosets:
                    self.infosets[infoset] = []
                    actions = node['actions']
                    self.infoset_actions[infoset] = [a['label'] for a in actions]
                self.infosets[infoset].append(node['id'])

        # Initialize regret tracking
        self.regrets = {}
        for infoset in self.infosets:
            num_actions = len(self.infoset_actions[infoset])
            self.regrets[infoset] = [0.0] * num_actions

        # Track strategy sums
        self.strategy_sums = {}
        for infoset in self.infosets:
            num_actions = len(self.infoset_actions[infoset])
            self.strategy_sums[infoset] = [0.0] * num_actions

    def get_strategy(self, infoset_name):
        regrets = self.regrets[infoset_name]
        positive_regrets = [max(0.0, r) for r in regrets]
        total = sum(positive_regrets)
        if total > 0:
            return [r / total for r in positive_regrets]
        return [1.0 / len(regrets)] * len(regrets)

    def cfr(self, node_id, reach_p0, reach_p1):
        node = self.nodes_by_id[node_id]

        if node['kind'] == 'terminal':
            return parse_number(node['payoff_p0'])

        if node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_number(outcome['prob'])
                value += prob * self.cfr(outcome['child'], reach_p0, reach_p1)
            return value

        player = node['player']
        infoset = node['infoset']
        actions = node['actions']

        strategy = self.get_strategy(infoset)

        # Compute values for each action
        values = []
        for i, action in enumerate(actions):
            if player == 0:
                v = self.cfr(action['child'], reach_p0 * strategy[i], reach_p1)
            else:
                v = self.cfr(action['child'], reach_p0, reach_p1 * strategy[i])
            values.append(v)

        # Compute node value
        node_value = sum(strategy[i] * values[i] for i in range(len(actions)))

        # Accumulate strategy
        my_reach = reach_p0 if player == 0 else reach_p1
        for i in range(len(actions)):
            self.strategy_sums[infoset][i] += my_reach * strategy[i]

        # Update regrets (vanilla CFR, includes negative regrets)
        opp_reach = reach_p1 if player == 0 else reach_p0
        for i in range(len(actions)):
            regret = opp_reach * (values[i] - node_value)
            self.regrets[infoset][i] += regret

        return node_value

    def solve(self, num_iterations):
        for t in range(num_iterations):
            self.cfr(self.root_id, 1.0, 1.0)
            if (t + 1) % 10000 == 0:
                print(f"Iteration {t + 1}/{num_iterations}")

    def get_average_strategy(self):
        avg_strategy = {}
        for infoset in self.infosets:
            strategy_sum = self.strategy_sums[infoset]
            total = sum(strategy_sum)
            if total > 0:
                avg_strategy[infoset] = [s / total for s in strategy_sum]
            else:
                avg_strategy[infoset] = [1.0 / len(strategy_sum)] * len(strategy_sum)
        return avg_strategy

    def evaluate(self, avg_strategy):
        def eval_node(node_id):
            node = self.nodes_by_id[node_id]

            if node['kind'] == 'terminal':
                return parse_number(node['payoff_p0'])

            if node['kind'] == 'chance':
                value = 0.0
                for outcome in node['outcomes']:
                    prob = parse_number(outcome['prob'])
                    value += prob * eval_node(outcome['child'])
                return value

            infoset = node['infoset']
            actions = node['actions']
            strategy = avg_strategy[infoset]

            value = 0.0
            for i, action in enumerate(actions):
                value += strategy[i] * eval_node(action['child'])
            return value

        return eval_node(self.root_id)

def main():
    solver = GameSolver('game.json')
    print("Starting CFR solver...")
    solver.solve(num_iterations=100000)

    # Use average strategy
    avg_strategy = solver.get_average_strategy()
    game_value = solver.evaluate(avg_strategy)
    print(f"Game value for player 0: {game_value}")

    output = {
        'format': 'bluff-strategy-v1',
        'game': 'kuhn',
        'producer': 'CFR solver',
        'probs': {}
    }

    for infoset in sorted(solver.infosets.keys()):
        strategy = avg_strategy[infoset]
        action_labels = solver.infoset_actions[infoset]
        output['probs'][infoset] = {
            action_labels[i]: str(strategy[i])
            for i in range(len(action_labels))
        }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("Strategy written to strategy.json")

if __name__ == '__main__':
    main()
