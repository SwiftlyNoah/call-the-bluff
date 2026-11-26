#!/usr/bin/env python3
import json
import sys
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class CFRSolver:
    def __init__(self, game: Dict[str, Any]):
        self.game = game
        self.nodes = {node['id']: node for node in game['nodes']}
        self.root_id = game['root']

        # Build infoset index
        self.infosets = {}
        for node in self.nodes.values():
            if node.get('kind') == 'decision':
                infoset_name = node.get('infoset')
                if infoset_name not in self.infosets:
                    self.infosets[infoset_name] = node

        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def parse_number(self, s: str) -> float:
        try:
            if '/' in s:
                parts = s.split('/')
                return float(parts[0]) / float(parts[1])
            else:
                return float(s)
        except:
            return float(s)

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get regret-matching strategy for an information set."""
        node = self.infosets.get(infoset)
        if node is None:
            return {}

        actions = [a['label'] for a in node['actions']]
        regrets_for_infoset = self.regrets[infoset]

        positive_regrets = [max(0, regrets_for_infoset[a]) for a in actions]
        total = sum(positive_regrets)

        if total <= 0:
            prob = 1.0 / len(actions)
            return {a: prob for a in actions}

        return {a: pr / total for a, pr in zip(actions, positive_regrets)}

    def get_average_strategy(self, infoset: str) -> Dict[str, float]:
        """Get average strategy across all iterations."""
        node = self.infosets.get(infoset)
        if node is None:
            return {}

        actions = [a['label'] for a in node['actions']]
        strategy_sum_for_infoset = self.strategy_sum[infoset]
        total = sum(strategy_sum_for_infoset[a] for a in actions)

        if total <= 0:
            prob = 1.0 / len(actions)
            return {a: prob for a in actions}

        return {a: strategy_sum_for_infoset[a] / total for a in actions}

    def run_cfr_iteration(self):
        """Run one iteration of CFR."""
        self._cfr_recursive(self.root_id, 1.0, 1.0)
        self.iterations += 1

    def _cfr_recursive(self, node_id: int, reach_p0: float, reach_p1: float) -> Tuple[float, float]:
        """Recursive CFR traversal."""
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            payoff = self.parse_number(node['payoff_p0'])
            return (payoff, -payoff)

        if node['kind'] == 'chance':
            value_p0 = 0.0
            value_p1 = 0.0
            for outcome in node['outcomes']:
                prob = self.parse_number(outcome['prob'])
                child_id = outcome['child']
                cv0, cv1 = self._cfr_recursive(child_id, reach_p0, reach_p1)
                value_p0 += prob * cv0
                value_p1 += prob * cv1
            return (value_p0, value_p1)

        # Decision node
        acting_player = node['player']
        infoset = node['infoset']
        strategy = self.get_strategy(infoset)
        actions = [a['label'] for a in node['actions']]

        action_values = {}
        node_value_p0 = 0.0
        node_value_p1 = 0.0

        for action in node['actions']:
            action_label = action['label']
            prob = strategy.get(action_label, 0.0)
            child_id = action['child']

            if acting_player == 0:
                av0, av1 = self._cfr_recursive(child_id, reach_p0 * prob, reach_p1)
            else:
                av0, av1 = self._cfr_recursive(child_id, reach_p0, reach_p1 * prob)

            action_values[action_label] = (av0, av1)
            node_value_p0 += prob * av0
            node_value_p1 += prob * av1

        # Update regrets and strategy sum
        if acting_player == 0:
            for action_label in actions:
                av0, av1 = action_values[action_label]
                regret = reach_p1 * (av0 - node_value_p0)
                self.regrets[infoset][action_label] += regret
                self.strategy_sum[infoset][action_label] += strategy.get(action_label, 0.0) * reach_p0
        else:
            for action_label in actions:
                av0, av1 = action_values[action_label]
                regret = reach_p0 * (av1 - node_value_p1)
                self.regrets[infoset][action_label] += regret
                self.strategy_sum[infoset][action_label] += strategy.get(action_label, 0.0) * reach_p1

        return (node_value_p0, node_value_p1)

    def run(self, iterations: int = 1000):
        """Run CFR for the specified number of iterations."""
        for i in range(iterations):
            if (i + 1) % 500 == 0:
                print(f"Iteration {i + 1}/{iterations}", flush=True)
            self.run_cfr_iteration()

    def write_strategy(self, filename: str):
        """Write the average strategy to a JSON file."""
        probs = {}

        for infoset in sorted(self.infosets.keys()):
            avg_strategy = self.get_average_strategy(infoset)
            probs[infoset] = {action: str(prob) for action, prob in avg_strategy.items()}

        result = {
            'format': 'bluff-strategy-v1',
            'game': self.game.get('name', 'unknown'),
            'producer': 'CFR Solver',
            'probs': probs
        }

        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)

    def compute_game_value(self) -> float:
        """Compute game value under current average strategy."""
        old_get_strategy = self.get_strategy
        self.get_strategy = self.get_average_strategy

        value_p0, value_p1 = self._cfr_recursive(self.root_id, 1.0, 1.0)

        self.get_strategy = old_get_strategy
        return value_p0


def main():
    try:
        print(f"Current directory: {os.getcwd()}", flush=True)
        print("Loading game...", flush=True)
        sys.stdout.flush()

        with open('game.json', 'r') as f:
            game = json.load(f)

        print(f"Game loaded. Running CFR solver with 2000 iterations...", flush=True)
        sys.stdout.flush()

        solver = CFRSolver(game)
        solver.run(iterations=2000)

        print("Computing game value...", flush=True)
        sys.stdout.flush()

        value_p0 = solver.compute_game_value()
        print(f"Game value for P0: {value_p0}", flush=True)
        sys.stdout.flush()

        print(f"Writing strategy to strategy.json in {os.getcwd()}...", flush=True)
        sys.stdout.flush()

        solver.write_strategy('strategy.json')

        # Verify the file was created
        if os.path.exists('strategy.json'):
            print(f"File created successfully. Size: {os.path.getsize('strategy.json')} bytes", flush=True)
        else:
            print("WARNING: strategy.json file was not created!", flush=True)

        sys.stdout.flush()

        print("Done!", flush=True)
        sys.stdout.flush()

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 1


if __name__ == '__main__':
    sys.exit(main())
