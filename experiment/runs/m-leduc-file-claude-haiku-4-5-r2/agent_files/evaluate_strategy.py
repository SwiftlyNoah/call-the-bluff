#!/usr/bin/env python3
import json
from fractions import Fraction

class StrategyEvaluator:
    def __init__(self, game_path: str, strategy_path: str):
        with open(game_path, 'r') as f:
            self.game = json.load(f)

        with open(strategy_path, 'r') as f:
            self.strategy_data = json.load(f)

        self.nodes = {node['id']: node for node in self.game['nodes']}
        self.root = self.game['root']
        self.strategy = self.strategy_data['probs']

    def _prob_to_float(self, prob_str):
        if isinstance(prob_str, (int, float)):
            return float(prob_str)
        prob_str = str(prob_str).strip()
        if '/' in prob_str:
            parts = prob_str.split('/')
            return float(parts[0]) / float(parts[1])
        else:
            return float(prob_str)

    def compute_expected_value(self):
        """Compute expected value when both players follow the strategy."""
        return self._visit_ev(self.root)

    def _visit_ev(self, node_id: int) -> float:
        """Recursively compute expected value for player 0."""
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            return self._prob_to_float(node['payoff_p0'])

        elif node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = self._prob_to_float(outcome['prob'])
                child_value = self._visit_ev(outcome['child'])
                value += prob * child_value
            return value

        elif node['kind'] == 'decision':
            infoset = node['infoset']
            actions = [a['label'] for a in node['actions']]

            if infoset not in self.strategy:
                # If infoset missing, use uniform strategy
                return sum(self._visit_ev(a['child']) for a in node['actions']) / len(node['actions'])

            strategy_probs = self.strategy[infoset]
            value = 0.0

            action_map = {a['label']: a['child'] for a in node['actions']}

            for action in actions:
                prob = self._prob_to_float(strategy_probs[action])
                child_id = action_map[action]
                child_value = self._visit_ev(child_id)
                value += prob * child_value

            return value

def main():
    evaluator = StrategyEvaluator('game.json', 'strategy.json')
    ev = evaluator.compute_expected_value()

    print(f"Expected value for player 0: {ev}")
    print(f"Expected value for player 1: {-ev}")

    # Analyze strategy convergence by looking at infoset counts
    strategy = evaluator.strategy
    total_infosets = len(strategy)

    # Count actions
    total_actions = sum(len(actions) for actions in strategy.values())
    avg_actions = total_actions / total_infosets if total_infosets > 0 else 0

    print(f"\nStrategy statistics:")
    print(f"  Total infosets: {total_infosets}")
    print(f"  Total actions: {total_actions}")
    print(f"  Average actions per infoset: {avg_actions:.2f}")

if __name__ == "__main__":
    main()
