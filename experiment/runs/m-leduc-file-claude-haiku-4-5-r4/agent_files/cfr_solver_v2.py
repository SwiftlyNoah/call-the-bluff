#!/usr/bin/env python3
"""CFR+ solver for Leduc hold'em with convergence metrics."""

import json
from fractions import Fraction
from collections import defaultdict
import sys

def parse_number(s):
    """Parse a number from string to Fraction."""
    if '/' in s:
        parts = s.split('/')
        return Fraction(int(parts[0]), int(parts[1]))
    else:
        return Fraction(s)

class CFRSolver:
    def __init__(self, game_path):
        with open(game_path) as f:
            self.game = json.load(f)

        self.nodes = self.game['nodes']
        self.root = self.game['root']
        self.name = self.game['name']

        # Build node lookup
        self.node_map = {node['id']: node for node in self.nodes}

        # Regrets and strategy sums
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

        # Collect all infosets
        self.infosets = {}
        for node in self.nodes:
            if node['kind'] == 'decision':
                infoset = node['infoset']
                if infoset not in self.infosets:
                    self.infosets[infoset] = {
                        'player': node['player'],
                        'actions': [a['label'] for a in node['actions']]
                    }

        # For convergence tracking
        self.game_values = []

    def get_strategy(self, infoset):
        """Get current strategy for an infoset using regret matching."""
        actions = self.infosets[infoset]['actions']
        regrets = self.regrets[infoset]

        # Regret matching: use positive regrets
        strategy = {}
        total_positive_regret = 0.0

        for action in actions:
            regret = max(0.0, regrets.get(action, 0.0))
            strategy[action] = regret
            total_positive_regret += regret

        # Normalize
        if total_positive_regret > 0:
            for action in actions:
                strategy[action] /= total_positive_regret
        else:
            # Uniform strategy
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform_prob

        return strategy

    def cfr(self, node_id, reach_probs, iteration):
        """
        Counterfactual Regret Minimization.

        reach_probs: dict mapping player (0, 1) to their reach probability
        Returns: value for player 0 at this node
        """
        node = self.node_map[node_id]

        if node['kind'] == 'terminal':
            payoff = parse_number(node['payoff_p0'])
            return float(payoff)

        if node['kind'] == 'chance':
            # Chance node: take expectation over outcomes
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_number(outcome['prob'])
                child_id = outcome['child']
                child_value = self.cfr(child_id, reach_probs.copy(), iteration)
                value += float(prob) * child_value
            return value

        # Decision node
        player = node['player']
        infoset = node['infoset']
        actions = node['actions']

        # Get current strategy
        strategy = self.get_strategy(infoset)

        # Compute value for each action
        action_values = {}
        node_value = 0.0

        for action in actions:
            action_obj = next(a for a in actions if a['label'] == action['label'])
            child_id = action_obj['child']

            # Update reach probability
            new_reach_probs = reach_probs.copy()
            new_reach_probs[player] *= strategy[action['label']]

            # Recursively compute value
            child_value = self.cfr(child_id, new_reach_probs, iteration)
            action_values[action['label']] = child_value
            node_value += strategy[action['label']] * child_value

        # Update regrets
        opponent_reach = reach_probs[1 - player]

        for action in actions:
            action_label = action['label']
            regret = action_values[action_label] - node_value
            # Weight by opponent's reach probability
            self.regrets[infoset][action_label] += opponent_reach * regret
            # Accumulate strategy
            self.strategy_sum[infoset][action_label] += reach_probs[player] * strategy[action_label]

        return node_value

    def solve(self, iterations=2000):
        """Run CFR+ for the specified number of iterations."""
        print(f"Running CFR+ for {iterations} iterations...")
        for i in range(iterations):
            if (i + 1) % 100 == 0:
                print(f"  Iteration {i + 1}/{iterations}")
            reach_probs = {0: 1.0, 1: 1.0}
            game_value = self.cfr(self.root, reach_probs, i)
            self.game_values.append(game_value)
        print("Done!")

    def get_average_strategy(self):
        """Get the average strategy from cumulative strategy sums."""
        avg_strategy = {}

        for infoset, actions in self.infosets.items():
            avg_strategy[infoset] = {}
            strategy_sum = self.strategy_sum[infoset]
            total = sum(strategy_sum.values())

            for action in actions['actions']:
                if total > 0:
                    prob = strategy_sum.get(action, 0.0) / total
                else:
                    prob = 1.0 / len(actions['actions'])
                avg_strategy[infoset][action] = prob

        return avg_strategy

    def compute_game_value(self):
        """Compute the game value by playing the average strategy."""
        avg_strategy = self.get_average_strategy()
        self.current_strategy = avg_strategy
        reach_probs = {0: 1.0, 1: 1.0}
        return self.evaluate(self.root, reach_probs)

    def evaluate(self, node_id, reach_probs):
        """Evaluate a node using the average strategy."""
        node = self.node_map[node_id]

        if node['kind'] == 'terminal':
            payoff = parse_number(node['payoff_p0'])
            return float(payoff)

        if node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_number(outcome['prob'])
                child_id = outcome['child']
                child_value = self.evaluate(child_id, reach_probs.copy())
                value += float(prob) * child_value
            return value

        # Decision node
        player = node['player']
        infoset = node['infoset']
        actions = node['actions']
        strategy = self.current_strategy[infoset]

        node_value = 0.0
        for action in actions:
            action_obj = next(a for a in actions if a['label'] == action['label'])
            child_id = action_obj['child']

            new_reach_probs = reach_probs.copy()
            new_reach_probs[player] *= strategy[action['label']]

            child_value = self.evaluate(child_id, new_reach_probs)
            node_value += strategy[action['label']] * child_value

        return node_value

    def save_strategy(self, output_path):
        """Save the strategy to a JSON file."""
        avg_strategy = self.get_average_strategy()

        result = {
            'format': 'bluff-strategy-v1',
            'game': self.name,
            'producer': 'CFR+ solver',
            'probs': {}
        }

        for infoset in sorted(avg_strategy.keys()):
            probs = {}
            for action in sorted(avg_strategy[infoset].keys()):
                # Convert to string representation
                prob = avg_strategy[infoset][action]
                probs[action] = repr(prob)
            result['probs'][infoset] = probs

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"Strategy saved to {output_path}")

def main():
    solver = CFRSolver('game.json')
    print(f"Game: {solver.name}")
    print(f"Number of infosets: {len(solver.infosets)}")

    solver.solve(iterations=2000)

    # Print convergence info
    print(f"\nGame value trajectory (last 20 iterations):")
    for i, v in enumerate(solver.game_values[-20:], start=len(solver.game_values)-19):
        print(f"  Iteration {i}: {v:.6f}")

    # Compute final game value
    game_value = solver.compute_game_value()
    print(f"\nFinal game value (Player 0): {game_value:.6f}")

    solver.save_strategy('strategy.json')

if __name__ == '__main__':
    main()
