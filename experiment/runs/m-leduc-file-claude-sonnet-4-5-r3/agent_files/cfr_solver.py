#!/usr/bin/env python3
"""
CFR+ solver for two-player zero-sum extensive-form games.
CFR+ converges faster than vanilla CFR by using regret floors.
"""

import json
import sys
from fractions import Fraction
from typing import Dict, List, Tuple
from collections import defaultdict


class GameNode:
    """Represents a node in the game tree."""
    def __init__(self, node_dict):
        self.id = node_dict['id']
        self.kind = node_dict['kind']

        if self.kind == 'chance':
            self.outcomes = node_dict['outcomes']
        elif self.kind == 'decision':
            self.player = node_dict['player']
            self.infoset = node_dict['infoset']
            self.actions = node_dict['actions']
        elif self.kind == 'terminal':
            self.payoff_p0 = self._parse_number(node_dict['payoff_p0'])

    @staticmethod
    def _parse_number(s):
        """Parse a rational number string to float."""
        if '/' in s:
            return float(Fraction(s))
        return float(s)


class CFRPlusSolver:
    """CFR+ solver with improved convergence."""

    def __init__(self, game_file):
        """Load game from JSON file."""
        with open(game_file, 'r') as f:
            game_data = json.load(f)

        self.game_name = game_data['name']
        self.root_id = game_data['root']

        # Build node lookup
        self.nodes = {}
        for node_data in game_data['nodes']:
            node = GameNode(node_data)
            self.nodes[node.id] = node

        # Track information sets and actions
        self.infosets = {}
        for node in self.nodes.values():
            if node.kind == 'decision':
                if node.infoset not in self.infosets:
                    self.infosets[node.infoset] = [a['label'] for a in node.actions]

        # CFR+ data structures
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def get_strategy(self, infoset):
        """Compute current strategy using regret matching with floor at 0."""
        actions = self.infosets[infoset]
        strategy = {}

        # CFR+: floor regrets at 0
        regrets = {a: max(0.0, self.regret_sum[infoset][a]) for a in actions}
        normalizing_sum = sum(regrets.values())

        if normalizing_sum > 0:
            for a in actions:
                strategy[a] = regrets[a] / normalizing_sum
        else:
            uniform_prob = 1.0 / len(actions)
            for a in actions:
                strategy[a] = uniform_prob

        return strategy

    def cfr(self, node_id, reach_p0, reach_p1):
        """Recursive CFR computation."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = GameNode._parse_number(outcome['prob'])
                child_value = self.cfr(outcome['child'], reach_p0, reach_p1)
                value += prob * child_value
            return value

        # Decision node
        player = node.player
        infoset = node.infoset
        actions = [a['label'] for a in node.actions]
        strategy = self.get_strategy(infoset)

        # Compute action values
        action_values = {}
        for action_data in node.actions:
            action = action_data['label']
            child_id = action_data['child']

            if player == 0:
                action_values[action] = self.cfr(child_id, reach_p0 * strategy[action], reach_p1)
            else:
                action_values[action] = self.cfr(child_id, reach_p0, reach_p1 * strategy[action])

        # Node value
        node_value = sum(strategy[a] * action_values[a] for a in actions)

        # Update regrets and strategy
        if player == 0:
            cf_reach = reach_p1
            for action in actions:
                regret = action_values[action] - node_value
                # CFR+: floor at 0
                self.regret_sum[infoset][action] = max(0.0, self.regret_sum[infoset][action] + cf_reach * regret)

            for action in actions:
                self.strategy_sum[infoset][action] += reach_p0 * strategy[action]
        else:
            cf_reach = reach_p0
            for action in actions:
                regret = action_values[action] - node_value
                # CFR+: floor at 0
                self.regret_sum[infoset][action] = max(0.0, self.regret_sum[infoset][action] + cf_reach * regret)

            for action in actions:
                self.strategy_sum[infoset][action] += reach_p1 * strategy[action]

        return node_value

    def train(self, iterations, check_interval=5000):
        """Run CFR+ for specified iterations."""
        print(f"Running CFR+ for {iterations} iterations...")
        sys.stdout.flush()

        for i in range(iterations):
            self.cfr(self.root_id, 1.0, 1.0)
            self.iterations += 1

            if (i + 1) % check_interval == 0:
                exploitability, value = self.compute_exploitability()
                print(f"  Iteration {i + 1}/{iterations} - Exploitability: {exploitability:.6f}, Value P0: {value:.6f}")
                sys.stdout.flush()

        print("Training complete.")
        sys.stdout.flush()

    def get_average_strategy(self):
        """Return the average strategy."""
        avg_strategy = {}

        for infoset in self.infosets:
            actions = self.infosets[infoset]
            avg_strategy[infoset] = {}

            normalizing_sum = sum(self.strategy_sum[infoset][a] for a in actions)

            if normalizing_sum > 0:
                for action in actions:
                    avg_strategy[infoset][action] = self.strategy_sum[infoset][action] / normalizing_sum
            else:
                uniform_prob = 1.0 / len(actions)
                for action in actions:
                    avg_strategy[infoset][action] = uniform_prob

        return avg_strategy

    def compute_exploitability(self):
        """Compute exploitability of the average strategy."""
        avg_strategy = self.get_average_strategy()

        value_p0 = self._eval_strategy(self.root_id, avg_strategy, avg_strategy)
        br_value_p0 = self._best_response_value(self.root_id, avg_strategy, 0)
        br_value_p1 = -self._best_response_value(self.root_id, avg_strategy, 1)

        exploitability = (br_value_p0 - value_p0 + br_value_p1 - (-value_p0)) / 2

        return exploitability, value_p0

    def _eval_strategy(self, node_id, strategy_p0, strategy_p1):
        """Evaluate expected value."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = GameNode._parse_number(outcome['prob'])
                value += prob * self._eval_strategy(outcome['child'], strategy_p0, strategy_p1)
            return value

        strategy = strategy_p0 if node.player == 0 else strategy_p1
        value = 0.0

        for action_data in node.actions:
            action = action_data['label']
            prob = strategy[node.infoset][action]
            value += prob * self._eval_strategy(action_data['child'], strategy_p0, strategy_p1)

        return value

    def _best_response_value(self, node_id, opponent_strategy, player):
        """Compute best response value."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0 if player == 0 else -node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = GameNode._parse_number(outcome['prob'])
                value += prob * self._best_response_value(outcome['child'], opponent_strategy, player)
            return value

        if node.player == player:
            best_value = float('-inf')
            for action_data in node.actions:
                child_value = self._best_response_value(action_data['child'], opponent_strategy, player)
                best_value = max(best_value, child_value)
            return best_value
        else:
            value = 0.0
            for action_data in node.actions:
                action = action_data['label']
                prob = opponent_strategy[node.infoset][action]
                value += prob * self._best_response_value(action_data['child'], opponent_strategy, player)
            return value

    def save_strategy(self, output_file):
        """Save average strategy to JSON."""
        avg_strategy = self.get_average_strategy()

        probs = {}
        for infoset, action_probs in avg_strategy.items():
            probs[infoset] = {action: repr(prob) for action, prob in action_probs.items()}

        strategy_data = {
            "format": "bluff-strategy-v1",
            "game": self.game_name,
            "producer": f"CFR+ solver ({self.iterations} iterations)",
            "probs": probs
        }

        with open(output_file, 'w') as f:
            json.dump(strategy_data, f, indent=2)

        print(f"Strategy saved to {output_file}")
        sys.stdout.flush()


def main():
    """Main entry point."""
    solver = CFRPlusSolver('game.json')

    # Train with CFR+
    solver.train(20000, check_interval=5000)

    # Final exploitability
    exploitability, value_p0 = solver.compute_exploitability()
    print(f"\nFinal Exploitability: {exploitability:.6f}")
    print(f"Final Value for P0: {value_p0:.6f}")
    sys.stdout.flush()

    # Save strategy
    solver.save_strategy('strategy.json')


if __name__ == '__main__':
    main()
