#!/usr/bin/env python3
"""
Counterfactual Regret Minimization solver for Leduc hold'em.
"""

import json
from fractions import Fraction
from typing import Dict, List, Tuple
from collections import defaultdict


def parse_rational(s: str) -> float:
    """Parse a rational number from string format."""
    if '/' in s:
        num, den = s.split('/')
        return float(Fraction(int(num), int(den)))
    return float(s)


class GameNode:
    """Represents a node in the game tree."""
    def __init__(self, node_data: dict):
        self.id = node_data['id']
        self.kind = node_data['kind']

        if self.kind == 'chance':
            self.outcomes = node_data['outcomes']
        elif self.kind == 'decision':
            self.player = node_data['player']
            self.infoset = node_data['infoset']
            self.actions = node_data['actions']
        elif self.kind == 'terminal':
            self.payoff_p0 = parse_rational(node_data['payoff_p0'])


class CFRSolver:
    """Implements Counterfactual Regret Minimization."""

    def __init__(self, game_file: str):
        """Load the game from JSON file."""
        with open(game_file, 'r') as f:
            game_data = json.load(f)

        self.game_name = game_data['name']
        self.root_id = game_data['root']

        # Build node lookup
        self.nodes = {}
        for node_data in game_data['nodes']:
            self.nodes[node_data['id']] = GameNode(node_data)

        # Get all infosets for each player
        self.infosets = {0: set(), 1: set()}
        self.infoset_actions = {}
        for node in self.nodes.values():
            if node.kind == 'decision':
                self.infosets[node.player].add(node.infoset)
                if node.infoset not in self.infoset_actions:
                    self.infoset_actions[node.infoset] = [a['label'] for a in node.actions]

        # Initialize regret sums and strategy sums
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an infoset using regret matching."""
        actions = self.infoset_actions[infoset]
        strategy = {}

        # Calculate normalizing sum (only positive regrets)
        normalizing_sum = 0.0
        for action in actions:
            normalizing_sum += max(0.0, self.regret_sum[infoset][action])

        # If all regrets are non-positive, use uniform strategy
        if normalizing_sum > 0:
            for action in actions:
                strategy[action] = max(0.0, self.regret_sum[infoset][action]) / normalizing_sum
        else:
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform_prob

        return strategy

    def cfr(self, node_id: int, reach_probs: List[float]) -> float:
        """
        Run CFR recursion.

        Args:
            node_id: Current node ID
            reach_probs: [prob_p0, prob_p1] - reach probabilities for each player

        Returns:
            Expected value for player 0 at this node
        """
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        elif node.kind == 'chance':
            # Chance node: expected value over outcomes
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_rational(outcome['prob'])
                value += prob * self.cfr(outcome['child'], reach_probs)
            return value

        elif node.kind == 'decision':
            player = node.player
            infoset = node.infoset
            actions = [a['label'] for a in node.actions]

            # Get current strategy
            strategy = self.get_strategy(infoset)

            # Compute value for each action
            action_values = {}
            node_value = 0.0

            for i, action_data in enumerate(node.actions):
                action = action_data['label']
                child_id = action_data['child']

                # Update reach probabilities for recursive call
                new_reach = reach_probs.copy()
                new_reach[player] *= strategy[action]

                action_values[action] = self.cfr(child_id, new_reach)
                node_value += strategy[action] * action_values[action]

            # Update regrets and strategy sum
            for action in actions:
                regret = action_values[action] - node_value
                opponent = 1 - player
                self.regret_sum[infoset][action] += reach_probs[opponent] * regret
                # CFR+: floor negative regrets at 0
                self.regret_sum[infoset][action] = max(0.0, self.regret_sum[infoset][action])
                self.strategy_sum[infoset][action] += reach_probs[player] * strategy[action]

            return node_value

    def train(self, iterations: int):
        """Run CFR+ for specified number of iterations."""
        import sys
        print(f"Running CFR+ for {iterations} iterations...", flush=True)
        for i in range(iterations):
            if (i + 1) % 1000 == 0:
                print(f"  Iteration {i + 1}/{iterations}", flush=True)
            self.cfr(self.root_id, [1.0, 1.0])
            self.iterations += 1
        print(f"Training complete. Total iterations: {self.iterations}", flush=True)

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy over all iterations."""
        avg_strategy = {}

        for player in [0, 1]:
            for infoset in self.infosets[player]:
                actions = self.infoset_actions[infoset]

                # Calculate normalizing sum
                normalizing_sum = sum(self.strategy_sum[infoset][a] for a in actions)

                if normalizing_sum > 0:
                    avg_strategy[infoset] = {}
                    for action in actions:
                        prob = self.strategy_sum[infoset][action] / normalizing_sum
                        avg_strategy[infoset][action] = repr(prob)
                else:
                    # If no data, use uniform
                    uniform_prob = 1.0 / len(actions)
                    avg_strategy[infoset] = {a: repr(uniform_prob) for a in actions}

        return avg_strategy

    def save_strategy(self, output_file: str):
        """Save the average strategy to JSON file."""
        avg_strategy = self.get_average_strategy()

        output = {
            "format": "bluff-strategy-v1",
            "game": self.game_name,
            "producer": "CFR+ solver",
            "probs": avg_strategy
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Strategy saved to {output_file}")


def main():
    """Main entry point."""
    print("Loading game...", flush=True)
    solver = CFRSolver('game.json')
    print(f"Game loaded. Found {len(solver.nodes)} nodes.", flush=True)

    # Train for sufficient iterations to converge
    # CFR+ converges faster than vanilla CFR
    solver.train(100000)

    # Save the resulting strategy
    solver.save_strategy('strategy.json')


if __name__ == '__main__':
    main()
