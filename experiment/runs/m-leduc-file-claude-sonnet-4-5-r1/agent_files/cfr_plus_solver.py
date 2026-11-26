#!/usr/bin/env python3
"""
CFR+ solver for Leduc hold'em (faster convergence than vanilla CFR).
Reads game.json, runs CFR+, writes strategy.json.
"""

import json
from fractions import Fraction
from collections import defaultdict
from typing import Dict, List, Tuple
import sys


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
            self.payoff_p0 = self._parse_number(node_data['payoff_p0'])

    @staticmethod
    def _parse_number(s: str) -> float:
        """Parse a number string (integer, fraction, or decimal)."""
        if '/' in s:
            return float(Fraction(s))
        return float(s)


class CFRPlusSolver:
    """CFR+ solver for two-player zero-sum games."""

    def __init__(self, game_file: str):
        with open(game_file, 'r') as f:
            self.game_data = json.load(f)

        self.name = self.game_data['name']
        self.nodes = {n['id']: GameNode(n) for n in self.game_data['nodes']}
        self.root = self.game_data['root']

        # CFR+ data structures
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def _parse_number(self, s: str) -> float:
        """Parse a number string."""
        if '/' in s:
            return float(Fraction(s))
        return float(s)

    def get_strategy(self, infoset: str, actions: List[dict]) -> Dict[str, float]:
        """Get current strategy for an infoset using regret matching."""
        action_labels = [a['label'] for a in actions]
        regrets = self.regret_sum[infoset]

        # Regret matching: use positive regrets (CFR+ floors regrets at 0)
        normalizing_sum = 0.0
        strategy = {}
        for action in action_labels:
            strategy[action] = max(0.0, regrets[action])
            normalizing_sum += strategy[action]

        # Normalize (or use uniform if all regrets non-positive)
        if normalizing_sum > 0:
            for action in action_labels:
                strategy[action] /= normalizing_sum
        else:
            uniform_prob = 1.0 / len(action_labels)
            for action in action_labels:
                strategy[action] = uniform_prob

        return strategy

    def cfr(self, node_id: int, reach_probs: Tuple[float, float]) -> float:
        """
        Run CFR recursively.

        Args:
            node_id: Current node ID
            reach_probs: (player 0 reach prob, player 1 reach prob)

        Returns:
            Expected value for player 0
        """
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = self._parse_number(outcome['prob'])
                value += prob * self.cfr(outcome['child'], reach_probs)
            return value

        # Decision node
        player = node.player
        infoset = node.infoset
        actions = node.actions

        strategy = self.get_strategy(infoset, actions)

        # Compute value for each action
        action_values = {}
        node_value = 0.0

        for action in actions:
            action_label = action['label']
            action_prob = strategy[action_label]

            # Update reach probabilities
            new_reach = list(reach_probs)
            new_reach[player] *= action_prob

            action_value = self.cfr(action['child'], tuple(new_reach))
            action_values[action_label] = action_value
            node_value += action_prob * action_value

        # Update regrets and strategy sum
        if player == 0:
            cfr_reach = reach_probs[1]
        else:
            cfr_reach = reach_probs[0]

        # CFR+: update regrets and floor at 0
        for action in actions:
            action_label = action['label']
            regret = action_values[action_label] - node_value
            self.regret_sum[infoset][action_label] = max(
                0.0,
                self.regret_sum[infoset][action_label] + cfr_reach * regret
            )

        # Add to strategy sum (weighted by own reach probability)
        for action in actions:
            action_label = action['label']
            self.strategy_sum[infoset][action_label] += reach_probs[player] * strategy[action_label]

        return node_value

    def train(self, iterations: int, report_every: int = 10000):
        """Run CFR+ for a given number of iterations."""
        print(f"Running CFR+ for {iterations} iterations...")
        for i in range(iterations):
            self.cfr(self.root, (1.0, 1.0))
            self.iterations += 1

            if (i + 1) % report_every == 0:
                print(f"  Iteration {i + 1}/{iterations}")

        print(f"Training complete: {self.iterations} total iterations")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy across all iterations."""
        avg_strategy = {}

        for infoset, action_sums in self.strategy_sum.items():
            normalizing_sum = sum(action_sums.values())

            if normalizing_sum > 0:
                avg_strategy[infoset] = {
                    action: repr(prob / normalizing_sum)
                    for action, prob in action_sums.items()
                }
            else:
                # Uniform strategy if never reached
                num_actions = len(action_sums)
                avg_strategy[infoset] = {
                    action: repr(1.0 / num_actions)
                    for action in action_sums.keys()
                }

        return avg_strategy

    def compute_exploitability(self) -> Tuple[float, float]:
        """
        Compute exploitability of the average strategy.
        Returns (exploitability, game_value).
        """
        avg_strategy = self.get_average_strategy()

        # Player 0 best response vs player 1's average strategy
        br0_value = self._best_response_value(0, avg_strategy, self.root)

        # Player 1 best response vs player 0's average strategy
        br1_value = self._best_response_value(1, avg_strategy, self.root)

        # Game value with both using average strategy
        game_value = self._strategy_value(avg_strategy, self.root)

        # Exploitability = average gain from best response
        exploitability = ((br0_value - game_value) + (br1_value + game_value)) / 2

        return exploitability, game_value

    def _strategy_value(self, strategy: Dict[str, Dict[str, str]], node_id: int) -> float:
        """Compute expected value when both players follow the given strategy."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = self._parse_number(outcome['prob'])
                value += prob * self._strategy_value(strategy, outcome['child'])
            return value

        # Decision node
        infoset = node.infoset
        value = 0.0

        for action in node.actions:
            action_label = action['label']
            prob = float(strategy[infoset][action_label])
            value += prob * self._strategy_value(strategy, action['child'])

        return value

    def _best_response_value(self, player: int, opp_strategy: Dict[str, Dict[str, str]],
                            node_id: int) -> float:
        """
        Compute best response value for a player against opponent's strategy.
        """
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0 if player == 0 else -node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = self._parse_number(outcome['prob'])
                value += prob * self._best_response_value(player, opp_strategy, outcome['child'])
            return value

        # Decision node
        if node.player == player:
            # Maximizing player: take best action
            best_value = float('-inf')
            for action in node.actions:
                action_value = self._best_response_value(player, opp_strategy, action['child'])
                best_value = max(best_value, action_value)
            return best_value
        else:
            # Opponent: follow their strategy
            infoset = node.infoset
            value = 0.0
            for action in node.actions:
                action_label = action['label']
                prob = float(opp_strategy[infoset][action_label])
                action_value = self._best_response_value(player, opp_strategy, action['child'])
                value += prob * action_value
            return value

    def save_strategy(self, output_file: str):
        """Save the average strategy to a JSON file."""
        avg_strategy = self.get_average_strategy()

        output = {
            "format": "bluff-strategy-v1",
            "game": self.name,
            "producer": f"CFR+ solver ({self.iterations} iterations)",
            "probs": avg_strategy
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Strategy saved to {output_file}")


def main():
    solver = CFRPlusSolver('game.json')

    # Run CFR+ for 100,000 iterations (converges faster than vanilla CFR)
    solver.train(100000, report_every=10000)

    # Save the strategy
    solver.save_strategy('strategy.json')

    # Compute exploitability
    print("\nComputing exploitability...")
    exploitability, game_value = solver.compute_exploitability()
    print(f"Exploitability: {exploitability:.6f}")
    print(f"Game value (P0): {game_value:.6f}")


if __name__ == '__main__':
    main()
