#!/usr/bin/env python3
"""
CFR+ solver for two-player zero-sum extensive-form games.
Optimized for performance on large game trees.
"""

import json
from fractions import Fraction
from collections import defaultdict
from typing import Dict, List, Any, Optional
import sys


def parse_rational(s: str) -> float:
    """Parse a rational number string to float."""
    if '/' in s:
        num, denom = s.split('/')
        return float(Fraction(int(num), int(denom)))
    return float(s)


class GameNode:
    """Represents a node in the game tree."""
    __slots__ = ['id', 'kind', 'outcomes', 'player', 'infoset', 'actions', 'payoff_p0']

    def __init__(self, node_data: Dict[str, Any]):
        self.id = node_data['id']
        self.kind = node_data['kind']
        self.outcomes = None
        self.player = None
        self.infoset = None
        self.actions = None
        self.payoff_p0 = None

        if self.kind == 'chance':
            self.outcomes = node_data['outcomes']
        elif self.kind == 'decision':
            self.player = node_data['player']
            self.infoset = node_data['infoset']
            self.actions = node_data['actions']
        elif self.kind == 'terminal':
            self.payoff_p0 = parse_rational(node_data['payoff_p0'])


class CFRSolver:
    """CFR+ solver for two-player zero-sum games."""

    def __init__(self, game_path: str):
        """Load game from JSON file."""
        print(f"Loading game from {game_path}...", flush=True)
        with open(game_path, 'r') as f:
            game_data = json.load(f)

        self.game_name = game_data['name']
        self.root_id = game_data['root']

        print(f"Building node lookup...", flush=True)
        # Build node lookup
        self.nodes = {}
        for node_data in game_data['nodes']:
            self.nodes[node_data['id']] = GameNode(node_data)

        print(f"Loaded {len(self.nodes)} nodes", flush=True)

        # Collect information sets and their actions
        self.infosets = {}  # infoset -> (player, [action_labels])
        for node in self.nodes.values():
            if node.kind == 'decision':
                if node.infoset not in self.infosets:
                    action_labels = [a['label'] for a in node.actions]
                    self.infosets[node.infoset] = (node.player, action_labels)

        print(f"Found {len(self.infosets)} information sets", flush=True)

        # CFR+ data structures
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Compute current strategy for an information set using regret matching."""
        _, actions = self.infosets[infoset]

        # Get positive regrets (CFR+)
        regrets = {}
        total_regret = 0.0
        for action in actions:
            regret = max(self.regret_sum[infoset][action], 0.0)
            regrets[action] = regret
            total_regret += regret

        # Normalize to get strategy
        strategy = {}
        if total_regret > 0:
            for action in actions:
                strategy[action] = regrets[action] / total_regret
        else:
            # Uniform strategy if no positive regret
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform_prob

        return strategy

    def cfr(self, node_id: int, player: int, reach_p0: float, reach_p1: float) -> float:
        """
        Run CFR recursively from a node.

        Returns:
            Expected payoff for player 0
        """
        node = self.nodes[node_id]

        # Terminal node
        if node.kind == 'terminal':
            return node.payoff_p0

        # Chance node
        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_rational(outcome['prob'])
                value += prob * self.cfr(outcome['child'], player, reach_p0, reach_p1)
            return value

        # Decision node
        infoset = node.infoset
        acting_player = node.player

        # Get current strategy
        strategy = self.get_strategy(infoset)

        # Compute value for each action
        action_values = {}
        node_value = 0.0

        for action_data in node.actions:
            action = action_data['label']
            child_id = action_data['child']

            # Update reach probabilities
            if acting_player == 0:
                new_reach_p0 = reach_p0 * strategy[action]
                new_reach_p1 = reach_p1
            else:
                new_reach_p0 = reach_p0
                new_reach_p1 = reach_p1 * strategy[action]

            action_value = self.cfr(child_id, player, new_reach_p0, new_reach_p1)
            action_values[action] = action_value
            node_value += strategy[action] * action_value

        # Update regrets and strategy for the player being updated
        if acting_player == player:
            # Counterfactual reach is the opponent's reach
            cf_reach = reach_p1 if player == 0 else reach_p0

            for action_data in node.actions:
                action = action_data['label']
                regret = action_values[action] - node_value
                self.regret_sum[infoset][action] += cf_reach * regret

            # Accumulate strategy (weighted by own reach)
            own_reach = reach_p0 if player == 0 else reach_p1
            for action_data in node.actions:
                action = action_data['label']
                self.strategy_sum[infoset][action] += own_reach * strategy[action]

        return node_value

    def train(self, num_iterations: int, print_every: int = 5000):
        """Run CFR training for specified iterations."""
        print(f"Starting CFR training for {num_iterations} iterations...", flush=True)

        for i in range(num_iterations):
            # Alternate updating each player
            self.cfr(self.root_id, 0, 1.0, 1.0)
            self.cfr(self.root_id, 1, 1.0, 1.0)
            self.iterations += 2

            if (i + 1) % print_every == 0:
                print(f"Completed {i + 1} iterations", flush=True)

        print(f"Training complete: {self.iterations} total iterations", flush=True)

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy across all iterations."""
        avg_strategy = {}

        for infoset, (player, actions) in self.infosets.items():
            # Sum of strategy weights
            total = sum(self.strategy_sum[infoset][a] for a in actions)

            if total > 0:
                probs = {}
                for action in actions:
                    prob = self.strategy_sum[infoset][action] / total
                    probs[action] = repr(prob)
                avg_strategy[infoset] = probs
            else:
                # Uniform if no data
                uniform_prob = 1.0 / len(actions)
                probs = {action: repr(uniform_prob) for action in actions}
                avg_strategy[infoset] = probs

        return avg_strategy

    def compute_exploitability(self) -> float:
        """Compute exploitability of average strategy."""
        print("Computing exploitability...", flush=True)
        avg_strategy = self.get_average_strategy()

        # Convert string probs back to floats for computation
        strategy_probs = {}
        for infoset, probs in avg_strategy.items():
            strategy_probs[infoset] = {a: float(p) for a, p in probs.items()}

        # Compute best response value for each player
        br_value_p0 = self.best_response_value(0, strategy_probs)
        br_value_p1 = self.best_response_value(1, strategy_probs)

        # Compute game value under average strategy
        game_value = self.compute_strategy_value(strategy_probs)

        # Exploitability
        exploitability = (br_value_p0 - game_value + br_value_p1 - (-game_value)) / 2.0
        return exploitability

    def best_response_value(self, player: int, opponent_strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute best response value for a player against opponent's strategy."""
        return self._best_response_recursive(self.root_id, player, opponent_strategy)

    def _best_response_recursive(self, node_id: int, player: int, opponent_strategy: Dict[str, Dict[str, float]]) -> float:
        """Recursive helper for computing best response."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0 if player == 0 else -node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_rational(outcome['prob'])
                value += prob * self._best_response_recursive(outcome['child'], player, opponent_strategy)
            return value

        # Decision node
        if node.player == player:
            # Best response: pick action with max value
            max_value = float('-inf')
            for action_data in node.actions:
                child_value = self._best_response_recursive(action_data['child'], player, opponent_strategy)
                max_value = max(max_value, child_value)
            return max_value
        else:
            # Opponent plays their strategy
            infoset = node.infoset
            strategy = opponent_strategy[infoset]
            value = 0.0
            for action_data in node.actions:
                action = action_data['label']
                prob = strategy[action]
                child_value = self._best_response_recursive(action_data['child'], player, opponent_strategy)
                value += prob * child_value
            return value

    def compute_strategy_value(self, strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute expected value for player 0 when both players follow the strategy."""
        return self._strategy_value_recursive(self.root_id, strategy)

    def _strategy_value_recursive(self, node_id: int, strategy: Dict[str, Dict[str, float]]) -> float:
        """Recursive helper for computing strategy value."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_rational(outcome['prob'])
                value += prob * self._strategy_value_recursive(outcome['child'], strategy)
            return value

        # Decision node
        infoset = node.infoset
        node_strategy = strategy[infoset]
        value = 0.0
        for action_data in node.actions:
            action = action_data['label']
            prob = node_strategy[action]
            child_value = self._strategy_value_recursive(action_data['child'], strategy)
            value += prob * child_value
        return value

    def save_strategy(self, output_path: str):
        """Save average strategy to JSON file."""
        print(f"Saving strategy to {output_path}...", flush=True)
        avg_strategy = self.get_average_strategy()

        output = {
            'format': 'bluff-strategy-v1',
            'game': self.game_name,
            'producer': f'CFR+ solver ({self.iterations} iterations)',
            'probs': avg_strategy
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=1)

        print(f"Strategy saved to {output_path}", flush=True)


def main():
    solver = CFRSolver('game.json')

    # Train with CFR+ (use fewer iterations for speed, but should still converge)
    solver.train(num_iterations=50000, print_every=5000)

    # Save strategy
    solver.save_strategy('strategy.json')

    # Compute final metrics
    print("\nComputing final metrics...", flush=True)
    avg_strategy = solver.get_average_strategy()
    strategy_probs = {infoset: {a: float(p) for a, p in probs.items()}
                      for infoset, probs in avg_strategy.items()}

    exploitability = solver.compute_exploitability()
    value_p0 = solver.compute_strategy_value(strategy_probs)

    print(f"\n=== Final Results ===", flush=True)
    print(f"Exploitability: {exploitability:.8f}", flush=True)
    print(f"Game value (P0): {value_p0:.8f}", flush=True)
    print(f"Iterations: {solver.iterations}", flush=True)


if __name__ == '__main__':
    main()
