#!/usr/bin/env python3
"""CFR solver for Leduc hold'em."""

import json
from fractions import Fraction
from collections import defaultdict
from typing import Dict
import time
import sys


def parse_rational(s: str) -> float:
    """Parse a rational number string to float."""
    if '/' in s:
        num, den = s.split('/')
        return float(Fraction(int(num), int(den)))
    return float(s)


class GameNode:
    """A node in the game tree."""

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
    """Counterfactual Regret Minimization solver."""

    def __init__(self, game_data: dict):
        self.game_name = game_data['name']
        self.nodes = {node['id']: GameNode(node) for node in game_data['nodes']}
        self.root_id = game_data['root']

        # Regret and strategy storage
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.infoset_actions = {}  # Cache action lists for each infoset

        # Collect all infosets
        for node in self.nodes.values():
            if node.kind == 'decision':
                if node.infoset not in self.infoset_actions:
                    self.infoset_actions[node.infoset] = [a['label'] for a in node.actions]

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an infoset using regret matching."""
        actions = self.infoset_actions[infoset]
        strategy = {}

        # Regret matching
        regrets = self.regret_sum[infoset]
        normalizing_sum = sum(max(0, regrets[a]) for a in actions)

        if normalizing_sum > 0:
            for action in actions:
                strategy[action] = max(0, regrets[action]) / normalizing_sum
        else:
            # Uniform strategy if no positive regrets
            uniform = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform

        return strategy

    def cfr(self, node_id: int, player: int, reach_p0: float, reach_p1: float) -> float:
        """
        Compute counterfactual regret for a node.

        Args:
            node_id: Current node ID
            player: Player to update (0 or 1)
            reach_p0: Reach probability for player 0
            reach_p1: Reach probability for player 1

        Returns:
            Expected utility for player 0
        """
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            # Chance node: take expectation over outcomes
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_rational(outcome['prob'])
                child_value = self.cfr(outcome['child'], player, reach_p0, reach_p1)
                value += prob * child_value
            return value

        # Decision node
        infoset = node.infoset
        actions = [a['label'] for a in node.actions]
        strategy = self.get_strategy(infoset)

        # Compute value for each action
        action_values = {}
        node_value = 0.0

        for action_data in node.actions:
            action = action_data['label']
            child_id = action_data['child']

            if node.player == 0:
                action_values[action] = self.cfr(
                    child_id, player,
                    reach_p0 * strategy[action],
                    reach_p1
                )
            else:
                action_values[action] = self.cfr(
                    child_id, player,
                    reach_p0,
                    reach_p1 * strategy[action]
                )

            node_value += strategy[action] * action_values[action]

        # Update regrets and strategy for the acting player
        if node.player == player:
            # Counterfactual reach (opponent's reach)
            cf_reach = reach_p1 if player == 0 else reach_p0

            for action in actions:
                regret = action_values[action] - node_value
                self.regret_sum[infoset][action] += cf_reach * regret

            # Add to strategy sum (weighted by own reach)
            own_reach = reach_p0 if player == 0 else reach_p1
            for action in actions:
                self.strategy_sum[infoset][action] += own_reach * strategy[action]

        return node_value

    def train(self, iterations: int, print_every: int = 1000):
        """Run CFR for a number of iterations."""
        print(f"Running CFR for {iterations} iterations...", flush=True)
        start_time = time.time()

        for i in range(iterations):
            if (i + 1) % print_every == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"Iteration {i + 1}/{iterations} ({elapsed:.1f}s, {rate:.1f} it/s)", flush=True)

            # Update player 0
            self.cfr(self.root_id, 0, 1.0, 1.0)
            # Update player 1
            self.cfr(self.root_id, 1, 1.0, 1.0)

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.1f}s", flush=True)

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy over all iterations."""
        avg_strategy = {}

        for infoset, actions in self.infoset_actions.items():
            strategy = {}
            normalizing_sum = sum(self.strategy_sum[infoset][a] for a in actions)

            if normalizing_sum > 0:
                for action in actions:
                    prob = self.strategy_sum[infoset][action] / normalizing_sum
                    strategy[action] = repr(prob)
            else:
                # Uniform if never visited (shouldn't happen with proper CFR)
                uniform_prob = 1.0 / len(actions)
                for action in actions:
                    strategy[action] = repr(uniform_prob)

            avg_strategy[infoset] = strategy

        return avg_strategy

    def compute_exploitability(self) -> float:
        """Compute exploitability of average strategy."""
        avg_strategy = self.get_average_strategy()

        # Convert to float strategy for best response calculation
        float_strategy = {}
        for infoset, probs in avg_strategy.items():
            float_strategy[infoset] = {a: float(p) for a, p in probs.items()}

        # Compute best response value for each player
        br_value_p0 = self.best_response_value(0, float_strategy)
        br_value_p1 = self.best_response_value(1, float_strategy)

        return (br_value_p0 + br_value_p1) / 2.0

    def best_response_value(self, player: int, opponent_strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute best response value for a player against an opponent strategy."""
        return self._br_recursive(self.root_id, player, opponent_strategy, 1.0)

    def _br_recursive(self, node_id: int, player: int, opponent_strategy: Dict[str, Dict[str, float]],
                      opponent_reach: float) -> float:
        """Recursive best response computation."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0 if player == 0 else -node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_rational(outcome['prob'])
                child_value = self._br_recursive(outcome['child'], player, opponent_strategy, opponent_reach)
                value += prob * child_value
            return value

        # Decision node
        if node.player == player:
            # Player to optimize: take maximum over actions
            best_value = float('-inf')
            for action_data in node.actions:
                child_value = self._br_recursive(action_data['child'], player, opponent_strategy, opponent_reach)
                best_value = max(best_value, child_value)
            return best_value
        else:
            # Opponent: take expectation over their strategy
            value = 0.0
            strategy = opponent_strategy[node.infoset]
            for action_data in node.actions:
                action = action_data['label']
                prob = strategy[action]
                child_value = self._br_recursive(action_data['child'], player, opponent_strategy,
                                                opponent_reach * prob)
                value += prob * child_value
            return value

    def compute_strategy_value(self) -> float:
        """Compute expected value for player 0 when both players follow average strategy."""
        avg_strategy = self.get_average_strategy()
        float_strategy = {}
        for infoset, probs in avg_strategy.items():
            float_strategy[infoset] = {a: float(p) for a, p in probs.items()}

        return self._strategy_value_recursive(self.root_id, float_strategy)

    def _strategy_value_recursive(self, node_id: int, strategy: Dict[str, Dict[str, float]]) -> float:
        """Recursive strategy value computation."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_rational(outcome['prob'])
                child_value = self._strategy_value_recursive(outcome['child'], strategy)
                value += prob * child_value
            return value

        # Decision node
        value = 0.0
        node_strategy = strategy[node.infoset]
        for action_data in node.actions:
            action = action_data['label']
            prob = node_strategy[action]
            child_value = self._strategy_value_recursive(action_data['child'], strategy)
            value += prob * child_value
        return value


def main():
    # Load game
    print("Loading game...", flush=True)
    with open('game.json', 'r') as f:
        game_data = json.load(f)

    print(f"Game loaded: {len(game_data['nodes'])} nodes", flush=True)

    # Create solver
    print("Initializing solver...", flush=True)
    solver = CFRSolver(game_data)
    print(f"Found {len(solver.infoset_actions)} information sets", flush=True)

    # Train - use 25000 iterations to fit within time constraints
    num_iterations = 25000
    solver.train(num_iterations, print_every=5000)

    # Get average strategy
    print("\nComputing average strategy...", flush=True)
    avg_strategy = solver.get_average_strategy()

    # Compute metrics
    print("Computing exploitability...", flush=True)
    exploitability = solver.compute_exploitability()
    print(f"Exploitability: {exploitability:.6f}", flush=True)

    print("Computing strategy value...", flush=True)
    strategy_value = solver.compute_strategy_value()
    print(f"Expected value for P0: {strategy_value:.6f}", flush=True)

    # Save strategy
    print("Saving strategy...", flush=True)
    output = {
        "format": "bluff-strategy-v1",
        "game": game_data['name'],
        "producer": f"CFR solver ({num_iterations} iterations)",
        "probs": avg_strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=1)

    print("\nStrategy saved to strategy.json", flush=True)
    print(f"\nFinal metrics:", flush=True)
    print(f"  Exploitability: {exploitability:.6f}", flush=True)
    print(f"  P0 value: {strategy_value:.6f}", flush=True)


if __name__ == '__main__':
    main()
