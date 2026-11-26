#!/usr/bin/env python3
"""
Counterfactual Regret Minimization solver for two-player zero-sum games.
"""

import json
from fractions import Fraction
from collections import defaultdict
from typing import Dict, List, Tuple
import sys
import time


def parse_rational(s: str) -> float:
    """Parse a rational number string (integer, fraction, or decimal) to float."""
    if '/' in s:
        num, denom = s.split('/')
        return float(Fraction(int(num), int(denom)))
    return float(s)


class GameNode:
    """Represents a node in the game tree."""
    def __init__(self, node_data: dict):
        self.id = node_data['id']
        self.kind = node_data['kind']

        if self.kind == 'chance':
            self.outcomes = []
            for outcome in node_data['outcomes']:
                self.outcomes.append({
                    'child': outcome['child'],
                    'prob': parse_rational(outcome['prob'])
                })
        elif self.kind == 'decision':
            self.player = node_data['player']
            self.infoset = node_data['infoset']
            self.actions = []
            for action in node_data['actions']:
                self.actions.append({
                    'label': action['label'],
                    'child': action['child']
                })
        elif self.kind == 'terminal':
            self.payoff_p0 = parse_rational(node_data['payoff_p0'])


class CFRSolver:
    """CFR+ solver for two-player zero-sum games."""

    def __init__(self, game_file: str):
        """Load game from JSON file."""
        print("Loading game...", flush=True)
        with open(game_file, 'r') as f:
            self.game_data = json.load(f)

        self.game_name = self.game_data['name']
        self.root_id = self.game_data['root']

        # Build node lookup
        print("Building game tree...", flush=True)
        self.nodes = {}
        for node_data in self.game_data['nodes']:
            self.nodes[node_data['id']] = GameNode(node_data)

        print(f"Loaded {len(self.nodes)} nodes", flush=True)

        # CFR+ data structures
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

        # Cache infosets and their actions
        self.infosets = {}
        self._discover_infosets()
        print(f"Found {len(self.infosets)} information sets", flush=True)

    def _discover_infosets(self):
        """Discover all infosets and their actions."""
        for node in self.nodes.values():
            if node.kind == 'decision':
                if node.infoset not in self.infosets:
                    self.infosets[node.infoset] = [action['label'] for action in node.actions]

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an infoset using regret matching."""
        actions = self.infosets[infoset]
        strategy = {}

        # Regret matching with CFR+ (use positive regrets)
        regret_sum_positive = 0.0
        for action in actions:
            regret = max(0.0, self.regret_sum[infoset][action])
            strategy[action] = regret
            regret_sum_positive += regret

        # Normalize
        if regret_sum_positive > 0:
            for action in actions:
                strategy[action] /= regret_sum_positive
        else:
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform_prob

        return strategy

    def cfr(self, node_id: int, reach_p0: float, reach_p1: float) -> float:
        """Run CFR recursively. Returns expected value for player 0."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                value += outcome['prob'] * self.cfr(outcome['child'], reach_p0, reach_p1)
            return value

        # Decision node
        player = node.player
        infoset = node.infoset
        strategy = self.get_strategy(infoset)

        # Compute action values
        action_values = []
        node_value = 0.0

        for action_data in node.actions:
            action = action_data['label']
            if player == 0:
                action_value = self.cfr(action_data['child'],
                                       reach_p0 * strategy[action], reach_p1)
            else:
                action_value = self.cfr(action_data['child'],
                                       reach_p0, reach_p1 * strategy[action])

            action_values.append(action_value)
            node_value += strategy[action] * action_value

        # Update regrets
        cfr_reach = reach_p1 if player == 0 else reach_p0
        for i, action_data in enumerate(node.actions):
            action = action_data['label']
            regret = action_values[i] - node_value
            self.regret_sum[infoset][action] += cfr_reach * regret
            # CFR+: floor at 0
            self.regret_sum[infoset][action] = max(0.0, self.regret_sum[infoset][action])

        # Update strategy sum
        reach_player = reach_p0 if player == 0 else reach_p1
        for action_data in node.actions:
            action = action_data['label']
            self.strategy_sum[infoset][action] += reach_player * strategy[action]

        return node_value

    def train(self, iterations: int, check_every: int = 1000):
        """Train the CFR solver."""
        print(f"Training CFR+ for {iterations} iterations...", flush=True)
        start_time = time.time()

        for i in range(iterations):
            if (i + 1) % check_every == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Iteration {i + 1}/{iterations} ({rate:.1f} iter/sec)", flush=True)

            self.cfr(self.root_id, 1.0, 1.0)

        elapsed = time.time() - start_time
        print(f"Training complete in {elapsed:.1f}s", flush=True)

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy across all iterations."""
        avg_strategy = {}

        for infoset in self.infosets:
            actions = self.infosets[infoset]
            strategy = {}

            strategy_sum_total = sum(self.strategy_sum[infoset][action] for action in actions)

            if strategy_sum_total > 0:
                for action in actions:
                    prob = self.strategy_sum[infoset][action] / strategy_sum_total
                    strategy[action] = repr(prob)
            else:
                uniform_prob = 1.0 / len(actions)
                for action in actions:
                    strategy[action] = repr(uniform_prob)

            avg_strategy[infoset] = strategy

        return avg_strategy

    def compute_exploitability(self) -> float:
        """Compute exploitability of the average strategy."""
        print("Computing exploitability...", flush=True)
        avg_strategy = self.get_average_strategy()

        strategy_floats = {}
        for infoset, action_probs in avg_strategy.items():
            strategy_floats[infoset] = {action: float(prob) for action, prob in action_probs.items()}

        br0_value = self._best_response_value(0, strategy_floats)
        br1_value = self._best_response_value(1, strategy_floats)

        exploitability = (br0_value + br1_value) / 2.0
        return exploitability

    def _best_response_value(self, br_player: int, opponent_strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute best response value for br_player."""
        memo = {}

        def compute_value(node_id: int) -> float:
            if node_id in memo:
                return memo[node_id]

            node = self.nodes[node_id]

            if node.kind == 'terminal':
                value = node.payoff_p0 if br_player == 0 else -node.payoff_p0
            elif node.kind == 'chance':
                value = sum(outcome['prob'] * compute_value(outcome['child'])
                           for outcome in node.outcomes)
            elif node.player == br_player:
                # BR player: take best action
                value = max(compute_value(action['child']) for action in node.actions)
            else:
                # Opponent: use their strategy
                strategy = opponent_strategy[node.infoset]
                value = sum(strategy[action['label']] * compute_value(action['child'])
                           for action in node.actions)

            memo[node_id] = value
            return value

        return compute_value(self.root_id)

    def compute_strategy_value(self) -> float:
        """Compute expected value for P0 when both play average strategy."""
        print("Computing strategy value...", flush=True)
        avg_strategy = self.get_average_strategy()

        strategy_floats = {}
        for infoset, action_probs in avg_strategy.items():
            strategy_floats[infoset] = {action: float(prob) for action, prob in action_probs.items()}

        def compute_value(node_id: int) -> float:
            node = self.nodes[node_id]

            if node.kind == 'terminal':
                return node.payoff_p0
            elif node.kind == 'chance':
                return sum(outcome['prob'] * compute_value(outcome['child'])
                          for outcome in node.outcomes)
            else:
                strategy = strategy_floats[node.infoset]
                return sum(strategy[action['label']] * compute_value(action['child'])
                          for action in node.actions)

        return compute_value(self.root_id)

    def save_strategy(self, output_file: str):
        """Save the average strategy to JSON."""
        avg_strategy = self.get_average_strategy()

        output = {
            "format": "bluff-strategy-v1",
            "game": self.game_name,
            "producer": "CFR+ solver",
            "probs": avg_strategy
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=1)

        print(f"Strategy saved to {output_file}", flush=True)


def main():
    solver = CFRSolver('game.json')

    # Train with more iterations for better convergence
    iterations = 100000
    solver.train(iterations, check_every=2000)

    # Save strategy
    solver.save_strategy('strategy.json')

    # Compute metrics
    print("\nComputing metrics...", flush=True)
    exploitability = solver.compute_exploitability()
    strategy_value = solver.compute_strategy_value()

    print(f"\nExploitability: {exploitability:.8f}", flush=True)
    print(f"Strategy value (P0): {strategy_value:.8f}", flush=True)

    return exploitability, strategy_value


if __name__ == '__main__':
    main()
