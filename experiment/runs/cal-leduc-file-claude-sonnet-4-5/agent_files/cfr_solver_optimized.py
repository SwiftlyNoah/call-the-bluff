#!/usr/bin/env python3
"""Optimized CFR+ solver for Leduc hold'em"""

import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import sys


def parse_rational(s: str) -> float:
    """Parse a rational number string to float."""
    if '/' in s:
        num, denom = s.split('/')
        return float(num) / float(denom)
    return float(s)


class OptimizedCFRSolver:
    def __init__(self, game_data: Dict[str, Any]):
        self.game_data = game_data
        self.nodes = game_data['nodes']
        self.root = game_data['root']

        # Build node lookup and preprocess
        self.node_by_id = {}
        for node in self.nodes:
            node_id = node['id']
            self.node_by_id[node_id] = node

            # Pre-parse probabilities and payoffs
            if node['kind'] == 'terminal':
                node['_payoff'] = parse_rational(node['payoff_p0'])
            elif node['kind'] == 'chance':
                for outcome in node['outcomes']:
                    outcome['_prob'] = parse_rational(outcome['prob'])

        # Identify all information sets and their actions
        self.infosets = {}
        self.infoset_to_player = {}
        for node in self.nodes:
            if node['kind'] == 'decision':
                infoset = node['infoset']
                if infoset not in self.infosets:
                    actions = tuple(action['label'] for action in node['actions'])
                    player = node['player']
                    self.infosets[infoset] = actions
                    self.infoset_to_player[infoset] = player

        # CFR+ data structures - use dicts for speed
        self.regret_sum = {}
        self.strategy_sum = {}
        for infoset in self.infosets:
            num_actions = len(self.infosets[infoset])
            self.regret_sum[infoset] = [0.0] * num_actions
            self.strategy_sum[infoset] = [0.0] * num_actions

        self.iterations = 0

    def get_strategy(self, infoset: str) -> List[float]:
        """Get current strategy for an information set using regret matching."""
        regrets = self.regret_sum[infoset]
        num_actions = len(regrets)

        # CFR+: use max(regret, 0)
        total_regret = 0.0
        for r in regrets:
            if r > 0:
                total_regret += r

        strategy = [0.0] * num_actions
        if total_regret > 0:
            for i in range(num_actions):
                strategy[i] = max(regrets[i], 0.0) / total_regret
        else:
            # Uniform strategy if no positive regrets
            uniform_prob = 1.0 / num_actions
            for i in range(num_actions):
                strategy[i] = uniform_prob

        return strategy

    def cfr(self, node_id: int, reach0: float, reach1: float) -> float:
        """
        Run CFR recursion.
        Returns: expected utility for player 0
        """
        node = self.node_by_id[node_id]
        kind = node['kind']

        if kind == 'terminal':
            return node['_payoff']

        if kind == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                child_value = self.cfr(outcome['child'], reach0, reach1)
                value += outcome['_prob'] * child_value
            return value

        # Decision node
        infoset = node['infoset']
        player = node['player']
        actions = node['actions']
        num_actions = len(actions)

        strategy = self.get_strategy(infoset)

        # Compute action utilities
        action_utils = [0.0] * num_actions
        for i, action_data in enumerate(actions):
            if player == 0:
                action_utils[i] = self.cfr(action_data['child'], reach0 * strategy[i], reach1)
            else:
                action_utils[i] = self.cfr(action_data['child'], reach0, reach1 * strategy[i])

        # Node utility
        node_util = 0.0
        for i in range(num_actions):
            node_util += strategy[i] * action_utils[i]

        # Update regrets and strategy sum
        opponent_reach = reach1 if player == 0 else reach0
        player_reach = reach0 if player == 0 else reach1

        regrets = self.regret_sum[infoset]
        strat_sum = self.strategy_sum[infoset]

        for i in range(num_actions):
            regret = action_utils[i] - node_util
            regrets[i] += opponent_reach * regret
            # CFR+: floor at 0
            if regrets[i] < 0:
                regrets[i] = 0.0

            # Accumulate strategy
            strat_sum[i] += player_reach * strategy[i]

        return node_util

    def train(self, iterations: int, check_every: int = 10000):
        """Run CFR for a number of iterations."""
        for i in range(iterations):
            self.cfr(self.root, 1.0, 1.0)
            self.iterations += 1

            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}/{iterations}", file=sys.stderr)

            if check_every > 0 and (i + 1) % check_every == 0:
                exploit = self.compute_exploitability()
                print(f"  Exploitability at {i + 1}: {exploit}", file=sys.stderr)

                # Early stopping if well converged
                if exploit < 0.1:
                    print(f"Converged! Stopping at iteration {i + 1}", file=sys.stderr)
                    break

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy from all iterations."""
        avg_strategy = {}

        for infoset in self.infosets:
            actions = self.infosets[infoset]
            strat_sum = self.strategy_sum[infoset]

            total = sum(strat_sum)

            if total > 0:
                probs = {actions[i]: repr(strat_sum[i] / total) for i in range(len(actions))}
            else:
                # If never reached, use uniform
                uniform_prob = 1.0 / len(actions)
                probs = {actions[i]: repr(uniform_prob) for i in range(len(actions))}

            avg_strategy[infoset] = probs

        return avg_strategy

    def compute_exploitability(self) -> float:
        """Compute exploitability of current average strategy."""
        # Get average strategy as floats
        strategy_floats = {}
        for infoset in self.infosets:
            actions = self.infosets[infoset]
            strat_sum = self.strategy_sum[infoset]
            total = sum(strat_sum)

            if total > 0:
                strategy_floats[infoset] = {actions[i]: strat_sum[i] / total for i in range(len(actions))}
            else:
                uniform_prob = 1.0 / len(actions)
                strategy_floats[infoset] = {actions[i]: uniform_prob for i in range(len(actions))}

        # Best response values
        br0_value = self.best_response_value(0, strategy_floats)
        br1_value = self.best_response_value(1, strategy_floats)

        return (br0_value + br1_value) / 2.0

    def best_response_value(self, br_player: int, strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute best response value for br_player against the given strategy."""
        return self.br_recurse(self.root, br_player, strategy)

    def br_recurse(self, node_id: int, br_player: int, strategy: Dict[str, Dict[str, float]]) -> float:
        """Recursively compute best response value."""
        node = self.node_by_id[node_id]
        kind = node['kind']

        if kind == 'terminal':
            payoff = node['_payoff']
            return payoff if br_player == 0 else -payoff

        if kind == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                child_value = self.br_recurse(outcome['child'], br_player, strategy)
                value += outcome['_prob'] * child_value
            return value

        # Decision node
        player = node['player']

        if player == br_player:
            # Best response player: take max
            max_value = float('-inf')
            for action_data in node['actions']:
                child_value = self.br_recurse(action_data['child'], br_player, strategy)
                if child_value > max_value:
                    max_value = child_value
            return max_value
        else:
            # Opponent: follow their strategy
            infoset = node['infoset']
            value = 0.0
            for action_data in node['actions']:
                action = action_data['label']
                prob = strategy[infoset][action]
                child_value = self.br_recurse(action_data['child'], br_player, strategy)
                value += prob * child_value
            return value

    def compute_strategy_value(self) -> float:
        """Compute expected value for player 0 when both follow the average strategy."""
        # Get average strategy as floats
        strategy_floats = {}
        for infoset in self.infosets:
            actions = self.infosets[infoset]
            strat_sum = self.strategy_sum[infoset]
            total = sum(strat_sum)

            if total > 0:
                strategy_floats[infoset] = {actions[i]: strat_sum[i] / total for i in range(len(actions))}
            else:
                uniform_prob = 1.0 / len(actions)
                strategy_floats[infoset] = {actions[i]: uniform_prob for i in range(len(actions))}

        return self.strategy_value_recurse(self.root, strategy_floats)

    def strategy_value_recurse(self, node_id: int, strategy: Dict[str, Dict[str, float]]) -> float:
        """Recursively compute strategy value."""
        node = self.node_by_id[node_id]
        kind = node['kind']

        if kind == 'terminal':
            return node['_payoff']

        if kind == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                child_value = self.strategy_value_recurse(outcome['child'], strategy)
                value += outcome['_prob'] * child_value
            return value

        # Decision node
        infoset = node['infoset']
        value = 0.0
        for action_data in node['actions']:
            action = action_data['label']
            prob = strategy[infoset][action]
            child_value = self.strategy_value_recurse(action_data['child'], strategy)
            value += prob * child_value
        return value


def main():
    # Load game
    print("Loading game...", file=sys.stderr)
    with open('game.json', 'r') as f:
        game_data = json.load(f)

    print(f"Game: {game_data['name']}", file=sys.stderr)
    print(f"Nodes: {len(game_data['nodes'])}", file=sys.stderr)

    # Create solver
    solver = OptimizedCFRSolver(game_data)
    print(f"Information sets: {len(solver.infosets)}", file=sys.stderr)

    # Train with more iterations
    num_iterations = 50000
    print(f"\nRunning CFR+ for up to {num_iterations} iterations...", file=sys.stderr)
    solver.train(num_iterations, check_every=10000)

    # Get average strategy
    print("\nComputing average strategy...", file=sys.stderr)
    avg_strategy = solver.get_average_strategy()

    # Compute metrics
    print("Computing exploitability...", file=sys.stderr)
    exploitability = solver.compute_exploitability()
    print(f"Exploitability: {exploitability}", file=sys.stderr)

    print("Computing strategy value...", file=sys.stderr)
    strategy_value = solver.compute_strategy_value()
    print(f"Strategy value (P0): {strategy_value}", file=sys.stderr)

    # Write strategy
    print("\nWriting strategy.json...", file=sys.stderr)
    output = {
        "format": "bluff-strategy-v1",
        "game": game_data['name'],
        "producer": f"CFR+ solver, {solver.iterations} iterations",
        "probs": avg_strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=1)

    print("Done!", file=sys.stderr)
    print(f"\nFinal exploitability: {exploitability}", file=sys.stderr)
    print(f"Final strategy value: {strategy_value}", file=sys.stderr)


if __name__ == '__main__':
    main()
