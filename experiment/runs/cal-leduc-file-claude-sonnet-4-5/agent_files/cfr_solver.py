#!/usr/bin/env python3
"""CFR+ solver for Leduc hold'em"""

import json
from fractions import Fraction
from collections import defaultdict
from typing import Dict, List, Any
import sys


def parse_rational(s: str) -> float:
    """Parse a rational number string to float."""
    if '/' in s:
        num, denom = s.split('/')
        return float(num) / float(denom)
    return float(s)


class CFRSolver:
    def __init__(self, game_data: Dict[str, Any]):
        self.game_data = game_data
        self.nodes = {node['id']: node for node in game_data['nodes']}
        self.root = game_data['root']

        # Identify all information sets and their actions
        self.infosets = {}
        for node in game_data['nodes']:
            if node['kind'] == 'decision':
                infoset = node['infoset']
                if infoset not in self.infosets:
                    actions = [action['label'] for action in node['actions']]
                    player = node['player']
                    self.infosets[infoset] = {
                        'actions': actions,
                        'player': player
                    }

        # CFR+ data structures
        self.regret_sum = {infoset: defaultdict(float) for infoset in self.infosets}
        self.strategy_sum = {infoset: defaultdict(float) for infoset in self.infosets}
        self.iterations = 0

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an information set using regret matching."""
        actions = self.infosets[infoset]['actions']
        regrets = self.regret_sum[infoset]

        # CFR+: use max(regret, 0)
        positive_regrets = {action: max(regrets[action], 0.0) for action in actions}
        total_regret = sum(positive_regrets.values())

        if total_regret > 0:
            strategy = {action: positive_regrets[action] / total_regret for action in actions}
        else:
            # Uniform strategy if no positive regrets
            uniform_prob = 1.0 / len(actions)
            strategy = {action: uniform_prob for action in actions}

        return strategy

    def cfr(self, node_id: int, reach_probs: List[float]) -> float:
        """
        Run CFR recursion.
        reach_probs[0] = probability player 0 reaches this node
        reach_probs[1] = probability player 1 reaches this node
        Returns: expected utility for player 0
        """
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_rational(node['payoff_p0'])

        if node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                child_value = self.cfr(outcome['child'], reach_probs)
                value += prob * child_value
            return value

        # Decision node
        infoset = node['infoset']
        player = node['player']
        actions = self.infosets[infoset]['actions']
        strategy = self.get_strategy(infoset)

        # Compute action utilities
        action_utils = {}
        for action_data in node['actions']:
            action = action_data['label']
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[action]
            action_utils[action] = self.cfr(action_data['child'], new_reach)

        # Node utility
        node_util = sum(strategy[action] * action_utils[action] for action in actions)

        # Update regrets and strategy sum for the acting player
        for action in actions:
            regret = action_utils[action] - node_util
            # Weight by opponent's reach probability
            opponent = 1 - player
            weighted_regret = reach_probs[opponent] * regret
            self.regret_sum[infoset][action] += weighted_regret

            # CFR+: floor regrets at 0
            if self.regret_sum[infoset][action] < 0:
                self.regret_sum[infoset][action] = 0.0

        # Accumulate strategy (weighted by player's reach probability)
        for action in actions:
            self.strategy_sum[infoset][action] += reach_probs[player] * strategy[action]

        return node_util

    def train(self, iterations: int):
        """Run CFR for a number of iterations."""
        for i in range(iterations):
            self.cfr(self.root, [1.0, 1.0])
            self.iterations += 1

            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}/{iterations}", file=sys.stderr)

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy from all iterations."""
        avg_strategy = {}

        for infoset in self.infosets:
            actions = self.infosets[infoset]['actions']
            total = sum(self.strategy_sum[infoset][action] for action in actions)

            if total > 0:
                probs = {action: self.strategy_sum[infoset][action] / total for action in actions}
            else:
                # If never reached, use uniform
                uniform_prob = 1.0 / len(actions)
                probs = {action: uniform_prob for action in actions}

            # Convert to strings
            avg_strategy[infoset] = {action: repr(probs[action]) for action in actions}

        return avg_strategy

    def compute_exploitability(self) -> float:
        """Compute exploitability of current average strategy."""
        avg_strategy = self.get_average_strategy()

        # Convert string probs back to floats for computation
        strategy_floats = {}
        for infoset, probs in avg_strategy.items():
            strategy_floats[infoset] = {action: float(prob) for action, prob in probs.items()}

        # Best response for player 0 (against player 1's strategy)
        br0_value = self.best_response_value(0, strategy_floats)

        # Best response for player 1 (against player 0's strategy)
        br1_value = self.best_response_value(1, strategy_floats)

        # Exploitability
        return (br0_value + br1_value) / 2.0

    def best_response_value(self, br_player: int, strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute best response value for br_player against the given strategy."""
        return self.br_recurse(self.root, br_player, strategy)

    def br_recurse(self, node_id: int, br_player: int, strategy: Dict[str, Dict[str, float]]) -> float:
        """Recursively compute best response value."""
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            payoff = parse_rational(node['payoff_p0'])
            return payoff if br_player == 0 else -payoff

        if node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                child_value = self.br_recurse(outcome['child'], br_player, strategy)
                value += prob * child_value
            return value

        # Decision node
        player = node['player']

        if player == br_player:
            # Best response player: take max
            max_value = float('-inf')
            for action_data in node['actions']:
                child_value = self.br_recurse(action_data['child'], br_player, strategy)
                max_value = max(max_value, child_value)
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
        avg_strategy = self.get_average_strategy()
        strategy_floats = {}
        for infoset, probs in avg_strategy.items():
            strategy_floats[infoset] = {action: float(prob) for action, prob in probs.items()}

        return self.strategy_value_recurse(self.root, strategy_floats)

    def strategy_value_recurse(self, node_id: int, strategy: Dict[str, Dict[str, float]]) -> float:
        """Recursively compute strategy value."""
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_rational(node['payoff_p0'])

        if node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                child_value = self.strategy_value_recurse(outcome['child'], strategy)
                value += prob * child_value
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
    solver = CFRSolver(game_data)
    print(f"Information sets: {len(solver.infosets)}", file=sys.stderr)

    # Train
    num_iterations = 100000
    print(f"\nRunning CFR+ for {num_iterations} iterations...", file=sys.stderr)
    solver.train(num_iterations)

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
        "producer": f"CFR+ solver, {num_iterations} iterations",
        "probs": avg_strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=1)

    print("Done!", file=sys.stderr)
    print(f"\nFinal exploitability: {exploitability}", file=sys.stderr)
    print(f"Final strategy value: {strategy_value}", file=sys.stderr)


if __name__ == '__main__':
    main()
