#!/usr/bin/env python3
"""CFR solver for two-player zero-sum games."""

import json
from fractions import Fraction
from collections import defaultdict
from typing import Dict, List, Any
import sys


def parse_rational(s: str) -> float:
    """Parse a rational number from string (integer, fraction, or decimal)."""
    if '/' in s:
        parts = s.split('/')
        return float(Fraction(int(parts[0]), int(parts[1])))
    return float(s)


class GameNode:
    """A node in the game tree."""
    def __init__(self, node_dict: Dict[str, Any]):
        self.id = node_dict['id']
        self.kind = node_dict['kind']

        if self.kind == 'chance':
            self.outcomes = node_dict['outcomes']
            for outcome in self.outcomes:
                outcome['prob'] = parse_rational(outcome['prob'])
        elif self.kind == 'decision':
            self.player = node_dict['player']
            self.infoset = node_dict['infoset']
            self.actions = node_dict['actions']
        elif self.kind == 'terminal':
            self.payoff_p0 = parse_rational(node_dict['payoff_p0'])


class CFRSolver:
    """CFR+ solver for two-player zero-sum games."""

    def __init__(self, game_data: Dict[str, Any]):
        self.game_name = game_data['name']
        self.root_id = game_data['root']

        # Build node lookup
        self.nodes = {}
        for node_dict in game_data['nodes']:
            self.nodes[node_dict['id']] = GameNode(node_dict)

        # Collect information sets and their actions
        self.infosets = defaultdict(list)  # infoset -> list of action labels
        for node in self.nodes.values():
            if node.kind == 'decision':
                if not self.infosets[node.infoset]:
                    self.infosets[node.infoset] = [a['label'] for a in node.actions]

        # CFR+ data structures
        self.regret_sum = {}  # (infoset, action) -> cumulative regret
        self.strategy_sum = {}  # (infoset, action) -> cumulative strategy

        # Initialize
        for infoset, actions in self.infosets.items():
            for action in actions:
                self.regret_sum[(infoset, action)] = 0.0
                self.strategy_sum[(infoset, action)] = 0.0

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an infoset using regret matching."""
        actions = self.infosets[infoset]

        # Regret matching with CFR+ (use max(0, regret))
        regrets = [max(0.0, self.regret_sum[(infoset, a)]) for a in actions]
        regret_sum = sum(regrets)

        if regret_sum > 0:
            strategy = {a: r / regret_sum for a, r in zip(actions, regrets)}
        else:
            # Uniform distribution if no positive regrets
            uniform_prob = 1.0 / len(actions)
            strategy = {a: uniform_prob for a in actions}

        return strategy

    def cfr(self, node_id: int, reach_probs: List[float]) -> float:
        """
        Run CFR recursion. Updates regrets for both players.
        reach_probs[0] = reach probability for player 0
        reach_probs[1] = reach probability for player 1
        Returns expected value for player 0.
        """
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                child_value = self.cfr(outcome['child'], reach_probs[:])
                value += outcome['prob'] * child_value
            return value

        # Decision node
        infoset = node.infoset
        acting_player = node.player
        strategy = self.get_strategy(infoset)

        # Compute value for each action
        action_values = {}
        node_value = 0.0

        for action_dict in node.actions:
            action = action_dict['label']
            action_prob = strategy[action]

            # Update reach probability for the acting player
            new_reach_probs = reach_probs[:]
            new_reach_probs[acting_player] *= action_prob

            action_value = self.cfr(action_dict['child'], new_reach_probs)
            action_values[action] = action_value
            node_value += action_prob * action_value

        # Update regrets and strategy for the acting player
        # Counterfactual reach probability (product of all players' reach probs except acting player)
        cf_reach = reach_probs[1 - acting_player]

        for action_dict in node.actions:
            action = action_dict['label']
            regret = action_values[action] - node_value

            # CFR+: update regrets
            self.regret_sum[(infoset, action)] += cf_reach * regret

            # Accumulate strategy (weighted by own reach prob)
            self.strategy_sum[(infoset, action)] += reach_probs[acting_player] * strategy[action]

        return node_value

    def train(self, iterations: int):
        """Run CFR for a number of iterations."""
        for i in range(iterations):
            if (i + 1) % 5000 == 0:
                print(f"Iteration {i + 1}/{iterations}", file=sys.stderr)

            # Single tree traversal updates both players
            self.cfr(self.root_id, [1.0, 1.0])

            # CFR+: Floor regrets at 0 after each iteration
            for key in self.regret_sum:
                self.regret_sum[key] = max(0.0, self.regret_sum[key])

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get the average strategy across all iterations."""
        avg_strategy = {}

        for infoset, actions in self.infosets.items():
            strategy_sum = sum(self.strategy_sum[(infoset, a)] for a in actions)

            if strategy_sum > 0:
                avg_strategy[infoset] = {
                    a: self.strategy_sum[(infoset, a)] / strategy_sum
                    for a in actions
                }
            else:
                # Uniform if never reached
                uniform_prob = 1.0 / len(actions)
                avg_strategy[infoset] = {a: uniform_prob for a in actions}

        return avg_strategy

    def compute_exploitability(self) -> tuple:
        """Compute exploitability and game value."""
        avg_strategy = self.get_average_strategy()

        # Compute best response value for player 0 against player 1's strategy
        br_value_p0 = self.best_response_value(0, avg_strategy, self.root_id, 1.0)

        # Compute best response value for player 1 against player 0's strategy
        br_value_p1 = self.best_response_value(1, avg_strategy, self.root_id, 1.0)

        # Exploitability
        exploitability = (br_value_p0 + br_value_p1) / 2.0

        # Game value (expected payoff to p0 when both follow strategy)
        game_value = self.strategy_value(avg_strategy, self.root_id)

        return exploitability, game_value, br_value_p0, -br_value_p1

    def best_response_value(self, player: int, strategy: Dict[str, Dict[str, float]],
                           node_id: int, reach_prob: float) -> float:
        """Compute best response value for a player against opponent's strategy."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            payoff = node.payoff_p0 if player == 0 else -node.payoff_p0
            return payoff

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                value += outcome['prob'] * self.best_response_value(
                    player, strategy, outcome['child'], reach_prob
                )
            return value

        # Decision node
        if node.player == player:
            # Best response: take action with highest value
            best_value = float('-inf')
            for action_dict in node.actions:
                child_value = self.best_response_value(
                    player, strategy, action_dict['child'], reach_prob
                )
                best_value = max(best_value, child_value)
            return best_value
        else:
            # Opponent follows their strategy
            infoset = node.infoset
            value = 0.0
            for action_dict in node.actions:
                action = action_dict['label']
                action_prob = strategy[infoset][action]
                child_value = self.best_response_value(
                    player, strategy, action_dict['child'], reach_prob * action_prob
                )
                value += action_prob * child_value
            return value

    def strategy_value(self, strategy: Dict[str, Dict[str, float]], node_id: int) -> float:
        """Compute expected value when both players follow the strategy."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                value += outcome['prob'] * self.strategy_value(strategy, outcome['child'])
            return value

        # Decision node
        infoset = node.infoset
        value = 0.0
        for action_dict in node.actions:
            action = action_dict['label']
            action_prob = strategy[infoset][action]
            value += action_prob * self.strategy_value(strategy, action_dict['child'])
        return value


def main():
    # Load game
    print("Loading game...", file=sys.stderr)
    with open('game.json', 'r') as f:
        game_data = json.load(f)

    # Create solver
    print("Initializing CFR solver...", file=sys.stderr)
    solver = CFRSolver(game_data)

    print(f"Game: {solver.game_name}", file=sys.stderr)
    print(f"Nodes: {len(solver.nodes)}", file=sys.stderr)
    print(f"Information sets: {len(solver.infosets)}", file=sys.stderr)

    # Train
    iterations = 60000
    print(f"\nTraining for {iterations} iterations...", file=sys.stderr)
    solver.train(iterations)

    # Get average strategy
    avg_strategy = solver.get_average_strategy()

    # Compute exploitability
    print("\nComputing exploitability...", file=sys.stderr)
    exploitability, game_value, br_p0, br_p1 = solver.compute_exploitability()
    print(f"Exploitability: {exploitability:.6f}", file=sys.stderr)
    print(f"Game value (P0): {game_value:.6f}", file=sys.stderr)
    print(f"Best response value P0: {br_p0:.6f}", file=sys.stderr)
    print(f"Best response value P1 (negated): {br_p1:.6f}", file=sys.stderr)

    # Write strategy to file
    print("\nWriting strategy to strategy.json...", file=sys.stderr)
    strategy_output = {
        "format": "bluff-strategy-v1",
        "game": solver.game_name,
        "producer": f"CFR+ solver, {iterations} iterations",
        "probs": {}
    }

    for infoset, action_probs in avg_strategy.items():
        strategy_output["probs"][infoset] = {
            action: repr(prob) for action, prob in action_probs.items()
        }

    with open('strategy.json', 'w') as f:
        json.dump(strategy_output, f, indent=1)

    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    main()
