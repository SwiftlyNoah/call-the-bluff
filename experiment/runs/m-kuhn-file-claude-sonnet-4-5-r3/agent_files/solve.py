#!/usr/bin/env python3
"""
Counterfactual Regret Minimization (CFR+) solver for two-player zero-sum games.
Reads game.json and writes strategy.json.
"""

import json
from fractions import Fraction
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def parse_number(s: str) -> float:
    """Parse a number string (integer, fraction, or decimal) to float."""
    if '/' in s:
        return float(Fraction(s))
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
            self.payoff_p0 = parse_number(node_data['payoff_p0'])


class CFRPlusSolver:
    """Counterfactual Regret Minimization Plus solver."""

    def __init__(self, game_data: dict):
        self.game_name = game_data['name']
        self.nodes = {node['id']: GameNode(node) for node in game_data['nodes']}
        self.root_id = game_data['root']

        # CFR+ data structures (CFR+ floors regrets at 0)
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.infoset_actions = {}  # Map infoset to list of action labels

        # Collect all infosets and their actions
        for node in self.nodes.values():
            if node.kind == 'decision':
                if node.infoset not in self.infoset_actions:
                    self.infoset_actions[node.infoset] = [a['label'] for a in node.actions]

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an infoset using regret matching."""
        actions = self.infoset_actions[infoset]
        strategy = {}

        # CFR+: use max(regret, 0) for regret matching
        regrets = self.regret_sum[infoset]
        normalizing_sum = sum(max(regrets.get(a, 0), 0) for a in actions)

        if normalizing_sum > 0:
            for action in actions:
                strategy[action] = max(regrets.get(action, 0), 0) / normalizing_sum
        else:
            # Uniform strategy if all regrets are non-positive
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform_prob

        return strategy

    def cfr(self, node_id: int, reach_p0: float, reach_p1: float) -> float:
        """
        Run CFR recursively.
        reach_p0: probability that player 0's actions lead to this node
        reach_p1: probability that player 1's actions lead to this node
        Returns the expected value for player 0.
        """
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_number(outcome['prob'])
                # Chance probability is external; don't affect reach_p0 or reach_p1
                child_value = self.cfr(outcome['child'], reach_p0, reach_p1)
                value += prob * child_value
            return value

        # Decision node
        player = node.player
        infoset = node.infoset
        actions = [a['label'] for a in node.actions]

        # Get current strategy
        strategy = self.get_strategy(infoset)

        # Calculate value for each action
        action_values = {}
        node_value = 0.0

        for action_data in node.actions:
            action = action_data['label']
            action_prob = strategy[action]

            # Update reach probability for this player
            if player == 0:
                action_values[action] = self.cfr(action_data['child'],
                                                 reach_p0 * action_prob, reach_p1)
            else:
                action_values[action] = self.cfr(action_data['child'],
                                                 reach_p0, reach_p1 * action_prob)

            node_value += action_prob * action_values[action]

        # Update regrets and strategy sum
        # Counterfactual reach is the opponent's reach probability
        cf_reach = reach_p1 if player == 0 else reach_p0
        player_reach = reach_p0 if player == 0 else reach_p1

        for action in actions:
            # Values are from P0's perspective, so flip for P1
            if player == 1:
                regret = -(action_values[action] - node_value)
            else:
                regret = action_values[action] - node_value
            # CFR+: floor regrets at 0 immediately
            self.regret_sum[infoset][action] += cf_reach * regret
            self.regret_sum[infoset][action] = max(self.regret_sum[infoset][action], 0)

            # Update strategy sum
            self.strategy_sum[infoset][action] += player_reach * strategy[action]

        return node_value

    def train(self, iterations: int):
        """Run CFR for the specified number of iterations."""
        for i in range(iterations):
            # Both players start with reach probability 1
            self.cfr(self.root_id, 1.0, 1.0)

            if (i + 1) % 100000 == 0:
                print(f"Completed {i + 1} iterations")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy across all iterations."""
        avg_strategy = {}

        for infoset in self.infoset_actions:
            actions = self.infoset_actions[infoset]
            strategy = {}

            normalizing_sum = sum(self.strategy_sum[infoset].get(a, 0) for a in actions)

            if normalizing_sum > 0:
                for action in actions:
                    prob = self.strategy_sum[infoset].get(action, 0) / normalizing_sum
                    strategy[action] = repr(prob)
            else:
                # If we never reached this infoset, use uniform
                uniform_prob = 1.0 / len(actions)
                for action in actions:
                    strategy[action] = repr(uniform_prob)

            avg_strategy[infoset] = strategy

        return avg_strategy

    def calculate_exploitability(self) -> float:
        """Calculate exploitability of the current average strategy."""
        avg_strat = self.get_average_strategy()

        # Convert string probs back to floats for calculation
        strategy_floats = {}
        for infoset, probs in avg_strat.items():
            strategy_floats[infoset] = {a: float(p) for a, p in probs.items()}

        # Calculate best response value for each player
        br_value_p0 = self.best_response_value(0, strategy_floats)
        br_value_p1 = self.best_response_value(1, strategy_floats)

        # Exploitability: sum of how much each player can improve
        game_value_p0 = self.calculate_game_value()
        exploit_p0 = br_value_p0 - game_value_p0
        exploit_p1 = br_value_p1 + game_value_p0
        return exploit_p0 + exploit_p1

    def best_response_value(self, player: int, opponent_strategy: Dict[str, Dict[str, float]]) -> float:
        """Calculate the best-response value for a player against opponent's strategy."""
        return self._br_recursive(self.root_id, player, opponent_strategy, {})

    def _br_recursive(self, node_id: int, player: int, opponent_strategy: Dict[str, Dict[str, float]],
                      memo: Dict[int, float]) -> float:
        """Recursively calculate best response value."""
        if node_id in memo:
            return memo[node_id]

        node = self.nodes[node_id]

        if node.kind == 'terminal':
            value = node.payoff_p0 if player == 0 else -node.payoff_p0
            memo[node_id] = value
            return value

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_number(outcome['prob'])
                child_value = self._br_recursive(outcome['child'], player, opponent_strategy, memo)
                value += prob * child_value
            memo[node_id] = value
            return value

        # Decision node
        if node.player == player:
            # Maximizing player: take best action
            best_value = float('-inf')
            for action_data in node.actions:
                child_value = self._br_recursive(action_data['child'], player, opponent_strategy, memo)
                best_value = max(best_value, child_value)
            memo[node_id] = best_value
            return best_value
        else:
            # Opponent: use their strategy
            value = 0.0
            for action_data in node.actions:
                action = action_data['label']
                prob = opponent_strategy[node.infoset][action]
                child_value = self._br_recursive(action_data['child'], player, opponent_strategy, memo)
                value += prob * child_value
            memo[node_id] = value
            return value

    def calculate_game_value(self) -> float:
        """Calculate expected value for player 0 when both play average strategy."""
        avg_strat = self.get_average_strategy()
        strategy_floats = {}
        for infoset, probs in avg_strat.items():
            strategy_floats[infoset] = {a: float(p) for a, p in probs.items()}

        return self._game_value_recursive(self.root_id, strategy_floats)

    def _game_value_recursive(self, node_id: int, strategy: Dict[str, Dict[str, float]]) -> float:
        """Recursively calculate game value."""
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_number(outcome['prob'])
                child_value = self._game_value_recursive(outcome['child'], strategy)
                value += prob * child_value
            return value

        # Decision node: use strategy
        value = 0.0
        for action_data in node.actions:
            action = action_data['label']
            prob = strategy[node.infoset][action]
            child_value = self._game_value_recursive(action_data['child'], strategy)
            value += prob * child_value
        return value


def main():
    # Load game
    with open('game.json', 'r') as f:
        game_data = json.load(f)

    print(f"Loaded game: {game_data['name']}")

    # Create solver and train
    solver = CFRPlusSolver(game_data)

    print("Training with CFR+...")
    iterations = 1000000
    solver.train(iterations)

    # Get average strategy
    avg_strategy = solver.get_average_strategy()

    # Calculate metrics
    print("\nCalculating metrics...")
    exploitability = solver.calculate_exploitability()
    game_value = solver.calculate_game_value()

    print(f"Exploitability: {exploitability}")
    print(f"Game value (P0): {game_value}")

    # Write strategy to file
    output = {
        "format": "bluff-strategy-v1",
        "game": game_data['name'],
        "producer": f"CFR+ solver with {iterations} iterations",
        "probs": avg_strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=1)

    print("\nStrategy written to strategy.json")
    print(f"\nFinal exploitability: {exploitability}")
    print(f"Final game value (P0): {game_value}")


if __name__ == '__main__':
    main()
