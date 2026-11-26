#!/usr/bin/env python3
"""
Evaluate a strategy: compute exploitability and expected value.
"""

import json
from fractions import Fraction
from typing import Dict, List


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


class StrategyEvaluator:
    """Evaluate strategy quality."""

    def __init__(self, game_file: str, strategy_file: str):
        """Load game and strategy."""
        with open(game_file, 'r') as f:
            game_data = json.load(f)

        self.root_id = game_data['root']
        self.nodes = {}
        for node_data in game_data['nodes']:
            self.nodes[node_data['id']] = GameNode(node_data)

        with open(strategy_file, 'r') as f:
            strategy_data = json.load(f)

        self.strategy = {}
        for infoset, probs in strategy_data['probs'].items():
            self.strategy[infoset] = {action: parse_rational(prob)
                                      for action, prob in probs.items()}

    def expected_value(self, node_id: int, player_strategies: List[Dict]) -> float:
        """
        Compute expected value for player 0 when both players follow given strategies.

        Args:
            node_id: Current node
            player_strategies: [strategy_p0, strategy_p1] where strategy is dict[infoset][action]
        """
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            return node.payoff_p0

        elif node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_rational(outcome['prob'])
                value += prob * self.expected_value(outcome['child'], player_strategies)
            return value

        elif node.kind == 'decision':
            player = node.player
            infoset = node.infoset
            strategy = player_strategies[player][infoset]

            value = 0.0
            for action_data in node.actions:
                action = action_data['label']
                child_id = action_data['child']
                prob = strategy.get(action, 0.0)
                value += prob * self.expected_value(child_id, player_strategies)

            return value

    def best_response_value(self, node_id: int, br_player: int, opponent_strategy: Dict) -> float:
        """
        Compute best response value for br_player against opponent's strategy.

        Args:
            node_id: Current node
            br_player: Player computing best response (0 or 1)
            opponent_strategy: Opponent's strategy dict[infoset][action]
        """
        node = self.nodes[node_id]

        if node.kind == 'terminal':
            payoff = node.payoff_p0
            return payoff if br_player == 0 else -payoff

        elif node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = parse_rational(outcome['prob'])
                value += prob * self.best_response_value(outcome['child'], br_player, opponent_strategy)
            return value

        elif node.kind == 'decision':
            player = node.player

            if player == br_player:
                # Best-responding player: take best action
                best_value = float('-inf')
                for action_data in node.actions:
                    child_id = action_data['child']
                    action_value = self.best_response_value(child_id, br_player, opponent_strategy)
                    best_value = max(best_value, action_value)
                return best_value
            else:
                # Opponent: follow their strategy
                infoset = node.infoset
                strategy = opponent_strategy[infoset]

                value = 0.0
                for action_data in node.actions:
                    action = action_data['label']
                    child_id = action_data['child']
                    prob = strategy.get(action, 0.0)
                    value += prob * self.best_response_value(child_id, br_player, opponent_strategy)
                return value

    def compute_exploitability(self) -> float:
        """
        Compute exploitability: average of best response values for both players.
        """
        # Best response value for player 0 against player 1's strategy
        br0_value = self.best_response_value(self.root_id, 0, self.strategy)

        # Best response value for player 1 against player 0's strategy
        br1_value = self.best_response_value(self.root_id, 1, self.strategy)

        # Exploitability is the average
        exploitability = (br0_value + br1_value) / 2.0

        return exploitability

    def compute_game_value(self) -> float:
        """
        Compute expected value for player 0 when both follow the strategy.
        """
        return self.expected_value(self.root_id, [self.strategy, self.strategy])

    def evaluate(self):
        """Run full evaluation."""
        print("Evaluating strategy...")

        game_value = self.compute_game_value()
        print(f"Game value (player 0): {game_value:.6f}")

        exploitability = self.compute_exploitability()
        print(f"Exploitability: {exploitability:.6f}")

        return {
            'game_value_p0': game_value,
            'exploitability': exploitability
        }


def main():
    """Main entry point."""
    evaluator = StrategyEvaluator('game.json', 'strategy.json')
    results = evaluator.evaluate()

    # Write results to file
    with open('evaluation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to evaluation.json")


if __name__ == '__main__':
    main()
