#!/usr/bin/env python3
"""
Counterfactual Regret Minimization (CFR+) solver for two-player zero-sum games.
"""

import json
import sys
from fractions import Fraction
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class GameNode:
    """Represents a node in the game tree."""
    def __init__(self, node_id: int, kind: str):
        self.id = node_id
        self.kind = kind


class ChanceNode(GameNode):
    def __init__(self, node_id: int, outcomes: List[Tuple[str, Fraction, int]]):
        super().__init__(node_id, "chance")
        self.outcomes = outcomes  # [(label, prob, child_id), ...]


class DecisionNode(GameNode):
    def __init__(self, node_id: int, player: int, infoset: str, actions: List[Tuple[str, int]]):
        super().__init__(node_id, "decision")
        self.player = player
        self.infoset = infoset
        self.actions = actions  # [(label, child_id), ...]


class TerminalNode(GameNode):
    def __init__(self, node_id: int, payoff_p0: Fraction):
        super().__init__(node_id, "terminal")
        self.payoff_p0 = payoff_p0


class Game:
    """Game tree representation."""
    def __init__(self, name: str, root_id: int, nodes: Dict[int, GameNode]):
        self.name = name
        self.root_id = root_id
        self.nodes = nodes

        # Build infoset action lists
        self.infosets: Dict[str, List[str]] = {}
        for node in nodes.values():
            if isinstance(node, DecisionNode):
                if node.infoset not in self.infosets:
                    self.infosets[node.infoset] = [action[0] for action in node.actions]


def parse_number(s: str) -> Fraction:
    """Parse a number string to Fraction."""
    if '/' in s:
        return Fraction(s)
    return Fraction(s)


def load_game(filename: str) -> Game:
    """Load game from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)

    name = data['name']
    root = data['root']
    nodes = {}

    for node_data in data['nodes']:
        node_id = node_data['id']
        kind = node_data['kind']

        if kind == 'chance':
            outcomes = [
                (out['label'], parse_number(out['prob']), out['child'])
                for out in node_data['outcomes']
            ]
            nodes[node_id] = ChanceNode(node_id, outcomes)
        elif kind == 'decision':
            player = node_data['player']
            infoset = node_data['infoset']
            actions = [(act['label'], act['child']) for act in node_data['actions']]
            nodes[node_id] = DecisionNode(node_id, player, infoset, actions)
        elif kind == 'terminal':
            payoff_p0 = parse_number(node_data['payoff_p0'])
            nodes[node_id] = TerminalNode(node_id, payoff_p0)

    return Game(name, root, nodes)


class CFRSolver:
    """CFR+ solver for two-player zero-sum games."""

    def __init__(self, game: Game):
        self.game = game

        # Regret and strategy tracking
        self.regret_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.strategy_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Initialize action lists for each infoset
        for infoset, actions in game.infosets.items():
            for action in actions:
                self.regret_sum[infoset][action] = 0.0
                self.strategy_sum[infoset][action] = 0.0

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy from regrets (regret matching)."""
        actions = self.game.infosets[infoset]

        # Use CFR+ approach: only positive regrets
        regrets = {a: max(0.0, self.regret_sum[infoset][a]) for a in actions}

        total_regret = sum(regrets.values())

        if total_regret > 0:
            strategy = {a: regrets[a] / total_regret for a in actions}
        else:
            # Uniform distribution if no positive regrets
            uniform_prob = 1.0 / len(actions)
            strategy = {a: uniform_prob for a in actions}

        return strategy

    def cfr(self, node_id: int, reach_p0: float, reach_p1: float) -> float:
        """
        Compute counterfactual value at a node.

        Args:
            node_id: Current node ID
            reach_p0: Reach probability for player 0
            reach_p1: Reach probability for player 1

        Returns:
            Expected payoff to player 0
        """
        node = self.game.nodes[node_id]

        if isinstance(node, TerminalNode):
            return float(node.payoff_p0)

        if isinstance(node, ChanceNode):
            value = 0.0
            for label, prob, child_id in node.outcomes:
                value += float(prob) * self.cfr(child_id, reach_p0, reach_p1)
            return value

        # Decision node
        assert isinstance(node, DecisionNode)
        infoset = node.infoset
        player = node.player
        strategy = self.get_strategy(infoset)

        # Compute value for each action
        action_values = {}
        node_value = 0.0

        for action_label, child_id in node.actions:
            action_prob = strategy[action_label]

            if player == 0:
                action_value = self.cfr(child_id, reach_p0 * action_prob, reach_p1)
            else:
                action_value = self.cfr(child_id, reach_p0, reach_p1 * action_prob)

            action_values[action_label] = action_value
            node_value += action_prob * action_value

        # Update regrets and strategy for the acting player
        if player == 0:
            reach_opponent = reach_p1
        else:
            reach_opponent = reach_p0

        for action_label in strategy.keys():
            regret = action_values[action_label] - node_value
            self.regret_sum[infoset][action_label] += reach_opponent * regret

        # Accumulate strategy (weighted by reach probability of acting player)
        reach_acting = reach_p0 if player == 0 else reach_p1
        for action_label, prob in strategy.items():
            self.strategy_sum[infoset][action_label] += reach_acting * prob

        return node_value

    def train(self, iterations: int):
        """Run CFR for specified iterations."""
        for i in range(iterations):
            self.cfr(self.game.root_id, 1.0, 1.0)

            # CFR+ specific: floor all negative regrets to 0
            if (i + 1) % 100 == 0:
                for infoset in self.regret_sum:
                    for action in self.regret_sum[infoset]:
                        if self.regret_sum[infoset][action] < 0:
                            self.regret_sum[infoset][action] = 0

            if (i + 1) % 5000 == 0:
                print(f"Iteration {i + 1}/{iterations}", file=sys.stderr)

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy across all iterations."""
        avg_strategy = {}

        for infoset, actions in self.game.infosets.items():
            total = sum(self.strategy_sum[infoset].values())

            if total > 0:
                avg_strategy[infoset] = {
                    action: repr(self.strategy_sum[infoset][action] / total)
                    for action in actions
                }
            else:
                # Uniform if never visited (shouldn't happen with proper game)
                uniform_prob = 1.0 / len(actions)
                avg_strategy[infoset] = {
                    action: repr(uniform_prob)
                    for action in actions
                }

        return avg_strategy

    def compute_exploitability(self) -> Tuple[float, float, float]:
        """
        Compute exploitability of average strategy.

        Returns:
            (exploitability, value_p0, best_response_sum)
        """
        avg_strat = self.get_average_strategy()

        # Convert to fixed strategy for evaluation
        fixed_strategy: Dict[str, Dict[str, float]] = {}
        for infoset, probs in avg_strat.items():
            fixed_strategy[infoset] = {a: float(p) for a, p in probs.items()}

        # Compute best response for each player
        br_value_p0 = self.best_response_value(0, fixed_strategy, self.game.root_id, 1.0)
        br_value_p1 = self.best_response_value(1, fixed_strategy, self.game.root_id, 1.0)

        exploitability = (br_value_p0 + br_value_p1) / 2

        # Compute value when both follow average strategy
        value_p0 = self.evaluate_strategy(fixed_strategy, self.game.root_id)

        return exploitability, value_p0, br_value_p0 + br_value_p1

    def best_response_value(self, player: int, opponent_strategy: Dict[str, Dict[str, float]],
                           node_id: int, reach_opp: float) -> float:
        """
        Compute best response value for a player against fixed opponent strategy.

        Args:
            player: Player computing best response (0 or 1)
            opponent_strategy: Fixed strategy for opponent
            node_id: Current node
            reach_opp: Opponent reach probability

        Returns:
            Best response value for player
        """
        node = self.game.nodes[node_id]

        if isinstance(node, TerminalNode):
            return float(node.payoff_p0) if player == 0 else -float(node.payoff_p0)

        if isinstance(node, ChanceNode):
            value = 0.0
            for label, prob, child_id in node.outcomes:
                value += float(prob) * self.best_response_value(player, opponent_strategy, child_id, reach_opp)
            return value

        assert isinstance(node, DecisionNode)

        if node.player == player:
            # Player's decision: take best action
            best_value = None
            for action_label, child_id in node.actions:
                action_value = self.best_response_value(player, opponent_strategy, child_id, reach_opp)
                if best_value is None or action_value > best_value:
                    best_value = action_value
            return best_value
        else:
            # Opponent's decision: they follow their strategy
            strategy = opponent_strategy[node.infoset]
            value = 0.0
            for action_label, child_id in node.actions:
                action_prob = strategy[action_label]
                value += action_prob * self.best_response_value(
                    player, opponent_strategy, child_id, reach_opp * action_prob
                )
            return value

    def evaluate_strategy(self, strategy: Dict[str, Dict[str, float]], node_id: int) -> float:
        """Evaluate expected payoff to player 0 when both players follow strategy."""
        node = self.game.nodes[node_id]

        if isinstance(node, TerminalNode):
            return float(node.payoff_p0)

        if isinstance(node, ChanceNode):
            value = 0.0
            for label, prob, child_id in node.outcomes:
                value += float(prob) * self.evaluate_strategy(strategy, child_id)
            return value

        assert isinstance(node, DecisionNode)
        strat = strategy[node.infoset]
        value = 0.0
        for action_label, child_id in node.actions:
            value += strat[action_label] * self.evaluate_strategy(strategy, child_id)
        return value


def save_strategy(filename: str, game_name: str, strategy: Dict[str, Dict[str, str]]):
    """Save strategy to JSON file."""
    output = {
        "format": "bluff-strategy-v1",
        "game": game_name,
        "producer": "CFR+ solver",
        "probs": strategy
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=1)

    print(f"Strategy saved to {filename}", file=sys.stderr)


def main():
    print("Loading game...", file=sys.stderr)
    game = load_game('game.json')
    print(f"Game '{game.name}' loaded: {len(game.nodes)} nodes, {len(game.infosets)} infosets",
          file=sys.stderr)

    print("Initializing CFR+ solver...", file=sys.stderr)
    solver = CFRSolver(game)

    # Run CFR
    iterations = 60000
    print(f"Running {iterations} iterations...", file=sys.stderr)
    solver.train(iterations)

    print("Computing exploitability...", file=sys.stderr)
    exploitability, value_p0, br_sum = solver.compute_exploitability()
    print(f"Exploitability: {exploitability:.6f}", file=sys.stderr)
    print(f"Value to P0: {value_p0:.6f}", file=sys.stderr)

    print("Extracting average strategy...", file=sys.stderr)
    avg_strategy = solver.get_average_strategy()

    print("Saving strategy...", file=sys.stderr)
    save_strategy('strategy.json', game.name, avg_strategy)

    print("Done!", file=sys.stderr)


if __name__ == '__main__':
    main()
