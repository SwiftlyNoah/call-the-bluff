#!/usr/bin/env python3
"""
CFR+ solver for two-player zero-sum games in extensive form.
Reads game.json and writes strategy.json.
"""

import json
from fractions import Fraction
from collections import defaultdict
import sys


def parse_rational(s):
    """Parse a rational number from string (integer, fraction, or decimal)."""
    if '/' in s:
        return float(Fraction(s))
    return float(s)


class GameTree:
    """Representation of the game tree."""

    def __init__(self, game_data):
        self.name = game_data['name']
        self.root = game_data['root']
        self.nodes = {node['id']: node for node in game_data['nodes']}

        # Collect all information sets and their actions
        self.infosets = {}  # infoset -> (player, [actions])
        for node in game_data['nodes']:
            if node['kind'] == 'decision':
                infoset = node['infoset']
                if infoset not in self.infosets:
                    actions = [a['label'] for a in node['actions']]
                    self.infosets[infoset] = (node['player'], actions)


class CFRSolver:
    """CFR+ solver for computing Nash equilibrium strategies."""

    def __init__(self, game):
        self.game = game

        # Strategy data structures
        self.regret_sum = defaultdict(lambda: defaultdict(float))  # infoset -> action -> regret
        self.strategy_sum = defaultdict(lambda: defaultdict(float))  # infoset -> action -> cumulative prob

        # Current iteration for CFR+ (use t=1 initially)
        self.iteration = 0

    def get_strategy(self, infoset):
        """Get current strategy for an information set using regret matching."""
        actions = self.game.infosets[infoset][1]

        # Regret matching: use max(0, regret) for each action
        regrets = self.regret_sum[infoset]
        positive_regrets = {a: max(0, regrets[a]) for a in actions}

        total = sum(positive_regrets.values())

        if total > 0:
            strategy = {a: positive_regrets[a] / total for a in actions}
        else:
            # Uniform strategy if no positive regrets
            uniform = 1.0 / len(actions)
            strategy = {a: uniform for a in actions}

        return strategy

    def cfr(self, node_id, reach_probs):
        """
        Counterfactual regret minimization recursive traversal.

        Args:
            node_id: Current node ID
            reach_probs: [reach_prob_p0, reach_prob_p1] - probability of reaching this node

        Returns:
            Expected value for player 0
        """
        node = self.game.nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_rational(node['payoff_p0'])

        if node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                value += prob * self.cfr(outcome['child'], reach_probs)
            return value

        # Decision node
        assert node['kind'] == 'decision'

        player = node['player']
        infoset = node['infoset']
        actions = [a['label'] for a in node['actions']]

        # Get current strategy
        strategy = self.get_strategy(infoset)

        # Compute value of each action
        action_values = {}
        for i, action_data in enumerate(node['actions']):
            action = action_data['label']

            # Update reach probabilities for this action
            new_reach = reach_probs.copy()
            new_reach[player] *= strategy[action]

            action_values[action] = self.cfr(action_data['child'], new_reach)

        # Node value is expected value over actions
        node_value = sum(strategy[action] * action_values[action] for action in actions)

        # Update regrets and strategy sum
        opponent = 1 - player
        cfr_reach = reach_probs[opponent]  # Counterfactual reach (opponent's reach)

        for action in actions:
            regret = action_values[action] - node_value
            self.regret_sum[infoset][action] += cfr_reach * regret

        # Add to strategy sum (weighted by player's reach)
        player_reach = reach_probs[player]
        for action in actions:
            self.strategy_sum[infoset][action] += player_reach * strategy[action]

        return node_value

    def train(self, iterations):
        """Run CFR for a number of iterations."""
        for i in range(iterations):
            self.iteration = i + 1

            # Traverse tree with both players reaching root with probability 1
            self.cfr(self.game.root, [1.0, 1.0])

            # CFR+: Floor negative regrets at 0
            for infoset in self.regret_sum:
                for action in self.regret_sum[infoset]:
                    if self.regret_sum[infoset][action] < 0:
                        self.regret_sum[infoset][action] = 0

            if (i + 1) % 1000 == 0:
                print(f"Completed iteration {i + 1}", file=sys.stderr)

    def get_average_strategy(self):
        """Get the average strategy over all iterations."""
        avg_strategy = {}

        for infoset, (player, actions) in self.game.infosets.items():
            strategy_sum = self.strategy_sum[infoset]
            total = sum(strategy_sum.values())

            if total > 0:
                avg_strategy[infoset] = {
                    action: str(strategy_sum[action] / total)
                    for action in actions
                }
            else:
                # If never reached, use uniform
                uniform = 1.0 / len(actions)
                avg_strategy[infoset] = {action: str(uniform) for action in actions}

        return avg_strategy


def main():
    # Load game
    print("Loading game...", file=sys.stderr)
    with open('game.json', 'r') as f:
        game_data = json.load(f)

    game = GameTree(game_data)
    print(f"Game: {game.name}", file=sys.stderr)
    print(f"Nodes: {len(game.nodes)}", file=sys.stderr)
    print(f"Information sets: {len(game.infosets)}", file=sys.stderr)

    # Create solver and train
    solver = CFRSolver(game)

    print("Running CFR+...", file=sys.stderr)
    iterations = 50000
    solver.train(iterations)

    # Get average strategy
    avg_strategy = solver.get_average_strategy()

    # Write strategy file
    print("Writing strategy...", file=sys.stderr)
    output = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": f"CFR+ solver ({iterations} iterations)",
        "probs": avg_strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=1)

    print(f"Strategy written to strategy.json", file=sys.stderr)


if __name__ == '__main__':
    main()
