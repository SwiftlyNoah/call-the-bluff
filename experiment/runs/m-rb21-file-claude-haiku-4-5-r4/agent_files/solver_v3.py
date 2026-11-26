#!/usr/bin/env python3
"""CFR+ solver for Red/Black game with better exploitability analysis."""

import json
from fractions import Fraction
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Callable


class GameNode:
    def __init__(self, node_dict):
        self.id = node_dict['id']
        self.kind = node_dict['kind']

        if self.kind == 'chance':
            self.outcomes = node_dict['outcomes']
        elif self.kind == 'decision':
            self.player = node_dict['player']
            self.infoset = node_dict['infoset']
            self.actions = node_dict['actions']
        elif self.kind == 'terminal':
            self.payoff_p0 = Fraction(node_dict['payoff_p0'])


def parse_number(s):
    """Parse a number string as a Fraction."""
    if isinstance(s, str):
        if '/' in s:
            parts = s.split('/')
            return Fraction(int(parts[0]), int(parts[1]))
        elif '.' in s:
            return Fraction(s).limit_denominator()
        else:
            return Fraction(int(s))
    return Fraction(s)


class CFRPlusSolver:
    def __init__(self, game_dict):
        self.game_dict = game_dict
        self.name = game_dict['name']
        self.root_id = game_dict['root']

        # Parse all nodes
        self.nodes = {}
        for node_dict in game_dict['nodes']:
            node = GameNode(node_dict)
            self.nodes[node.id] = node

        # Initialize strategy tracking with CFR+
        self.cumulative_regrets = defaultdict(lambda: defaultdict(float))
        self.cumulative_strategy = defaultdict(lambda: defaultdict(float))

        # Store action labels for each infoset
        self.infoset_actions = {}
        for node in self.nodes.values():
            if node.kind == 'decision':
                if node.infoset not in self.infoset_actions:
                    self.infoset_actions[node.infoset] = [a['label'] for a in node.actions]

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an infoset using regret matching+."""
        actions = self.infoset_actions[infoset]
        regrets = self.cumulative_regrets[infoset]

        # Regret matching+: use max(0, regret)
        strategy_sum = 0.0
        strategy = {}
        for action in actions:
            regret = max(0.0, regrets.get(action, 0.0))
            strategy[action] = regret
            strategy_sum += regret

        # Normalize
        if strategy_sum > 0:
            for action in actions:
                strategy[action] /= strategy_sum
        else:
            # Uniform strategy if no positive regrets
            for action in actions:
                strategy[action] = 1.0 / len(actions)

        return strategy

    def cfr_recursive(self, node_id: int, reach_probs: Tuple[float, float]) -> float:
        """
        Recursively compute the value of the game from this node using current strategies.
        reach_probs is (reach_prob_p0, reach_prob_p1).
        Returns the value for player 0.
        """
        node = self.nodes[node_id]
        reach_p0, reach_p1 = reach_probs

        if node.kind == 'terminal':
            return float(node.payoff_p0)

        if node.kind == 'chance':
            value = 0.0
            for outcome in node.outcomes:
                prob = float(parse_number(outcome['prob']))
                child_id = outcome['child']
                value += prob * self.cfr_recursive(child_id, reach_probs)
            return value

        # Decision node
        player = node.player
        infoset = node.infoset
        actions = node.actions

        # Get current strategy
        strategy = self.get_strategy(infoset)

        # Compute value of each action
        action_values = {}
        node_value = 0.0

        for action in actions:
            action_label = action['label']
            child_id = action['child']

            if player == 0:
                new_reach_p0 = reach_p0 * strategy[action_label]
                new_reach_p1 = reach_p1
            else:
                new_reach_p0 = reach_p0
                new_reach_p1 = reach_p1 * strategy[action_label]

            action_value = self.cfr_recursive(child_id, (new_reach_p0, new_reach_p1))
            action_values[action_label] = action_value
            node_value += strategy[action_label] * action_value

        # Accumulate regrets and strategy
        if player == 0:
            reach_opponent = reach_p1
        else:
            reach_opponent = reach_p0

        for action in actions:
            action_label = action['label']
            regret = reach_opponent * (action_values[action_label] - node_value)
            self.cumulative_regrets[infoset][action_label] += regret
            self.cumulative_strategy[infoset][action_label] += reach_opponent * strategy[action_label]

        return node_value

    def solve(self, iterations=20000):
        """Run CFR+ iterations."""
        print(f"Running CFR+ for {iterations} iterations...")
        for i in range(iterations):
            if (i + 1) % 2000 == 0:
                print(f"  Iteration {i + 1}/{iterations}")
            self.cfr_recursive(self.root_id, (1.0, 1.0))
        print("CFR+ complete.")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy across all iterations."""
        result = {}
        for infoset in sorted(self.infoset_actions.keys()):
            actions = self.infoset_actions[infoset]
            cumsum = sum(self.cumulative_strategy[infoset].get(action, 0.0) for action in actions)

            action_probs = {}
            if cumsum > 0:
                for action in actions:
                    prob = self.cumulative_strategy[infoset].get(action, 0.0) / cumsum
                    action_probs[action] = prob
            else:
                # Uniform if no data
                for action in actions:
                    action_probs[action] = 1.0 / len(actions)

            result[infoset] = action_probs

        return result

    def get_best_response_strategy(self, avg_strategy: Dict[str, Dict[str, float]],
                                   br_player: int) -> Dict[str, Dict[str, float]]:
        """
        Compute best response strategy for a player against a fixed strategy.
        Returns the best response strategy (which should be deterministic or greedy).
        """
        br_strategy = {}

        def br_recursive(node_id: int) -> float:
            node = self.nodes[node_id]

            if node.kind == 'terminal':
                payoff = float(node.payoff_p0)
                return payoff if br_player == 0 else -payoff

            if node.kind == 'chance':
                value = 0.0
                for outcome in node.outcomes:
                    prob = float(parse_number(outcome['prob']))
                    child_id = outcome['child']
                    value += prob * br_recursive(child_id)
                return value

            # Decision node
            player = node.player
            infoset = node.infoset
            actions = node.actions

            if player == br_player:
                # Best response: pick best action
                best_value = float('-inf')
                best_actions = []
                for action in actions:
                    action_label = action['label']
                    child_id = action['child']

                    action_value = br_recursive(child_id)
                    if action_value > best_value:
                        best_value = action_value
                        best_actions = [action_label]
                    elif action_value == best_value:
                        best_actions.append(action_label)

                # Create greedy strategy
                if infoset not in br_strategy:
                    num_best = len(best_actions)
                    for action in actions:
                        if action['label'] in best_actions:
                            br_strategy[infoset][action['label']] = 1.0 / num_best
                        else:
                            br_strategy[infoset][action['label']] = 0.0

                return best_value
            else:
                # Follow fixed strategy
                if infoset not in br_strategy:
                    br_strategy[infoset] = {}

                strategy = avg_strategy[infoset]
                node_value = 0.0
                for action in actions:
                    action_label = action['label']
                    child_id = action['child']

                    prob = strategy.get(action_label, 1.0 / len(actions))

                    action_value = br_recursive(child_id)
                    node_value += prob * action_value
                    if infoset not in br_strategy:
                        br_strategy[infoset] = {}
                    br_strategy[infoset][action_label] = prob
                return node_value

        br_recursive(self.root_id)
        return br_strategy

    def compute_strategy_value(self, strategy0: Dict, strategy1: Dict):
        """
        Compute the value when player 0 follows strategy0 and player 1 follows strategy1.
        """
        def value_recursive(node_id: int) -> float:
            node = self.nodes[node_id]

            if node.kind == 'terminal':
                return float(node.payoff_p0)

            if node.kind == 'chance':
                value = 0.0
                for outcome in node.outcomes:
                    prob = float(parse_number(outcome['prob']))
                    child_id = outcome['child']
                    value += prob * value_recursive(child_id)
                return value

            # Decision node
            player = node.player
            infoset = node.infoset
            actions = node.actions

            strategy = strategy0[infoset] if player == 0 else strategy1[infoset]
            node_value = 0.0

            for action in actions:
                action_label = action['label']
                child_id = action['child']
                prob = strategy.get(action_label, 1.0 / len(actions))

                action_value = value_recursive(child_id)
                node_value += prob * action_value

            return node_value

        return value_recursive(self.root_id)

    def compute_exploitability(self, avg_strategy: Dict[str, Dict[str, float]]) -> float:
        """
        Compute exploitability: (BR_value_p0 against strategy1 + BR_value_p1 against strategy0) / 2
        where BR_value is measured from player's perspective.
        """
        # Get best response strategies
        br_strategy_p0 = self.get_best_response_strategy(avg_strategy, 0)
        br_strategy_p1 = self.get_best_response_strategy(avg_strategy, 1)

        # Value when p0 plays BR against p1's avg strategy
        value_br_p0_vs_avg_p1 = self.compute_strategy_value(br_strategy_p0, avg_strategy)

        # Value when p1 plays BR against p0's avg strategy
        value_br_p1_vs_avg_p0 = self.compute_strategy_value(avg_strategy, br_strategy_p1)

        # Exploitability
        exploitability = (value_br_p0_vs_avg_p1 - self.compute_strategy_value(avg_strategy, avg_strategy)) / 2
        exploitability += (-value_br_p1_vs_avg_p0 + self.compute_strategy_value(avg_strategy, avg_strategy)) / 2

        return exploitability

    def save_strategy(self, filename):
        """Save the strategy to a JSON file."""
        avg_strategy = self.get_average_strategy()

        output = {
            "format": "bluff-strategy-v1",
            "game": self.name,
            "producer": "CFR+ solver",
            "probs": {}
        }

        for infoset in sorted(avg_strategy.keys()):
            probs = {}
            for action, prob in avg_strategy[infoset].items():
                # Format as Python float
                probs[action] = repr(prob)
            output["probs"][infoset] = probs

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Strategy saved to {filename}")


def main():
    # Load game
    with open('game.json', 'r') as f:
        game_dict = json.load(f)

    # Create and run solver
    solver = CFRPlusSolver(game_dict)
    solver.solve(iterations=20000)

    # Save strategy
    solver.save_strategy('strategy.json')

    # Analyze convergence
    avg_strategy = solver.get_average_strategy()
    value_avg = solver.compute_strategy_value(avg_strategy, avg_strategy)
    print(f"\nStrategy value for player 0 (both play avg): {value_avg}")

    # Compute exploitability
    print("Computing exploitability...")
    exploitability = solver.compute_exploitability(avg_strategy)
    print(f"Exploitability: {exploitability}")


if __name__ == '__main__':
    main()
