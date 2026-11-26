#!/usr/bin/env python3
import json
from fractions import Fraction
from collections import defaultdict
from typing import Any, Dict, List, Tuple

class CFRSolver:
    def __init__(self, game_path: str):
        with open(game_path) as f:
            self.game = json.load(f)

        self.nodes = {n['id']: n for n in self.game['nodes']}
        self.root_id = self.game['root']

        # Map information set to list of (player, node_id) pairs
        self.infosets = defaultdict(list)
        for node in self.game['nodes']:
            if node['kind'] == 'decision':
                self.infosets[node['infoset']].append((node['player'], node['id']))

        # Cumulative regrets: infoset_name -> action_index -> cumulative_regret (CFR+)
        self.regrets = defaultdict(lambda: defaultdict(float))
        # Strategy sum: infoset_name -> action_index -> sum of probabilities
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        # Cumulative positive regrets for CFR+
        self.positive_regrets = defaultdict(lambda: defaultdict(float))

    def parse_number(self, s: str) -> float:
        """Parse a rational number string to float."""
        if '/' in s:
            parts = s.split('/')
            return float(parts[0]) / float(parts[1])
        else:
            return float(s)

    def get_strategy(self, infoset: str) -> List[float]:
        """Get current strategy for an infoset using regret matching+ (CFR+)."""
        node = self.nodes[self.infosets[infoset][0][1]]
        num_actions = len(node['actions'])

        # Regret matching+: use only positive cumulative regrets
        regrets = [self.positive_regrets[infoset][a] for a in range(num_actions)]
        regret_sum = sum(regrets)

        if regret_sum > 0:
            return [r / regret_sum for r in regrets]
        else:
            # Uniform strategy
            return [1.0 / num_actions] * num_actions

    def cfr_recursive(self, node_id: int, reach_probs: List[float]) -> Tuple[float, float]:
        """
        Counterfactual value: traverse game tree and accumulate regrets.
        reach_probs: [reach_prob_p0, reach_prob_p1]
        Returns: (value_for_p0, value_for_p1)
        """
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            payoff = self.parse_number(node['payoff_p0'])
            return (payoff, -payoff)

        if node['kind'] == 'chance':
            # Expected value over chance outcomes
            value_p0 = 0.0
            value_p1 = 0.0
            for outcome in node['outcomes']:
                prob = self.parse_number(outcome['prob'])
                v0, v1 = self.cfr_recursive(outcome['child'], reach_probs)
                value_p0 += prob * v0
                value_p1 += prob * v1
            return (value_p0, value_p1)

        # Decision node
        acting_player = node['player']
        infoset = node['infoset']
        num_actions = len(node['actions'])

        # Get current strategy
        strategy = self.get_strategy(infoset)

        # Compute value of each action and node value
        action_values_p0 = []
        action_values_p1 = []

        for a, action in enumerate(node['actions']):
            # Update reach probabilities for the child
            child_reach_probs = reach_probs[:]
            child_reach_probs[acting_player] *= strategy[a]

            v0, v1 = self.cfr_recursive(action['child'], child_reach_probs)
            action_values_p0.append(v0)
            action_values_p1.append(v1)

        # Compute node values as weighted average
        node_value_p0 = sum(strategy[a] * action_values_p0[a] for a in range(num_actions))
        node_value_p1 = sum(strategy[a] * action_values_p1[a] for a in range(num_actions))

        # Accumulate regrets and strategy sum for the acting player
        opp_reach_prob = reach_probs[1 - acting_player]
        acting_reach_prob = reach_probs[acting_player]

        if acting_player == 0:
            # Update regrets for player 0 (CFR+)
            for a in range(num_actions):
                regret = action_values_p0[a] - node_value_p0
                weighted_regret = opp_reach_prob * regret
                self.regrets[infoset][a] += weighted_regret
                self.positive_regrets[infoset][a] += max(0.0, weighted_regret)
            # Accumulate strategy
            for a in range(num_actions):
                self.strategy_sum[infoset][a] += acting_reach_prob * strategy[a]
        else:
            # Update regrets for player 1 (CFR+)
            for a in range(num_actions):
                regret = action_values_p1[a] - node_value_p1
                weighted_regret = opp_reach_prob * regret
                self.regrets[infoset][a] += weighted_regret
                self.positive_regrets[infoset][a] += max(0.0, weighted_regret)
            # Accumulate strategy
            for a in range(num_actions):
                self.strategy_sum[infoset][a] += acting_reach_prob * strategy[a]

        return (node_value_p0, node_value_p1)

    def run(self, num_iterations: int = 1000):
        """Run CFR for num_iterations iterations."""
        for iteration in range(num_iterations):
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}")

            # Run one iteration of CFR
            reach_probs = [1.0, 1.0]
            value_p0, value_p1 = self.cfr_recursive(self.root_id, reach_probs)

            if (iteration + 1) % 100 == 0:
                print(f"  Value for player 0: {value_p0:.6f}, player 1: {value_p1:.6f}")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get average strategy normalized from strategy sum."""
        result = {}

        for infoset in sorted(self.infosets.keys()):
            node = self.nodes[self.infosets[infoset][0][1]]
            num_actions = len(node['actions'])

            strategy_weights = [self.strategy_sum[infoset][a] for a in range(num_actions)]
            total = sum(strategy_weights)

            if total > 0:
                probs = [w / total for w in strategy_weights]
            else:
                probs = [1.0 / num_actions] * num_actions

            action_names = [action['label'] for action in node['actions']]
            result[infoset] = {}

            for action_name, prob in zip(action_names, probs):
                result[infoset][action_name] = repr(prob)

        return result

    def compute_game_value(self) -> Tuple[float, float]:
        """Compute value of the game with average strategy."""
        return self._compute_value_with_avg_strategy(self.root_id, [1.0, 1.0])

    def _compute_value_with_avg_strategy(self, node_id: int, reach_probs: List[float]) -> Tuple[float, float]:
        """Compute value given average strategy."""
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            payoff = self.parse_number(node['payoff_p0'])
            return (payoff, -payoff)

        if node['kind'] == 'chance':
            value_p0 = 0.0
            value_p1 = 0.0
            for outcome in node['outcomes']:
                prob = self.parse_number(outcome['prob'])
                v0, v1 = self._compute_value_with_avg_strategy(outcome['child'], reach_probs)
                value_p0 += prob * v0
                value_p1 += prob * v1
            return (value_p0, value_p1)

        # Decision node - use average strategy
        infoset = node['infoset']
        num_actions = len(node['actions'])

        # Get average strategy
        strategy_weights = [self.strategy_sum[infoset][a] for a in range(num_actions)]
        total = sum(strategy_weights)
        if total > 0:
            strategy = [w / total for w in strategy_weights]
        else:
            strategy = [1.0 / num_actions] * num_actions

        value_p0 = 0.0
        value_p1 = 0.0
        for a, action in enumerate(node['actions']):
            child_reach_probs = reach_probs[:]
            child_reach_probs[node['player']] *= strategy[a]

            v0, v1 = self._compute_value_with_avg_strategy(action['child'], child_reach_probs)
            value_p0 += strategy[a] * v0
            value_p1 += strategy[a] * v1

        return (value_p0, value_p1)

def main():
    game_path = '/private/var/folders/km/r10tsfhx0815z44_014w924r0000gn/T/cfr-844heclc/workspace/game.json'
    solver = CFRSolver(game_path)

    print("Running CFR+ solver...")
    solver.run(num_iterations=5000)

    print("\nComputing average strategy...")
    avg_strategy = solver.get_average_strategy()

    print("Computing game value...")
    value_p0, value_p1 = solver.compute_game_value()
    print(f"Game value for player 0: {value_p0:.6f}")
    print(f"Game value for player 1: {value_p1:.6f}")

    with open(game_path) as f:
        game = json.load(f)

    output = {
        "format": "bluff-strategy-v1",
        "game": game['name'],
        "producer": "CFR solver",
        "probs": avg_strategy
    }

    output_path = '/private/var/folders/km/r10tsfhx0815z44_014w924r0000gn/T/cfr-844heclc/workspace/strategy.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nStrategy written to {output_path}")
    print(f"Found {len(avg_strategy)} information sets")

if __name__ == '__main__':
    main()
