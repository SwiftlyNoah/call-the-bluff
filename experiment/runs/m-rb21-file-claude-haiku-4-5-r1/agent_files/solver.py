#!/usr/bin/env python3
"""CFR+ solver for mini Red/Black game."""

import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple


class GameTree:
    def __init__(self, game_data: Dict[str, Any]):
        self.game_data = game_data
        self.nodes = {node['id']: node for node in game_data['nodes']}
        self.root_id = game_data['root']
        self.name = game_data.get('name', 'game')

        # Build infoset mapping
        self.infosets = defaultdict(list)
        for node_id, node in self.nodes.items():
            if node['kind'] == 'decision':
                self.infosets[node['infoset']].append(node_id)


class CFRSolver:
    def __init__(self, game: GameTree):
        self.game = game
        self.infosets = game.infosets

        # Maps infoset -> action -> cumulative regret
        self.cumulative_regrets = defaultdict(lambda: defaultdict(float))

        # Maps infoset -> action -> cumulative strategy sum
        self.cumulative_strategy = defaultdict(lambda: defaultdict(float))

        # Current strategy
        self.strategy = defaultdict(lambda: defaultdict(float))
        self._initialize_strategy()

    def _initialize_strategy(self):
        """Initialize all strategies to uniform."""
        for infoset_name, node_ids in self.infosets.items():
            node = self.game.nodes[node_ids[0]]
            num_actions = len(node['actions'])
            uniform_prob = 1.0 / num_actions
            for action_idx in range(num_actions):
                self.strategy[infoset_name][action_idx] = uniform_prob

    def _prob_to_float(self, prob_str: str) -> float:
        """Convert probability string to float."""
        if '/' in prob_str:
            parts = prob_str.split('/')
            return float(parts[0]) / float(parts[1])
        elif '.' in prob_str:
            return float(prob_str)
        else:
            return float(prob_str)

    def _compute_value(self, node_id: int, player: int,
                      reach_p0: float, reach_p1: float) -> float:
        """Compute value of a node for a given player."""
        node = self.game.nodes[node_id]

        if node['kind'] == 'terminal':
            payoff = self._prob_to_float(node['payoff_p0'])
            return payoff if player == 0 else -payoff

        if node['kind'] == 'chance':
            total_value = 0.0
            for outcome in node['outcomes']:
                prob = self._prob_to_float(outcome['prob'])
                child_id = outcome['child']
                child_value = self._compute_value(child_id, player, reach_p0, reach_p1)
                total_value += prob * child_value
            return total_value

        # Decision node
        acting_player = node['player']
        infoset = node['infoset']
        strategy = self.strategy[infoset]

        node_value = 0.0
        for action_idx, action in enumerate(node['actions']):
            child_id = action['child']
            if acting_player == 0:
                child_value = self._compute_value(child_id, player, reach_p0 * strategy[action_idx], reach_p1)
            else:
                child_value = self._compute_value(child_id, player, reach_p0, reach_p1 * strategy[action_idx])
            node_value += strategy[action_idx] * child_value

        return node_value

    def _update_regrets(self, node_id: int, player: int,
                       reach_p0: float, reach_p1: float):
        """Traverse tree and update regrets for a given player."""
        node = self.game.nodes[node_id]

        if node['kind'] == 'terminal':
            return

        if node['kind'] == 'chance':
            for outcome in node['outcomes']:
                prob = self._prob_to_float(outcome['prob'])
                child_id = outcome['child']
                self._update_regrets(child_id, player, reach_p0, reach_p1)
            return

        # Decision node
        acting_player = node['player']
        infoset = node['infoset']
        actions = node['actions']
        strategy = self.strategy[infoset]

        if acting_player == player:
            my_reach = reach_p0 if player == 0 else reach_p1
            opp_reach = reach_p1 if player == 0 else reach_p0

            # Compute value for each action
            action_values = []
            for action in actions:
                child_id = action['child']
                child_value = self._compute_value(child_id, player, reach_p0, reach_p1)
                action_values.append(child_value)

            # Compute node value
            node_value = sum(strategy[i] * action_values[i] for i in range(len(actions)))

            # Update regrets (CFR+)
            for action_idx in range(len(actions)):
                regret = action_values[action_idx] - node_value
                self.cumulative_regrets[infoset][action_idx] += regret * opp_reach

            # Update cumulative strategy
            for action_idx in range(len(actions)):
                self.cumulative_strategy[infoset][action_idx] += strategy[action_idx] * my_reach

        # Recurse to children
        for action_idx, action in enumerate(actions):
            child_id = action['child']
            if acting_player == 0:
                new_reach_p0 = reach_p0 * strategy[action_idx]
                new_reach_p1 = reach_p1
            else:
                new_reach_p0 = reach_p0
                new_reach_p1 = reach_p1 * strategy[action_idx]
            self._update_regrets(child_id, player, new_reach_p0, new_reach_p1)

    def _update_strategy(self):
        """Update strategy based on cumulative regrets (CFR+)."""
        for infoset_name in self.infosets:
            node_id = self.infosets[infoset_name][0]
            node = self.game.nodes[node_id]
            num_actions = len(node['actions'])

            # Get positive regrets
            regrets = self.cumulative_regrets[infoset_name]
            positive_regrets = [max(regrets[i], 0.0) for i in range(num_actions)]

            # Normalize to get strategy
            total = sum(positive_regrets)
            if total > 0:
                for action_idx in range(num_actions):
                    self.strategy[infoset_name][action_idx] = positive_regrets[action_idx] / total
            else:
                # Uniform if no positive regrets
                uniform = 1.0 / num_actions
                for action_idx in range(num_actions):
                    self.strategy[infoset_name][action_idx] = uniform

    def run_iterations(self, num_iterations: int):
        """Run CFR for the specified number of iterations."""
        for iteration in range(num_iterations):
            if (iteration + 1) % 1000 == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}")

            # Update regrets for each player
            for player in range(2):
                self._update_regrets(self.game.root_id, player, 1.0, 1.0)

            # Update strategy
            self._update_strategy()

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy from cumulative strategy sums."""
        result = {}
        for infoset_name in self.infosets:
            node_id = self.infosets[infoset_name][0]
            node = self.game.nodes[node_id]
            actions = [action['label'] for action in node['actions']]

            # Normalize cumulative strategy
            cumsum = self.cumulative_strategy[infoset_name]
            total = sum(cumsum[i] for i in range(len(actions)))

            if total > 0:
                probs = {actions[i]: str(cumsum[i] / total) for i in range(len(actions))}
            else:
                # Fallback to uniform
                uniform = 1.0 / len(actions)
                probs = {actions[i]: str(uniform) for i in range(len(actions))}

            result[infoset_name] = probs

        return result


def main():
    # Load game
    with open('game.json', 'r') as f:
        game_data = json.load(f)

    game = GameTree(game_data)
    print(f"Loaded game: {game.name}")
    print(f"Number of nodes: {len(game.nodes)}")
    print(f"Number of infosets: {len(game.infosets)}")

    # Create solver and run
    solver = CFRSolver(game)
    print("Running CFR+...")
    solver.run_iterations(10000)

    # Get average strategy
    avg_strategy = solver.get_average_strategy()

    # Write strategy to file
    output = {
        "format": "bluff-strategy-v1",
        "game": game.name,
        "producer": "CFR+ solver",
        "probs": avg_strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("Strategy written to strategy.json")


if __name__ == '__main__':
    main()
