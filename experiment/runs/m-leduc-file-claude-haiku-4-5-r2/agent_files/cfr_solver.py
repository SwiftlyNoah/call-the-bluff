#!/usr/bin/env python3
import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple

class CFRSolver:
    def __init__(self, game_path: str):
        with open(game_path, 'r') as f:
            self.game = json.load(f)

        self.nodes = {node['id']: node for node in self.game['nodes']}
        self.root = self.game['root']
        self.game_name = self.game.get('name', 'unknown')

        # infoset -> {action: cumulative_regret}
        self.cumulative_regrets = defaultdict(lambda: defaultdict(float))

        # infoset -> {action: sum_of_strategy_weights}
        self.strategy_sums = defaultdict(lambda: defaultdict(float))

        # infoset -> list of actions
        self.infoset_actions = {}

        # node_id -> {action: child_id}
        self.node_actions = {}

        self._init_infosets()

    def _init_infosets(self):
        """Extract all infosets and their actions from the game tree."""
        for node in self.game['nodes']:
            if node['kind'] == 'decision':
                infoset = node['infoset']
                node_id = node['id']

                if infoset not in self.infoset_actions:
                    actions = [a['label'] for a in node['actions']]
                    self.infoset_actions[infoset] = actions

                # Cache action->child mapping per node
                self.node_actions[node_id] = {a['label']: a['child'] for a in node['actions']}

    def _prob_to_float(self, prob_str: str) -> float:
        """Convert probability string to float."""
        if isinstance(prob_str, (int, float)):
            return float(prob_str)
        prob_str = str(prob_str).strip()
        if '/' in prob_str:
            parts = prob_str.split('/')
            return float(parts[0]) / float(parts[1])
        elif '.' in prob_str:
            return float(prob_str)
        else:
            return float(int(prob_str))

    def cfr_iteration(self, iterations: int):
        """Run CFR for a given number of iterations."""
        for t in range(iterations):
            if (t + 1) % 100 == 0:
                print(f"  Iteration {t + 1}/{iterations}")
            self._cfr_visit(self.root, 1.0, 1.0, 1.0)

    def _cfr_visit(self, node_id: int, reach_p0: float, reach_p1: float,
                   reach_chance: float) -> float:
        """
        Recursively compute counterfactual values using CFR.
        Returns the value of the node for player 0.
        """
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            payoff = self._prob_to_float(node['payoff_p0'])
            return payoff

        elif node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = self._prob_to_float(outcome['prob'])
                child_value = self._cfr_visit(
                    outcome['child'],
                    reach_p0, reach_p1,
                    reach_chance * prob
                )
                value += prob * child_value
            return value

        elif node['kind'] == 'decision':
            player = node['player']
            infoset = node['infoset']
            actions = self.infoset_actions[infoset]

            # Get reach probabilities for this player
            if player == 0:
                reach = reach_p0
                opponent_reach = reach_p1
            else:
                reach = reach_p1
                opponent_reach = reach_p0

            # Get current strategy for this infoset
            strategy = self._get_strategy(infoset)

            # Compute value of each action
            action_values = []
            node_value = 0.0

            action_map = self.node_actions[node_id]

            for action in actions:
                child_id = action_map[action]
                action_prob = strategy[action]

                if player == 0:
                    child_value = self._cfr_visit(
                        child_id,
                        reach_p0 * action_prob,
                        reach_p1,
                        reach_chance
                    )
                else:
                    child_value = self._cfr_visit(
                        child_id,
                        reach_p0,
                        reach_p1 * action_prob,
                        reach_chance
                    )

                action_values.append(child_value)
                node_value += action_prob * child_value

            # Accumulate regrets and strategy
            cf_reach = opponent_reach * reach_chance

            for i, action in enumerate(actions):
                action_value = action_values[i]
                regret = action_value - node_value

                self.cumulative_regrets[infoset][action] += regret * cf_reach
                self.strategy_sums[infoset][action] += strategy[action] * reach

            return node_value

    def _get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy using regret matching."""
        actions = self.infoset_actions[infoset]

        strategy = {}
        total_positive_regret = 0.0

        for action in actions:
            regret = self.cumulative_regrets[infoset][action]
            positive_regret = max(regret, 0.0)
            total_positive_regret += positive_regret

        if total_positive_regret > 0:
            for action in actions:
                regret = self.cumulative_regrets[infoset][action]
                positive_regret = max(regret, 0.0)
                strategy[action] = positive_regret / total_positive_regret
        else:
            uniform = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform

        return strategy

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy from strategy sums."""
        avg_strategy = {}

        for infoset in self.infoset_actions:
            actions = self.infoset_actions[infoset]
            avg_strategy[infoset] = {}

            total = sum(self.strategy_sums[infoset][action] for action in actions)

            if total > 0:
                for action in actions:
                    prob = self.strategy_sums[infoset][action] / total
                    avg_strategy[infoset][action] = str(prob)
            else:
                uniform = 1.0 / len(actions)
                for action in actions:
                    avg_strategy[infoset][action] = str(uniform)

        return avg_strategy

    def solve(self, iterations: int = 1000):
        """Run CFR and return average strategy."""
        print(f"Running CFR for {iterations} iterations...")
        self.cfr_iteration(iterations)
        strategy = self.get_average_strategy()
        print(f"CFR complete. Found {len(strategy)} infosets.")
        return strategy

    def save_strategy(self, strategy: Dict[str, Dict[str, str]], output_path: str):
        """Save strategy to JSON file."""
        output = {
            "format": "bluff-strategy-v1",
            "game": self.game_name,
            "producer": "CFR+ solver",
            "probs": strategy
        }
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Strategy saved to {output_path}")


if __name__ == "__main__":
    import sys

    game_path = "game.json"
    output_path = "strategy.json"
    iterations = 1000

    solver = CFRSolver(game_path)
    strategy = solver.solve(iterations)
    solver.save_strategy(strategy, output_path)
