#!/usr/bin/env python3
import json
from fractions import Fraction
from collections import defaultdict
from typing import Any, Dict, List, Tuple

class CFRSolver:
    def __init__(self, game: Dict[str, Any]):
        self.game = game
        self.nodes = {node["id"]: node for node in game["nodes"]}
        self.root = game["root"]

        self.infoset_actions = {}  # infoset -> list of action labels
        self.infoset_player = {}   # infoset -> player id

        # CFR+ data structures
        self.cumulative_regret = defaultdict(lambda: defaultdict(float))  # infoset -> action -> regret
        self.cumulative_strategy = defaultdict(lambda: defaultdict(float))  # infoset -> action -> strategy_sum

        self._preprocess_infosets()

    def _preprocess_infosets(self):
        """Extract information sets and their actions."""
        for node_id, node in self.nodes.items():
            if node["kind"] == "decision":
                infoset = node["infoset"]
                player = node["player"]
                actions = [action["label"] for action in node["actions"]]

                if infoset not in self.infoset_actions:
                    self.infoset_actions[infoset] = actions
                    self.infoset_player[infoset] = player

    def _parse_number(self, s: str) -> float:
        """Parse a number (integer, fraction, or decimal) to float."""
        if "/" in s:
            parts = s.split("/")
            return float(parts[0]) / float(parts[1])
        else:
            return float(s)

    def _get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an infoset using regret matching."""
        actions = self.infoset_actions[infoset]
        regrets = self.cumulative_regret[infoset]

        # Regret matching: use max(regret, 0)
        positive_regrets = {action: max(regrets[action], 0.0) for action in actions}
        total = sum(positive_regrets.values())

        if total <= 0:
            # Uniform strategy if no positive regrets
            return {action: 1.0 / len(actions) for action in actions}
        else:
            return {action: positive_regrets[action] / total for action in actions}

    def _traverse(self, node_id: int, reach_probs: Tuple[float, float]) -> float:
        """
        Traverse the game tree and perform CFR updates.
        reach_probs: (reach_prob_p0, reach_prob_p1)
        Returns: expected value for player 0
        """
        node = self.nodes[node_id]
        reach_p0, reach_p1 = reach_probs

        if node["kind"] == "terminal":
            payoff = self._parse_number(node["payoff_p0"])
            return payoff

        elif node["kind"] == "chance":
            value = 0.0
            for outcome in node["outcomes"]:
                prob = self._parse_number(outcome["prob"])
                child_id = outcome["child"]
                child_value = self._traverse(child_id, reach_probs)
                value += prob * child_value
            return value

        elif node["kind"] == "decision":
            player = node["player"]
            infoset = node["infoset"]
            actions = node["actions"]

            strategy = self._get_strategy(infoset)

            # Compute value for each action
            action_values = []
            node_value = 0.0

            for i, action in enumerate(actions):
                action_label = action["label"]
                child_id = action["child"]
                action_prob = strategy[action_label]

                if player == 0:
                    new_reach = (reach_p0 * action_prob, reach_p1)
                else:
                    new_reach = (reach_p0, reach_p1 * action_prob)

                action_value = self._traverse(child_id, new_reach)
                action_values.append(action_value)
                node_value += action_prob * action_value

            # Compute counterfactual values and regrets
            if player == 0:
                cf_reach = reach_p1
            else:
                cf_reach = reach_p0

            for i, action in enumerate(actions):
                action_label = action["label"]
                action_value = action_values[i]
                regret = (action_value - node_value) * cf_reach
                self.cumulative_regret[infoset][action_label] += regret

            # Update cumulative strategy for averaging
            for action in actions:
                action_label = action["label"]
                action_prob = strategy[action_label]
                if player == 0:
                    weight = reach_p0
                else:
                    weight = reach_p1
                self.cumulative_strategy[infoset][action_label] += weight * action_prob

            return node_value

    def solve(self, iterations: int = 1000):
        """Run CFR+ for the specified number of iterations."""
        for t in range(iterations):
            self._traverse(self.root, (1.0, 1.0))
            if (t + 1) % 100 == 0:
                print(f"Iteration {t + 1}/{iterations}")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy as strings."""
        result = {}
        for infoset in sorted(self.infoset_actions.keys()):
            actions = self.infoset_actions[infoset]
            strategy_sum = self.cumulative_strategy[infoset]
            total = sum(strategy_sum[action] for action in actions)

            if total <= 0:
                # Uniform strategy if no visits
                probs = {action: 1.0 / len(actions) for action in actions}
            else:
                probs = {action: strategy_sum[action] / total for action in actions}

            result[infoset] = {action: repr(probs[action]) for action in actions}

        return result

def main():
    # Read game
    with open("game.json", "r") as f:
        game = json.load(f)

    print(f"Loaded game: {game['name']}")
    print(f"Number of nodes: {len(game['nodes'])}")

    # Create solver
    solver = CFRSolver(game)
    print(f"Number of information sets: {len(solver.infoset_actions)}")

    # Solve
    print("Running CFR+...")
    solver.solve(iterations=1000)

    # Get average strategy
    avg_strategy = solver.get_average_strategy()

    # Write output
    output = {
        "format": "bluff-strategy-v1",
        "game": game["name"],
        "producer": "CFR+ solver",
        "probs": avg_strategy
    }

    with open("strategy.json", "w") as f:
        json.dump(output, f, indent=2)

    print("Strategy written to strategy.json")

if __name__ == "__main__":
    main()
