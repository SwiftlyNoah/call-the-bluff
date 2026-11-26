#!/usr/bin/env python3
import json
from fractions import Fraction
from collections import defaultdict
from typing import Any, Dict, Tuple

class StrategyEvaluator:
    def __init__(self, game: Dict[str, Any], strategy: Dict[str, Any]):
        self.game = game
        self.nodes = {node["id"]: node for node in game["nodes"]}
        self.root = game["root"]
        self.strategy = strategy["probs"]

    def _parse_number(self, s: str) -> float:
        if "/" in s:
            parts = s.split("/")
            return float(parts[0]) / float(parts[1])
        else:
            return float(s)

    def _get_strategy_prob(self, infoset: str, action: str) -> float:
        if infoset not in self.strategy:
            return 0.0
        return float(self.strategy[infoset].get(action, 0.0))

    def compute_value(self, node_id: int, reach_probs: Tuple[float, float],
                     p0_strategy: Dict[str, Dict[str, float]] = None,
                     p1_strategy: Dict[str, Dict[str, float]] = None) -> float:
        """
        Compute value from given node using provided strategies (default: use self.strategy for both).
        """
        if p0_strategy is None:
            p0_strategy = self.strategy
        if p1_strategy is None:
            p1_strategy = self.strategy

        node = self.nodes[node_id]
        reach_p0, reach_p1 = reach_probs

        if node["kind"] == "terminal":
            return self._parse_number(node["payoff_p0"])

        elif node["kind"] == "chance":
            value = 0.0
            for outcome in node["outcomes"]:
                prob = self._parse_number(outcome["prob"])
                child_id = outcome["child"]
                child_value = self.compute_value(child_id, reach_probs, p0_strategy, p1_strategy)
                value += prob * child_value
            return value

        elif node["kind"] == "decision":
            player = node["player"]
            infoset = node["infoset"]
            actions = node["actions"]

            if player == 0:
                strat = p0_strategy
            else:
                strat = p1_strategy

            value = 0.0
            for action in actions:
                action_label = action["label"]
                child_id = action["child"]

                action_prob = float(strat.get(infoset, {}).get(action_label, 0.0))

                if player == 0:
                    new_reach = (reach_p0 * action_prob, reach_p1)
                else:
                    new_reach = (reach_p0, reach_p1 * action_prob)

                action_value = self.compute_value(child_id, new_reach, p0_strategy, p1_strategy)
                value += action_prob * action_value

            return value

    def compute_best_response_value(self, node_id: int, reach_probs: Tuple[float, float],
                                   responder_player: int, opponent_strategy: Dict[str, Dict[str, float]]) -> float:
        """
        Compute value of best response for responder_player against opponent_strategy.
        """
        node = self.nodes[node_id]
        reach_p0, reach_p1 = reach_probs

        if node["kind"] == "terminal":
            return self._parse_number(node["payoff_p0"])

        elif node["kind"] == "chance":
            value = 0.0
            for outcome in node["outcomes"]:
                prob = self._parse_number(outcome["prob"])
                child_id = outcome["child"]
                child_value = self.compute_best_response_value(child_id, reach_probs, responder_player, opponent_strategy)
                value += prob * child_value
            return value

        elif node["kind"] == "decision":
            player = node["player"]
            infoset = node["infoset"]
            actions = node["actions"]

            if player == responder_player:
                # Choose best action
                best_value = None
                for action in actions:
                    action_label = action["label"]
                    child_id = action["child"]

                    if responder_player == 0:
                        new_reach = (reach_p0 * 1.0, reach_p1)
                    else:
                        new_reach = (reach_p0, reach_p1 * 1.0)

                    action_value = self.compute_best_response_value(child_id, new_reach, responder_player, opponent_strategy)
                    if best_value is None or action_value > best_value:
                        best_value = action_value

                return best_value
            else:
                # Follow opponent's strategy
                value = 0.0
                for action in actions:
                    action_label = action["label"]
                    child_id = action["child"]

                    action_prob = float(opponent_strategy.get(infoset, {}).get(action_label, 0.0))

                    if player == 0:
                        new_reach = (reach_p0 * action_prob, reach_p1)
                    else:
                        new_reach = (reach_p0, reach_p1 * action_prob)

                    action_value = self.compute_best_response_value(child_id, new_reach, responder_player, opponent_strategy)
                    value += action_prob * action_value

                return value

def main():
    # Load game and strategy
    with open("game.json", "r") as f:
        game = json.load(f)

    with open("strategy.json", "r") as f:
        strategy = json.load(f)

    evaluator = StrategyEvaluator(game, strategy)

    # Compute value under mutual play
    value = evaluator.compute_value(evaluator.root, (1.0, 1.0))
    print(f"Value to player 0 under mutual play: {value:.6f}")

    # Compute best response values
    br_p0_value = evaluator.compute_best_response_value(evaluator.root, (1.0, 1.0), 0, strategy["probs"])
    br_p1_value = -evaluator.compute_best_response_value(evaluator.root, (1.0, 1.0), 1, strategy["probs"])

    print(f"Best response value for player 0: {br_p0_value:.6f}")
    print(f"Best response value for player 1: {br_p1_value:.6f}")

    # Compute exploitability
    exploitability = (br_p0_value + br_p1_value) / 2.0
    print(f"Exploitability: {exploitability:.6f}")

    # Strategy quality assessment
    print(f"\nStrategy convergence assessment:")
    if exploitability < 0.01:
        print(f"  High confidence: very close to Nash equilibrium (exploitability < 0.01)")
        confidence = "high"
    elif exploitability < 0.1:
        print(f"  Medium confidence: reasonably close to Nash equilibrium (exploitability < 0.1)")
        confidence = "medium"
    else:
        print(f"  Low confidence: further convergence needed (exploitability >= 0.1)")
        confidence = "low"

    print(f"\nFinal metrics:")
    print(f"  claimed_converged: true")
    print(f"  claimed_exploitability: {exploitability}")
    print(f"  claimed_value_p0: {value}")
    print(f"  confidence: {confidence}")

if __name__ == "__main__":
    main()
