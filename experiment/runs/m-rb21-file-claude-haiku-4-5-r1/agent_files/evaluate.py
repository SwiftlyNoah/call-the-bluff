#!/usr/bin/env python3
"""Evaluate a CFR strategy."""

import json
from typing import Dict, Tuple


def prob_to_float(prob_str: str) -> float:
    if '/' in prob_str:
        parts = prob_str.split('/')
        return float(parts[0]) / float(parts[1])
    elif '.' in prob_str:
        return float(prob_str)
    else:
        return float(prob_str)


def compute_game_value(game_data: Dict, strategy_data: Dict) -> Tuple[float, float, float]:
    """
    Compute game value and exploitability under the given strategy.
    Returns (value_p0, br_p0, br_p1) where:
      - value_p0: value of the game when both players follow the strategy
      - br_p0: best response value for player 0 against player 1's strategy
      - br_p1: best response value for player 1 against player 0's strategy
    """
    nodes = {node['id']: node for node in game_data['nodes']}
    root_id = game_data['root']
    probs = strategy_data['probs']

    # Memoization for game value computation
    memo_value = {}
    memo_br = {}

    def compute_value_given_strategy(node_id: int, player: int,
                                     reach_p0: float, reach_p1: float) -> float:
        """Compute value under given strategy (not best response)."""
        key = (node_id, player, 'value')
        if key in memo_value:
            return memo_value[key]

        node = nodes[node_id]

        if node['kind'] == 'terminal':
            payoff = prob_to_float(node['payoff_p0'])
            result = payoff if player == 0 else -payoff
            memo_value[key] = result
            return result

        if node['kind'] == 'chance':
            total = 0.0
            for outcome in node['outcomes']:
                prob = prob_to_float(outcome['prob'])
                child_id = outcome['child']
                child_value = compute_value_given_strategy(child_id, player, reach_p0, reach_p1)
                total += prob * child_value
            memo_value[key] = total
            return total

        # Decision node
        acting_player = node['player']
        infoset = node['infoset']
        actions = node['actions']
        strategy = {i: prob_to_float(probs[infoset][actions[i]['label']]) for i in range(len(actions))}

        node_value = 0.0
        for action_idx, action in enumerate(actions):
            child_id = action['child']
            if acting_player == 0:
                child_value = compute_value_given_strategy(child_id, player, reach_p0 * strategy[action_idx], reach_p1)
            else:
                child_value = compute_value_given_strategy(child_id, player, reach_p0, reach_p1 * strategy[action_idx])
            node_value += strategy[action_idx] * child_value

        memo_value[key] = node_value
        return node_value

    def compute_br_value(node_id: int, br_player: int, opp_player: int,
                         reach_p0: float, reach_p1: float) -> float:
        """Compute best response value for br_player against opp_player's given strategy."""
        key = (node_id, br_player, 'br')
        if key in memo_br:
            return memo_br[key]

        node = nodes[node_id]

        if node['kind'] == 'terminal':
            payoff = prob_to_float(node['payoff_p0'])
            result = payoff if br_player == 0 else -payoff
            memo_br[key] = result
            return result

        if node['kind'] == 'chance':
            total = 0.0
            for outcome in node['outcomes']:
                prob = prob_to_float(outcome['prob'])
                child_id = outcome['child']
                child_value = compute_br_value(child_id, br_player, opp_player, reach_p0, reach_p1)
                total += prob * child_value
            memo_br[key] = total
            return total

        # Decision node
        acting_player = node['player']
        infoset = node['infoset']
        actions = node['actions']

        if acting_player == br_player:
            # This is our turn - pick best action
            best_value = None
            for action_idx, action in enumerate(actions):
                child_id = action['child']
                if br_player == 0:
                    child_value = compute_br_value(child_id, br_player, opp_player, reach_p0, reach_p1)
                else:
                    child_value = compute_br_value(child_id, br_player, opp_player, reach_p0, reach_p1)

                if best_value is None or child_value > best_value:
                    best_value = child_value
            memo_br[key] = best_value
            return best_value
        else:
            # Opponent's turn - follow their strategy
            opp_strategy = {i: prob_to_float(probs[infoset][actions[i]['label']]) for i in range(len(actions))}

            node_value = 0.0
            for action_idx, action in enumerate(actions):
                child_id = action['child']
                if acting_player == 0:
                    child_value = compute_br_value(child_id, br_player, opp_player, reach_p0 * opp_strategy[action_idx], reach_p1)
                else:
                    child_value = compute_br_value(child_id, br_player, opp_player, reach_p0, reach_p1 * opp_strategy[action_idx])
                node_value += opp_strategy[action_idx] * child_value

            memo_br[key] = node_value
            return node_value

    # Compute value when both follow strategy
    value_p0 = compute_value_given_strategy(root_id, 0, 1.0, 1.0)

    # Compute best response values
    br_p0 = compute_br_value(root_id, 0, 1, 1.0, 1.0)
    br_p1 = compute_br_value(root_id, 1, 0, 1.0, 1.0)

    return value_p0, br_p0, br_p1


def main():
    with open('game.json', 'r') as f:
        game_data = json.load(f)

    with open('strategy.json', 'r') as f:
        strategy_data = json.load(f)

    print("Evaluating strategy...")
    value_p0, br_p0, br_p1 = compute_game_value(game_data, strategy_data)

    print(f"Value to player 0 when both follow strategy: {value_p0:.6f}")
    print(f"Best response value for player 0: {br_p0:.6f}")
    print(f"Best response value for player 1: {-br_p1:.6f}")  # Player 1 payoff (negation)

    # Exploitability
    exploitability = (br_p0 - value_p0 + (-br_p1 - (-value_p0))) / 2
    print(f"Exploitability: {exploitability:.6f}")

    print(f"\nPlayer 0 can gain at most {br_p0 - value_p0:.6f} by deviating")
    print(f"Player 1 can gain at most {(-br_p1) - (-value_p0):.6f} by deviating")


if __name__ == '__main__':
    main()
