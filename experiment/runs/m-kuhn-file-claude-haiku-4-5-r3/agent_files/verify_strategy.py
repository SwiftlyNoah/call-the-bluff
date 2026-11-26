#!/usr/bin/env python3
"""Verify the computed strategy and check for Nash equilibrium."""

import json
from cfr_solver import parse_rational, CFRSolver


def compute_game_value(game_data, strategy_p0, strategy_p1):
    """Compute the value of the game given strategies for both players."""
    solver = CFRSolver(game_data)
    nodes = solver.nodes

    def traverse(node_id, p0_reach, p1_reach):
        node = nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_rational(node['payoff_p0']), p0_reach * p1_reach

        if node['kind'] == 'chance':
            total_value = 0.0
            total_prob = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                value, reach_prob = traverse(outcome['child'], p0_reach, p1_reach)
                total_value += prob * value
                total_prob += prob * reach_prob
            return total_value, total_prob

        player = node['player']
        infoset = node['infoset']

        if player == 0:
            strategy = strategy_p0
            new_reach = p0_reach
        else:
            strategy = strategy_p1
            new_reach = p1_reach

        actions = node['actions']
        value = 0.0
        reach = 0.0

        infoset_strategy = strategy.get(infoset, {})
        for action in actions:
            prob = infoset_strategy.get(action['label'], 1.0 / len(actions))
            if player == 0:
                child_value, child_reach = traverse(action['child'], new_reach * prob, p1_reach)
            else:
                child_value, child_reach = traverse(action['child'], p0_reach, new_reach * prob)

            value += prob * child_value
            reach += child_reach * prob

        return value, reach

    value, _ = traverse(solver.root_id, 1.0, 1.0)
    return value


def compute_br_value(game_data, opp_strategy, player):
    """Compute best-response value for a player against opponent strategy."""
    # For simplicity, we'll compute this via minimax search
    solver = CFRSolver(game_data)
    nodes = solver.nodes

    def br_traverse(node_id, p0_reach, p1_reach):
        node = nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_rational(node['payoff_p0'])

        if node['kind'] == 'chance':
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                value += prob * br_traverse(outcome['child'], p0_reach, p1_reach)
            return value

        acting_player = node['player']
        infoset = node['infoset']
        actions = node['actions']

        if acting_player == player:
            # Maximize for this player
            best_value = None
            for action in actions:
                if player == 0:
                    value = br_traverse(action['child'], p0_reach * 1.0, p1_reach)
                else:
                    value = br_traverse(action['child'], p0_reach, p1_reach * 1.0)

                if best_value is None or (player == 0 and value > best_value) or (player == 1 and -value > -best_value):
                    best_value = value
            return best_value
        else:
            # Play according to strategy
            strategy = opp_strategy
            value = 0.0
            infoset_strategy = strategy.get(infoset, {})
            for action in actions:
                prob = infoset_strategy.get(action['label'], 1.0 / len(actions))
                if player == 0:
                    child_value = br_traverse(action['child'], p0_reach, p1_reach * prob)
                else:
                    child_value = br_traverse(action['child'], p0_reach * prob, p1_reach)
                value += prob * child_value
            return value

    return br_traverse(solver.root_id, 1.0, 1.0)


def main():
    with open('game.json', 'r') as f:
        game = json.load(f)

    with open('strategy.json', 'r') as f:
        strategy_data = json.load(f)

    strategy = strategy_data['probs']

    # Convert strategy values to floats
    strategy_floats = {}
    for infoset, actions in strategy.items():
        strategy_floats[infoset] = {}
        for action, prob_str in actions.items():
            if 'e-' in prob_str or 'e+' in prob_str:
                # Scientific notation
                strategy_floats[infoset][action] = float(prob_str)
            else:
                try:
                    strategy_floats[infoset][action] = float(prob_str)
                except:
                    if '/' in prob_str:
                        parts = prob_str.split('/')
                        strategy_floats[infoset][action] = float(parts[0]) / float(parts[1])
                    else:
                        strategy_floats[infoset][action] = float(prob_str)

    # Compute game value with both players using this strategy
    game_value = compute_game_value(game, strategy_floats, strategy_floats)
    print(f"Game value (both players using strategy): {game_value}")

    # Compute best response values
    print("\nComputing best response values...")
    br_value_p0 = compute_br_value(game, strategy_floats, 0)
    br_value_p1 = compute_br_value(game, strategy_floats, 1)

    print(f"P0 best-response value against P1's strategy: {br_value_p0}")
    print(f"P1 best-response value against P0's strategy: {-br_value_p1}")

    # Exploitability
    exploitability = (br_value_p0 - game_value + ((-br_value_p1) - (-game_value))) / 2
    print(f"Exploitability: {exploitability}")

    print("\nStrategy summary:")
    for infoset in sorted(strategy_floats.keys()):
        print(f"  {infoset}: {strategy_floats[infoset]}")


if __name__ == '__main__':
    main()
