#!/usr/bin/env python3
import json
from fractions import Fraction

def parse_number(s):
    if '/' in str(s):
        parts = str(s).split('/')
        return float(parts[0]) / float(parts[1])
    return float(s)

def evaluate_best_response(node_id, opponent_strategy, nodes, is_p0_playing, reach_p0, reach_p1):
    """
    Calculate best response value.
    If is_p0_playing=True, calculate P0's best response against P1's strategy (and vice versa).
    """
    node = nodes[node_id]

    if node['kind'] == 'chance':
        value = 0
        for outcome in node['outcomes']:
            prob = parse_number(outcome['prob'])
            child_id = outcome['child']
            child_value = evaluate_best_response(child_id, opponent_strategy, nodes, is_p0_playing,
                                                  reach_p0 * prob, reach_p1 * prob)
            value += prob * child_value
        return value

    elif node['kind'] == 'decision':
        player = node['player']
        infoset = node['infoset']
        actions = node['actions']

        # If this is our turn, we play best response
        if (player == 0 and is_p0_playing) or (player == 1 and not is_p0_playing):
            # Find best action
            best_value = float('-inf')
            for i, action in enumerate(actions):
                child_id = action['child']
                if player == 0:
                    child_reach_p0 = reach_p0
                    child_reach_p1 = reach_p1
                else:
                    child_reach_p0 = reach_p0
                    child_reach_p1 = reach_p1

                child_value = evaluate_best_response(child_id, opponent_strategy, nodes, is_p0_playing,
                                                     child_reach_p0, child_reach_p1)
                if player == 0 and is_p0_playing:
                    # P0 wants to maximize
                    best_value = max(best_value, child_value)
                else:
                    # P1 wants to minimize
                    best_value = min(best_value, child_value)
            return best_value
        else:
            # Opponent plays their strategy
            if infoset in opponent_strategy:
                strat_dict = opponent_strategy[infoset]
                node_value = 0
                for i, action in enumerate(actions):
                    child_id = action['child']
                    prob = parse_number(strat_dict[action['label']])

                    if player == 0:
                        child_reach_p0 = reach_p0 * prob
                        child_reach_p1 = reach_p1
                    else:
                        child_reach_p0 = reach_p0
                        child_reach_p1 = reach_p1 * prob

                    child_value = evaluate_best_response(child_id, opponent_strategy, nodes, is_p0_playing,
                                                        child_reach_p0, child_reach_p1)
                    node_value += prob * child_value
                return node_value
            else:
                # Uniform strategy if not in strategy dict
                node_value = 0
                for i, action in enumerate(actions):
                    child_id = action['child']
                    prob = 1.0 / len(actions)

                    if player == 0:
                        child_reach_p0 = reach_p0 * prob
                        child_reach_p1 = reach_p1
                    else:
                        child_reach_p0 = reach_p0
                        child_reach_p1 = reach_p1 * prob

                    child_value = evaluate_best_response(child_id, opponent_strategy, nodes, is_p0_playing,
                                                        child_reach_p0, child_reach_p1)
                    node_value += prob * child_value
                return node_value

    elif node['kind'] == 'terminal':
        return parse_number(node['payoff_p0'])

def main():
    with open('game.json') as f:
        game_data = json.load(f)

    with open('strategy.json') as f:
        strategy_data = json.load(f)

    nodes = {node['id']: node for node in game_data['nodes']}
    root = game_data['root']

    # Convert strategy format
    opponent_strategy = {}
    for infoset_name, actions_dict in strategy_data['probs'].items():
        opponent_strategy[infoset_name] = actions_dict

    # Calculate best responses
    print("Calculating best responses...")

    # P0's best response value against P1's current strategy
    p0_br_value = evaluate_best_response(root, opponent_strategy, nodes, True, 1.0, 1.0)

    # P1's best response value against P0's current strategy (negation)
    p1_br_value = -evaluate_best_response(root, opponent_strategy, nodes, False, 1.0, 1.0)

    # Exploitability = (P0's BR value + P1's BR value) / 2
    exploitability = (p0_br_value + p1_br_value) / 2

    # Game value with current strategy
    # This requires evaluating with the actual strategy

    print(f"\nExploitability Analysis:")
    print(f"  P0's best response value: {p0_br_value:.8f}")
    print(f"  P1's best response value: {p1_br_value:.8f}")
    print(f"  Exploitability: {exploitability:.8f}")

if __name__ == '__main__':
    main()
