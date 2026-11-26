#!/usr/bin/env python3
import json
from fractions import Fraction
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import math

def parse_number(s: str) -> float:
    """Parse a number string (integer, fraction, or decimal) to float."""
    if '/' in s:
        parts = s.split('/')
        return float(parts[0]) / float(parts[1])
    else:
        return float(s)

class KuhnSolver:
    def __init__(self, game_file: str):
        with open(game_file) as f:
            self.game = json.load(f)

        self.nodes = {node['id']: node for node in self.game['nodes']}
        self.root_id = self.game['root']

        # Track regrets for each information set and action
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

        # Build infoset structure
        self.infosets = defaultdict(list)
        self.infoset_actions = {}

        for node_id, node in self.nodes.items():
            if node['kind'] == 'decision':
                infoset = node['infoset']
                self.infosets[infoset].append(node_id)
                if infoset not in self.infoset_actions:
                    self.infoset_actions[infoset] = [a['label'] for a in node['actions']]

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy for an infoset using regret matching."""
        actions = self.infoset_actions[infoset]
        regrets = self.regrets[infoset]

        # Regret matching: use positive regrets, normalized
        strategy = {}
        total = 0.0
        for action in actions:
            regret = max(0, regrets[action])
            strategy[action] = regret
            total += regret

        if total > 0:
            for action in actions:
                strategy[action] /= total
        else:
            # Uniform strategy
            prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = prob

        return strategy

    def cfr(self, node_id: int, reach_p0: float, reach_p1: float) -> float:
        """
        Counterfactual Regret Minimization.
        Returns the value of the game for player 0 from this node.
        """
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_number(node['payoff_p0'])

        if node['kind'] == 'chance':
            # Sum over all chance outcomes
            value = 0.0
            for outcome in node['outcomes']:
                prob = parse_number(outcome['prob'])
                child_id = outcome['child']
                value += prob * self.cfr(child_id, reach_p0, reach_p1)
            return value

        # Decision node
        player = node['player']
        infoset = node['infoset']
        actions = [a['label'] for a in node['actions']]
        action_children = {a['label']: a['child'] for a in node['actions']}

        # Get current strategy
        strategy = self.get_strategy(infoset)

        # Compute value of each action
        action_values = {}
        node_value = 0.0

        for action in actions:
            child_id = action_children[action]

            if player == 0:
                child_value = self.cfr(child_id, reach_p0 * strategy[action], reach_p1)
            else:
                child_value = self.cfr(child_id, reach_p0, reach_p1 * strategy[action])

            action_values[action] = child_value
            node_value += strategy[action] * child_value

        # Update regrets
        if player == 0:
            reach = reach_p1
        else:
            reach = reach_p0

        for action in actions:
            regret = (action_values[action] - node_value) * reach
            self.regrets[infoset][action] += regret
            self.strategy_sum[infoset][action] += strategy[action] * reach

        return node_value

    def solve(self, iterations: int = 10000):
        """Run CFR for the given number of iterations."""
        print(f"Running CFR for {iterations} iterations...")
        for i in range(iterations):
            value = self.cfr(self.root_id, 1.0, 1.0)
            if (i + 1) % 1000 == 0:
                print(f"  Iteration {i + 1}: value = {value}")
        print(f"Final game value for player 0: {value}")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy from cumulative strategy sums."""
        result = {}

        for infoset in sorted(self.infoset_actions.keys()):
            actions = self.infoset_actions[infoset]
            strategy = {}

            total = sum(self.strategy_sum[infoset][a] for a in actions)

            if total > 0:
                for action in actions:
                    prob = self.strategy_sum[infoset][action] / total
                    strategy[action] = str(prob)
            else:
                # Uniform fallback
                prob = 1.0 / len(actions)
                for action in actions:
                    strategy[action] = str(prob)

            result[infoset] = strategy

        return result

    def save_strategy(self, output_file: str):
        """Save the strategy to a JSON file."""
        strategy = self.get_average_strategy()

        output = {
            'format': 'bluff-strategy-v1',
            'game': self.game['name'],
            'producer': 'CFR (Counterfactual Regret Minimization) solver',
            'probs': strategy
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Strategy saved to {output_file}")

    def compute_game_value(self, strategy_dict: Dict[str, Dict[str, float]]) -> float:
        """Compute game value under a given strategy."""
        def evaluate(node_id: int) -> float:
            node = self.nodes[node_id]

            if node['kind'] == 'terminal':
                return parse_number(node['payoff_p0'])

            if node['kind'] == 'chance':
                value = 0.0
                for outcome in node['outcomes']:
                    prob = parse_number(outcome['prob'])
                    child_id = outcome['child']
                    value += prob * evaluate(child_id)
                return value

            infoset = node['infoset']
            actions = [a['label'] for a in node['actions']]
            action_children = {a['label']: a['child'] for a in node['actions']}
            strategy_here = strategy_dict[infoset]

            value = 0.0
            for action in actions:
                child_id = action_children[action]
                prob = strategy_here[action]
                value += prob * evaluate(child_id)

            return value

        return evaluate(self.root_id)

    def compute_best_response_value(self, opponent_strategy: Dict[str, Dict[str, float]], br_player: int) -> float:
        """Compute best-response value for a player against an opponent's strategy."""
        def evaluate_br(node_id: int) -> float:
            node = self.nodes[node_id]

            if node['kind'] == 'terminal':
                return parse_number(node['payoff_p0'])

            if node['kind'] == 'chance':
                value = 0.0
                for outcome in node['outcomes']:
                    prob = parse_number(outcome['prob'])
                    child_id = outcome['child']
                    value += prob * evaluate_br(child_id)
                return value

            node_player = node['player']
            infoset = node['infoset']
            actions = [a['label'] for a in node['actions']]
            action_children = {a['label']: a['child'] for a in node['actions']}

            if node_player == br_player:
                # This is the BR player's decision: pick best action
                if br_player == 0:
                    best_value = float('-inf')
                    for action in actions:
                        child_value = evaluate_br(action_children[action])
                        best_value = max(best_value, child_value)
                else:
                    best_value = float('inf')
                    for action in actions:
                        child_value = evaluate_br(action_children[action])
                        best_value = min(best_value, child_value)
                return best_value
            else:
                # Use opponent's strategy
                strategy_here = opponent_strategy[infoset]
                value = 0.0
                for action in actions:
                    child_id = action_children[action]
                    prob = strategy_here[action]
                    value += prob * evaluate_br(child_id)
                return value

        return evaluate_br(self.root_id)

    def compute_exploitability(self, strategy: Dict[str, Dict[str, str]]) -> Tuple[float, float]:
        """Compute exploitability and game value."""
        # Convert to float dict
        strat_float = {}
        for infoset, actions in strategy.items():
            strat_float[infoset] = {a: parse_number(p) for a, p in actions.items()}

        game_value = self.compute_game_value(strat_float)

        # For exploitability, compute best response values
        br_p0_value = self.compute_best_response_value(strat_float, 0)
        br_p1_value = self.compute_best_response_value(strat_float, 1)

        # In zero-sum: exploitability = how much players can gain with BR
        exploitability = (abs(br_p0_value - game_value) + abs(br_p1_value - (-game_value))) / 2

        return exploitability, game_value

def main():
    solver = KuhnSolver('game.json')
    solver.solve(iterations=10000)
    solver.save_strategy('strategy.json')

    # Compute value of the converged strategy
    strategy = solver.get_average_strategy()
    exploitability, game_value = solver.compute_exploitability(strategy)

    print(f"\nFinal Results:")
    print(f"  Game value for player 0: {game_value}")
    print(f"  Exploitability: {exploitability}")

if __name__ == '__main__':
    main()
