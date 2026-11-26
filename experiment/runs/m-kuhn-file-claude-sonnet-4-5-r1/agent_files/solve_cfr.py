#!/usr/bin/env python3
"""
Counterfactual Regret Minimization (CFR) solver for two-player zero-sum games.
"""

import json
from collections import defaultdict
from typing import Dict, List


def parse_rational(s: str) -> float:
    """Parse a rational number string into a float."""
    if '/' in s:
        num, den = s.split('/')
        return float(num) / float(den)
    return float(s)


class InfoSetData:
    """Stores CFR data for one information set."""
    def __init__(self, actions: List[str]):
        self.actions = actions
        self.regret_sum = {a: 0.0 for a in actions}
        self.strategy_sum = {a: 0.0 for a in actions}

    def get_strategy(self) -> Dict[str, float]:
        """Get current strategy using regret matching."""
        normalizing_sum = sum(max(0.0, self.regret_sum[a]) for a in self.actions)

        if normalizing_sum > 0:
            return {a: max(0.0, self.regret_sum[a]) / normalizing_sum for a in self.actions}
        else:
            # Uniform if all regrets non-positive
            uniform_prob = 1.0 / len(self.actions)
            return {a: uniform_prob for a in self.actions}

    def get_average_strategy(self) -> Dict[str, float]:
        """Get average strategy over all iterations."""
        normalizing_sum = sum(self.strategy_sum.values())

        if normalizing_sum > 0:
            return {a: self.strategy_sum[a] / normalizing_sum for a in self.actions}
        else:
            uniform_prob = 1.0 / len(self.actions)
            return {a: uniform_prob for a in self.actions}


class CFRSolver:
    def __init__(self, game_data: dict):
        self.game_data = game_data
        self.nodes = {node['id']: node for node in game_data['nodes']}
        self.root_id = game_data['root']

        # Build infoset data
        self.infosets: Dict[str, InfoSetData] = {}
        self.infoset_player: Dict[str, int] = {}

        for node in game_data['nodes']:
            if node['kind'] == 'decision':
                infoset = node['infoset']
                if infoset not in self.infosets:
                    actions = [a['label'] for a in node['actions']]
                    self.infosets[infoset] = InfoSetData(actions)
                    self.infoset_player[infoset] = node['player']

    def cfr_recursive(self, node_id: int, p0: float, p1: float) -> float:
        """
        CFR traversal. Returns utility for player 0.
        p0, p1 = reach probabilities for each player.
        """
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_rational(node['payoff_p0'])

        if node['kind'] == 'chance':
            utility = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                utility += prob * self.cfr_recursive(outcome['child'], p0, p1)
            return utility

        # Decision node
        player = node['player']
        infoset = node['infoset']
        infoset_data = self.infosets[infoset]

        # Get current strategy
        strategy = infoset_data.get_strategy()

        # Recursively compute utilities for each action
        action_utils = {}
        for action_node in node['actions']:
            action = action_node['label']

            if player == 0:
                action_utils[action] = self.cfr_recursive(
                    action_node['child'],
                    p0 * strategy[action],
                    p1
                )
            else:
                action_utils[action] = self.cfr_recursive(
                    action_node['child'],
                    p0,
                    p1 * strategy[action]
                )

        # Node utility is weighted sum (from P0's perspective)
        util = sum(strategy[a] * action_utils[a] for a in strategy.keys())

        # Update regrets and strategy sum
        reach_prob = p0 if player == 0 else p1
        cf_reach_prob = p1 if player == 0 else p0

        for action in strategy.keys():
            # Convert to acting player's perspective for regret
            if player == 0:
                action_value = action_utils[action]
                node_value = util
            else:
                # Player 1 maximizes their own utility = -P0's utility
                action_value = -action_utils[action]
                node_value = -util

            # Regret = how much better this action is than current strategy
            regret = action_value - node_value
            infoset_data.regret_sum[action] += cf_reach_prob * regret

            # Accumulate strategy (weighted by own reach probability)
            infoset_data.strategy_sum[action] += reach_prob * strategy[action]

        return util

    def train(self, iterations: int):
        """Run CFR for specified iterations."""
        for i in range(iterations):
            self.cfr_recursive(self.root_id, 1.0, 1.0)

            if (i + 1) % 10000 == 0:
                print(f"Iteration {i + 1}/{iterations}")

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get average strategy for all infosets."""
        return {
            infoset: data.get_average_strategy()
            for infoset, data in self.infosets.items()
        }

    def evaluate_strategy(self, strategy: Dict[str, Dict[str, float]], node_id: int) -> float:
        """Compute expected utility for P0 when both players follow given strategy."""
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            return parse_rational(node['payoff_p0'])

        if node['kind'] == 'chance':
            util = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                util += prob * self.evaluate_strategy(strategy, outcome['child'])
            return util

        # Decision node
        infoset = node['infoset']
        strat = strategy[infoset]

        util = 0.0
        for action_node in node['actions']:
            action = action_node['label']
            util += strat[action] * self.evaluate_strategy(strategy, action_node['child'])

        return util

    def best_response(self, player: int, strategy: Dict[str, Dict[str, float]], node_id: int) -> float:
        """
        Compute best-response utility for given player against opponent's strategy.
        Returns utility from the specified player's perspective.
        """
        node = self.nodes[node_id]

        if node['kind'] == 'terminal':
            payoff_p0 = parse_rational(node['payoff_p0'])
            return payoff_p0 if player == 0 else -payoff_p0

        if node['kind'] == 'chance':
            util = 0.0
            for outcome in node['outcomes']:
                prob = parse_rational(outcome['prob'])
                util += prob * self.best_response(player, strategy, outcome['child'])
            return util

        # Decision node
        acting_player = node['player']

        if acting_player == player:
            # Best respond: maximize
            best_util = float('-inf')
            for action_node in node['actions']:
                util = self.best_response(player, strategy, action_node['child'])
                best_util = max(best_util, util)
            return best_util
        else:
            # Opponent follows their strategy
            infoset = node['infoset']
            strat = strategy[infoset]

            util = 0.0
            for action_node in node['actions']:
                action = action_node['label']
                util += strat[action] * self.best_response(player, strategy, action_node['child'])
            return util

    def compute_exploitability(self, strategy: Dict[str, Dict[str, float]]) -> float:
        """
        Compute exploitability of the strategy.
        exploitability = (BR_0 value + BR_1 value) / 2 - game_value
        In zero-sum games, this simplifies to (BR_0 - BR_1) / 2
        where both are measured from P0's perspective.
        """
        # Best response values (from each player's perspective)
        br0_value = self.best_response(0, strategy, self.root_id)  # P0's value when BR
        br1_value = self.best_response(1, strategy, self.root_id)  # P1's value when BR

        # Game value under current strategy
        game_value = self.evaluate_strategy(strategy, self.root_id)

        # Exploitability: average gain from best-responding
        # For P0: br0_value - game_value
        # For P1: br1_value - (-game_value) = br1_value + game_value
        # Average: [(br0_value - game_value) + (br1_value + game_value)] / 2
        #        = (br0_value + br1_value) / 2
        # But br1_value is from P1's perspective, so from P0's perspective it's -br1_value
        # So we need: (br0_value - game_value) + ((-br1_value) - game_value) / 2
        #           = (br0_value - br1_value - 2*game_value) / 2
        #           = (br0_value - br1_value) / 2 - game_value

        # Actually, the standard formula is simpler:
        # exploitability = (br0_value + br1_value) / 2 - |game_value|
        # But since br1_value is from P1's view, we convert: br1_value_from_p0_view = -br1_value

        # The exploitability is how much each player can gain on average by deviating
        # = [(br0_value - game_value) + (br1_value - (-game_value))] / 2
        # = [(br0_value - game_value) + (br1_value + game_value)] / 2
        # = (br0_value + br1_value) / 2

        # Since br1_value is P1's utility, we need to convert to P0's perspective: -br1_value
        # Exploitability from P0's perspective:
        # = (br0_value - game_value - (-(-br1_value) - game_value)) / 2
        # Let me think more carefully...

        # Actually, the exploitability is just:
        # exp = (value_when_P0_BR - value_when_both_play_sigma) + (value_when_P1_BR_from_P1_view - value_when_both_play_sigma_from_P1_view) / 2
        # In zero-sum: value_from_P1_view = -value_from_P0_view
        # So: exp = (br0 - v) + (br1 - (-v)) / 2 = (br0 - v + br1 + v) / 2 = (br0 + br1) / 2
        # where br1 is P1's value = -P0's value when P1 BR

        # To be safe, let me compute it the straightforward way:
        p0_loss_from_p1_br = game_value - (-br1_value)  # What P0 loses when P1 BR (br1 is P1's gain)
        p0_gain_from_p0_br = br0_value - game_value  # What P0 gains when P0 BR

        exploitability = (p0_gain_from_p0_br + p0_loss_from_p1_br) / 2

        return exploitability


def main():
    # Load game
    with open('game.json', 'r') as f:
        game_data = json.load(f)

    print(f"Loaded game: {game_data['name']}")

    # Create solver
    solver = CFRSolver(game_data)
    print(f"Found {len(solver.infosets)} information sets")

    # Train
    iterations = 10000000
    print(f"\nTraining vanilla CFR for {iterations} iterations...")
    solver.train(iterations)

    # Get results
    avg_strategy = solver.get_average_strategy()

    print("\nComputing metrics...")
    game_value = solver.evaluate_strategy(avg_strategy, solver.root_id)
    exploitability = solver.compute_exploitability(avg_strategy)

    print(f"\nExpected value (P0): {game_value:.10f}")
    print(f"Exploitability: {exploitability:.10f}")

    # Write strategy
    strategy_output = {
        "format": "bluff-strategy-v1",
        "game": game_data['name'],
        "producer": f"vanilla CFR solver ({iterations} iterations)",
        "probs": {
            infoset: {action: repr(prob) for action, prob in probs.items()}
            for infoset, probs in avg_strategy.items()
        }
    }

    with open('strategy.json', 'w') as f:
        json.dump(strategy_output, f, indent=1)

    print("\nStrategy written to strategy.json")
    print("\n" + "="*60)
    print(f"Converged: {exploitability < 0.001}")
    print(f"Exploitability: {exploitability:.10f}")
    print(f"Game value (P0): {game_value:.10f}")
    print("="*60)


if __name__ == '__main__':
    main()
