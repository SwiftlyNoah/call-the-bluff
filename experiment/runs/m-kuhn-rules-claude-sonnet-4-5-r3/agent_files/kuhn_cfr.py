#!/usr/bin/env python3
"""Kuhn poker solver using Counterfactual Regret Minimization (CFR)."""

import json
from collections import defaultdict
from typing import Dict

# Card rankings
CARDS = ['J', 'Q', 'K']
CARD_RANK = {'J': 0, 'Q': 1, 'K': 2}

# Actions
CHECK = 'check'
BET = 'bet'
FOLD = 'fold'
CALL = 'call'


class KuhnCFR:
    def __init__(self):
        # Regret and strategy sums for each information set and action
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.action_map = {
            # Player 0 initial
            'J:': [CHECK, BET],
            'Q:': [CHECK, BET],
            'K:': [CHECK, BET],
            # Player 0 after check-bet
            'J:cb': [FOLD, CALL],
            'Q:cb': [FOLD, CALL],
            'K:cb': [FOLD, CALL],
            # Player 1 after check
            'J:c': [CHECK, BET],
            'Q:c': [CHECK, BET],
            'K:c': [CHECK, BET],
            # Player 1 after bet
            'J:b': [FOLD, CALL],
            'Q:b': [FOLD, CALL],
            'K:b': [FOLD, CALL],
        }

    def get_strategy(self, info_set: str) -> Dict[str, float]:
        """Get current mixed strategy for an information set using regret matching."""
        actions = self.action_map[info_set]
        strategy = {}

        # Calculate strategy from regrets using regret matching
        normalizing_sum = 0.0
        for action in actions:
            strategy[action] = max(0.0, self.regret_sum[info_set][action])
            normalizing_sum += strategy[action]

        # Normalize to probabilities
        if normalizing_sum > 0:
            for action in actions:
                strategy[action] /= normalizing_sum
        else:
            # Uniform strategy if all regrets are non-positive
            uniform_prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = uniform_prob

        return strategy

    def cfr(self, p0_card: str, p1_card: str, history: str,
            reach_p0: float, reach_p1: float) -> float:
        """
        Run CFR recursively.
        Always returns the expected value for player 0.
        """
        # Check if we're at a terminal node (exact matches only)
        if history == 'cc':
            # Both checked, showdown
            if CARD_RANK[p0_card] > CARD_RANK[p1_card]:
                return 1  # P0 wins 1
            else:
                return -1  # P0 loses 1
        elif history == 'bc':
            # P0 bet, P1 called, showdown
            if CARD_RANK[p0_card] > CARD_RANK[p1_card]:
                return 2  # P0 wins 2
            else:
                return -2  # P0 loses 2
        elif history == 'cbc':
            # P0 checked, P1 bet, P0 called, showdown
            if CARD_RANK[p0_card] > CARD_RANK[p1_card]:
                return 2  # P0 wins 2
            else:
                return -2  # P0 loses 2
        elif history == 'bf':
            # P0 bet, P1 folded
            return 1  # P0 wins the antes
        elif history == 'cbf':
            # P0 checked, P1 bet, P0 folded
            return -1  # P0 loses their ante

        # Determine current player by counting actions
        num_actions = len(history)
        player = num_actions % 2

        # Get information set
        card = p0_card if player == 0 else p1_card
        info_set = f"{card}:{history}"

        # Get current strategy
        strategy = self.get_strategy(info_set)
        actions = self.action_map[info_set]

        # Compute action utilities (from P0's perspective)
        action_utils = {}
        for action in actions:
            # Map action to history character
            if action == CHECK:
                action_char = 'c'
            elif action == BET:
                action_char = 'b'
            elif action == FOLD:
                action_char = 'f'
            elif action == CALL:
                action_char = 'c'

            new_history = history + action_char

            # Recursively compute utility
            if player == 0:
                action_utils[action] = self.cfr(p0_card, p1_card, new_history,
                                                reach_p0 * strategy[action], reach_p1)
            else:
                action_utils[action] = self.cfr(p0_card, p1_card, new_history,
                                                reach_p0, reach_p1 * strategy[action])

        # Compute node utility (expected value from P0's perspective)
        node_util = sum(strategy[action] * action_utils[action] for action in actions)

        # Update regrets and strategy sum
        for action in actions:
            if player == 0:
                # P0's regret: values are already from P0's perspective
                regret = action_utils[action] - node_util
                self.regret_sum[info_set][action] += reach_p1 * regret
                self.strategy_sum[info_set][action] += reach_p0 * strategy[action]
            else:
                # P1's regret: values are from P0's perspective, so negate for P1
                regret = -action_utils[action] + node_util
                self.regret_sum[info_set][action] += reach_p0 * regret
                self.strategy_sum[info_set][action] += reach_p1 * strategy[action]

        return node_util

    def train(self, iterations: int):
        """Train the CFR algorithm for the specified number of iterations."""
        util_sum = 0.0

        for i in range(iterations):
            # Iterate over all possible deals
            iter_util = 0.0
            for p0_card in CARDS:
                for p1_card in CARDS:
                    if p0_card == p1_card:
                        continue
                    # Run CFR from the root
                    util = self.cfr(p0_card, p1_card, '', 1.0, 1.0)
                    iter_util += util

            # Average over all deals
            iter_util /= 6.0  # 6 possible deals
            util_sum += iter_util

            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}/{iterations}, Avg value: {util_sum / (i + 1):.6f}")

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get the average strategy across all training iterations."""
        avg_strategy = {}

        for info_set in self.action_map:
            actions = self.action_map[info_set]
            normalizing_sum = sum(self.strategy_sum[info_set][action] for action in actions)

            avg_strategy[info_set] = {}
            if normalizing_sum > 0:
                for action in actions:
                    avg_strategy[info_set][action] = self.strategy_sum[info_set][action] / normalizing_sum
            else:
                # Uniform if never visited
                uniform_prob = 1.0 / len(actions)
                for action in actions:
                    avg_strategy[info_set][action] = uniform_prob

        return avg_strategy

    def compute_exploitability(self, strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute exploitability of a strategy."""
        # For each player, compute best response value against opponent's strategy
        br_values = []

        for player in [0, 1]:
            value = self.best_response_value(strategy, player)
            br_values.append(value)

        # Exploitability is average of best response values
        return (br_values[0] + br_values[1]) / 2

    def best_response_value(self, strategy: Dict[str, Dict[str, float]], br_player: int) -> float:
        """Compute best response value for a player against opponent's strategy."""
        total_value = 0.0
        count = 0

        # Enumerate all possible deals
        for p0_card in CARDS:
            for p1_card in CARDS:
                if p0_card == p1_card:
                    continue
                value = self.best_response_recursive(strategy, br_player, p0_card, p1_card, '')
                total_value += value
                count += 1

        return total_value / count

    def best_response_recursive(self, strategy: Dict[str, Dict[str, float]],
                                br_player: int, p0_card: str, p1_card: str,
                                history: str) -> float:
        """Recursively compute best response value (always from P0's perspective)."""
        # Check terminal nodes
        if history == 'cc':
            if CARD_RANK[p0_card] > CARD_RANK[p1_card]:
                return 1
            else:
                return -1
        elif history == 'bc':
            if CARD_RANK[p0_card] > CARD_RANK[p1_card]:
                return 2
            else:
                return -2
        elif history == 'cbc':
            if CARD_RANK[p0_card] > CARD_RANK[p1_card]:
                return 2
            else:
                return -2
        elif history == 'bf':
            return 1
        elif history == 'cbf':
            return -1

        num_actions = len(history)
        current_player = num_actions % 2

        card = p0_card if current_player == 0 else p1_card
        info_set = f"{card}:{history}"
        actions = self.action_map[info_set]

        if current_player == br_player:
            # Best response player: choose best action
            best_value = float('-inf')
            for action in actions:
                if action == CHECK:
                    action_char = 'c'
                elif action == BET:
                    action_char = 'b'
                elif action == FOLD:
                    action_char = 'f'
                elif action == CALL:
                    action_char = 'c'

                new_history = history + action_char
                value = self.best_response_recursive(strategy, br_player, p0_card, p1_card, new_history)
                best_value = max(best_value, value)
            return best_value
        else:
            # Opponent: follow strategy
            expected_value = 0.0
            for action in actions:
                if action == CHECK:
                    action_char = 'c'
                elif action == BET:
                    action_char = 'b'
                elif action == FOLD:
                    action_char = 'f'
                elif action == CALL:
                    action_char = 'c'

                new_history = history + action_char
                prob = strategy[info_set][action]
                value = self.best_response_recursive(strategy, br_player, p0_card, p1_card, new_history)
                expected_value += prob * value
            return expected_value

    def compute_strategy_value(self, strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute expected value for player 0 when both follow the strategy."""
        total_value = 0.0
        count = 0

        for p0_card in CARDS:
            for p1_card in CARDS:
                if p0_card == p1_card:
                    continue
                value = self.strategy_value_recursive(strategy, p0_card, p1_card, '')
                total_value += value
                count += 1

        return total_value / count

    def strategy_value_recursive(self, strategy: Dict[str, Dict[str, float]],
                                 p0_card: str, p1_card: str, history: str) -> float:
        """Recursively compute strategy value (from P0's perspective)."""
        # Check terminal nodes
        if history == 'cc':
            if CARD_RANK[p0_card] > CARD_RANK[p1_card]:
                return 1
            else:
                return -1
        elif history == 'bc':
            if CARD_RANK[p0_card] > CARD_RANK[p1_card]:
                return 2
            else:
                return -2
        elif history == 'cbc':
            if CARD_RANK[p0_card] > CARD_RANK[p1_card]:
                return 2
            else:
                return -2
        elif history == 'bf':
            return 1
        elif history == 'cbf':
            return -1

        num_actions = len(history)
        current_player = num_actions % 2

        card = p0_card if current_player == 0 else p1_card
        info_set = f"{card}:{history}"
        actions = self.action_map[info_set]

        expected_value = 0.0
        for action in actions:
            if action == CHECK:
                action_char = 'c'
            elif action == BET:
                action_char = 'b'
            elif action == FOLD:
                action_char = 'f'
            elif action == CALL:
                action_char = 'c'

            new_history = history + action_char
            prob = strategy[info_set][action]
            value = self.strategy_value_recursive(strategy, p0_card, p1_card, new_history)
            expected_value += prob * value

        return expected_value


def main():
    print("Starting Kuhn poker CFR solver...")

    # Initialize and train
    solver = KuhnCFR()
    iterations = 500000
    print(f"Training for {iterations} iterations...")
    solver.train(iterations)

    # Get average strategy
    avg_strategy = solver.get_average_strategy()

    print("\nAverage strategy:")
    for info_set in sorted(avg_strategy.keys()):
        print(f"{info_set}: {avg_strategy[info_set]}")

    # Compute exploitability and value
    print("\nComputing exploitability...")
    exploitability = solver.compute_exploitability(avg_strategy)
    print(f"Exploitability: {exploitability:.6f}")

    print("\nComputing strategy value...")
    value_p0 = solver.compute_strategy_value(avg_strategy)
    print(f"Value for player 0: {value_p0:.6f}")

    # Convert to string probabilities for output
    output_strategy = {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": "CFR solver with 500k iterations",
        "probs": {}
    }

    for info_set, actions in avg_strategy.items():
        output_strategy["probs"][info_set] = {
            action: f"{prob:.10f}" for action, prob in actions.items()
        }

    # Write to file
    with open('strategy.json', 'w') as f:
        json.dump(output_strategy, f, indent=2, sort_keys=True)

    print("\nStrategy written to strategy.json")

    # Return metrics
    return exploitability, value_p0


if __name__ == '__main__':
    exploitability, value_p0 = main()
