#!/usr/bin/env python3
"""
Kuhn Poker solver using Counterfactual Regret Minimization (CFR).
"""

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple


class KuhnCFR:
    def __init__(self):
        # Cards: 0=J, 1=Q, 2=K
        self.cards = [0, 1, 2]

        # Information sets store cumulative regrets and strategy sums
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

        # Map card numbers to names for output
        self.card_names = {0: 'J', 1: 'Q', 2: 'K'}

    def get_strategy(self, infoset: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy for an information set using regret matching."""
        regrets = self.regret_sum[infoset]

        # Calculate positive regrets
        positive_regret_sum = sum(max(regrets[a], 0) for a in actions)

        strategy = {}
        if positive_regret_sum > 0:
            # Use regret matching
            for action in actions:
                strategy[action] = max(regrets[action], 0) / positive_regret_sum
        else:
            # Uniform strategy if no positive regrets
            for action in actions:
                strategy[action] = 1.0 / len(actions)

        return strategy

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get the average strategy across all iterations."""
        avg_strategy = {}

        for infoset, action_sums in self.strategy_sum.items():
            total = sum(action_sums.values())
            avg_strategy[infoset] = {}

            if total > 0:
                for action, sum_val in action_sums.items():
                    avg_strategy[infoset][action] = str(sum_val / total)
            else:
                # If never reached, use uniform
                num_actions = len(action_sums)
                for action in action_sums.keys():
                    avg_strategy[infoset][action] = str(1.0 / num_actions)

        return avg_strategy

    def get_payoff(self, cards: Tuple[int, int], history: str) -> float:
        """
        Get the payoff for a terminal history from player 0's perspective.

        Args:
            cards: (player0_card, player1_card)
            history: action history

        Returns:
            Payoff for player 0
        """
        if history == 'cc':  # Both checked
            # Showdown, pot is 2
            return 1.0 if cards[0] > cards[1] else -1.0

        if history == 'bc':  # P0 bet, P1 called
            # Showdown, pot is 4
            return 2.0 if cards[0] > cards[1] else -2.0

        if history == 'bf':  # P0 bet, P1 folded
            # P0 wins pot of 2 (antes), net +1
            return 1.0

        if history == 'cbc':  # P0 checked, P1 bet, P0 called
            # Showdown, pot is 4
            return 2.0 if cards[0] > cards[1] else -2.0

        if history == 'cbf':  # P0 checked, P1 bet, P0 folded
            # P1 wins pot of 2 (antes), P0 net -1
            return -1.0

        raise ValueError(f"Not a terminal history: {history}")

    def is_terminal(self, history: str) -> bool:
        """Check if the history represents a terminal state."""
        return history in ['cc', 'bc', 'bf', 'cbc', 'cbf']

    def get_actions(self, history: str) -> List[str]:
        """Get available actions for a given history."""
        if len(history) == 0:
            return ['check', 'bet']
        elif history == 'c':
            return ['check', 'bet']
        elif history == 'b':
            return ['fold', 'call']
        elif history == 'cb':
            return ['fold', 'call']
        else:
            raise ValueError(f"Unexpected history: {history}")

    def cfr(self, cards: Tuple[int, int], history: str, player: int,
            reach_probs: Tuple[float, float]) -> float:
        """
        Counterfactual regret minimization recursive function.

        Args:
            cards: (player0_card, player1_card)
            history: action history
            player: current player to act (0 or 1)
            reach_probs: (reach_prob_p0, reach_prob_p1)

        Returns:
            Expected utility for player 0
        """
        # Terminal state
        if self.is_terminal(history):
            return self.get_payoff(cards, history)

        # Get available actions
        actions = self.get_actions(history)

        # Build information set
        card_name = self.card_names[cards[player]]
        infoset = f"{card_name}:{history}"

        # Get current strategy
        strategy = self.get_strategy(infoset, actions)

        # Add strategy to strategy sum (weighted by reach probability)
        for action in actions:
            self.strategy_sum[infoset][action] += reach_probs[player] * strategy[action]

        # Compute counterfactual values for each action
        action_utilities = {}
        for action in actions:
            # Build next history
            if action == 'check':
                next_history = history + 'c'
            elif action == 'bet':
                next_history = history + 'b'
            elif action == 'fold':
                next_history = history + 'f'
            elif action == 'call':
                next_history = history + 'c'
            else:
                raise ValueError(f"Unexpected action: {action}")

            # Update reach probabilities
            next_reach_probs = list(reach_probs)
            next_reach_probs[player] *= strategy[action]
            next_reach_probs = tuple(next_reach_probs)

            # Recurse
            action_utilities[action] = self.cfr(cards, next_history, 1 - player, next_reach_probs)

        # Compute node utility (expected value under current strategy)
        node_utility = sum(strategy[action] * action_utilities[action] for action in actions)

        # Update regrets
        # For player 1, we need to negate utilities since they're from player 0's perspective
        for action in actions:
            if player == 0:
                action_util = action_utilities[action]
                node_util = node_utility
            else:
                action_util = -action_utilities[action]
                node_util = -node_utility

            regret = action_util - node_util
            self.regret_sum[infoset][action] += reach_probs[1 - player] * regret

        return node_utility

    def train(self, iterations: int = 100000):
        """Train the CFR algorithm."""
        for i in range(iterations):
            # Shuffle cards and deal
            cards_shuffled = self.cards.copy()
            random.shuffle(cards_shuffled)
            cards = (cards_shuffled[0], cards_shuffled[1])

            # Run CFR (starting with player 0)
            self.cfr(cards, '', 0, (1.0, 1.0))

            if (i + 1) % 10000 == 0:
                print(f"Iteration {i + 1}/{iterations}")

        print(f"\nTraining complete.")

    def save_strategy(self, filename: str = "strategy.json"):
        """Save the average strategy to a JSON file."""
        avg_strategy = self.get_average_strategy()

        output = {
            "format": "bluff-strategy-v1",
            "game": "kuhn",
            "producer": "CFR solver for Kuhn poker",
            "probs": avg_strategy
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Strategy saved to {filename}")

    def compute_exploitability(self) -> float:
        """Compute the exploitability of the current average strategy."""
        avg_strategy = self.get_average_strategy()

        # Compute best response value for each player
        br_value_p0 = self.best_response_value(avg_strategy, 0)
        br_value_p1 = self.best_response_value(avg_strategy, 1)

        # Exploitability is the average of BR values
        exploitability = (br_value_p0 + br_value_p1) / 2
        return exploitability

    def best_response_value(self, strategy: Dict[str, Dict[str, str]], br_player: int) -> float:
        """
        Compute the value gained by br_player when deviating to best response.

        Returns how much more br_player can gain over the equilibrium.
        """
        total_eq_value = 0
        total_br_value = 0
        num_deals = 0

        # Iterate over all possible card deals
        for p0_card in self.cards:
            for p1_card in self.cards:
                if p0_card == p1_card:
                    continue

                cards = (p0_card, p1_card)

                # Value when both play strategy
                eq_value_p0 = self.expected_value(cards, '', 0, strategy)

                # Value when br_player plays BR
                br_value_p0 = self.br_recursive(cards, '', 0, strategy, br_player)

                if br_player == 0:
                    total_eq_value += eq_value_p0
                    total_br_value += br_value_p0
                else:
                    total_eq_value += -eq_value_p0
                    total_br_value += -br_value_p0

                num_deals += 1

        eq_value = total_eq_value / num_deals
        br_value = total_br_value / num_deals

        # Return how much br_player gains by deviating
        return br_value - eq_value

    def br_recursive(self, cards: Tuple[int, int], history: str, player: int,
                     strategy: Dict[str, Dict[str, str]], br_player: int) -> float:
        """
        Compute value with best response for br_player and given strategy for opponent.
        Returns value from player 0's perspective.
        """
        # Terminal state
        if self.is_terminal(history):
            return self.get_payoff(cards, history)

        actions = self.get_actions(history)
        card_name = self.card_names[cards[player]]
        infoset = f"{card_name}:{history}"

        if player == br_player:
            # Best response: choose action with maximum value
            best_value = float('-inf')

            for action in actions:
                if action == 'check':
                    next_history = history + 'c'
                elif action == 'bet':
                    next_history = history + 'b'
                elif action == 'fold':
                    next_history = history + 'f'
                elif action == 'call':
                    next_history = history + 'c'

                value = self.br_recursive(cards, next_history, 1 - player, strategy, br_player)
                best_value = max(best_value, value)

            return best_value
        else:
            # Play according to strategy
            strat = strategy[infoset]
            expected_value = 0

            for action in actions:
                if action == 'check':
                    next_history = history + 'c'
                elif action == 'bet':
                    next_history = history + 'b'
                elif action == 'fold':
                    next_history = history + 'f'
                elif action == 'call':
                    next_history = history + 'c'

                action_prob = float(strat[action])
                value = self.br_recursive(cards, next_history, 1 - player, strategy, br_player)
                expected_value += action_prob * value

            return expected_value

    def compute_game_value(self) -> float:
        """Compute expected value for player 0 when both play the average strategy."""
        avg_strategy = self.get_average_strategy()

        total_value = 0
        num_deals = 0

        for p0_card in self.cards:
            for p1_card in self.cards:
                if p0_card == p1_card:
                    continue

                cards = (p0_card, p1_card)
                value = self.expected_value(cards, '', 0, avg_strategy)
                total_value += value
                num_deals += 1

        return total_value / num_deals

    def expected_value(self, cards: Tuple[int, int], history: str, player: int,
                       strategy: Dict[str, Dict[str, str]]) -> float:
        """Compute expected value from player 0's perspective with both playing strategy."""
        # Terminal state
        if self.is_terminal(history):
            return self.get_payoff(cards, history)

        actions = self.get_actions(history)
        card_name = self.card_names[cards[player]]
        infoset = f"{card_name}:{history}"
        strat = strategy[infoset]

        expected_value = 0
        for action in actions:
            if action == 'check':
                next_history = history + 'c'
            elif action == 'bet':
                next_history = history + 'b'
            elif action == 'fold':
                next_history = history + 'f'
            elif action == 'call':
                next_history = history + 'c'

            action_prob = float(strat[action])
            value = self.expected_value(cards, next_history, 1 - player, strategy)
            expected_value += action_prob * value

        return expected_value


def main():
    print("Kuhn Poker CFR Solver")
    print("=" * 50)

    # Create solver
    solver = KuhnCFR()

    # Train
    print("\nTraining CFR...")
    solver.train(iterations=500000)

    # Save strategy
    print()
    solver.save_strategy("strategy.json")

    # Compute metrics
    print("\nComputing metrics...")
    exploitability = solver.compute_exploitability()
    game_value = solver.compute_game_value()

    print(f"Exploitability: {exploitability:.10f}")
    print(f"Game value (player 0): {game_value:.10f}")

    # Print sample strategies
    avg_strategy = solver.get_average_strategy()
    print("\nSample strategies:")
    for infoset in sorted(avg_strategy.keys()):
        probs = avg_strategy[infoset]
        prob_str = ", ".join([f"{a}: {float(probs[a]):.4f}" for a in sorted(probs.keys())])
        print(f"  {infoset}: {prob_str}")

    return exploitability, game_value


if __name__ == "__main__":
    exploitability, game_value = main()
