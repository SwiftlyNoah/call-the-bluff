#!/usr/bin/env python3
"""
Kuhn Poker solver using Counterfactual Regret Minimization (CFR+).
"""

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple


class KuhnPoker:
    """Kuhn poker game implementation."""

    CARDS = ['J', 'Q', 'K']
    CARD_RANK = {'J': 0, 'Q': 1, 'K': 2}

    def __init__(self):
        # Regret and strategy sums for each info set and action
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def get_strategy(self, infoset: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy for an information set using regret matching."""
        regrets = self.regret_sum[infoset]

        # Use CFR+ (ignore negative regrets)
        positive_regrets = {a: max(0, regrets[a]) for a in actions}
        total_regret = sum(positive_regrets.values())

        if total_regret > 0:
            strategy = {a: positive_regrets[a] / total_regret for a in actions}
        else:
            # Uniform strategy if no positive regrets
            strategy = {a: 1.0 / len(actions) for a in actions}

        return strategy

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get average strategy across all iterations."""
        avg_strategy = {}

        for infoset in self.strategy_sum:
            total = sum(self.strategy_sum[infoset].values())
            if total > 0:
                avg_strategy[infoset] = {
                    action: repr(prob / total)
                    for action, prob in self.strategy_sum[infoset].items()
                }
            else:
                # If never reached, use uniform
                actions = list(self.strategy_sum[infoset].keys())
                avg_strategy[infoset] = {
                    action: repr(1.0 / len(actions)) for action in actions
                }

        return avg_strategy

    def cfr(self, cards: Tuple[str, str], history: str,
            reach_probs: Tuple[float, float]) -> Tuple[float, float]:
        """
        Counterfactual regret minimization recursive function.

        Args:
            cards: (player0_card, player1_card)
            history: action history ('c' for check, 'b' for bet, 'f' for fold)
            reach_probs: (prob player 0 reaches this node, prob player 1 reaches this node)

        Returns:
            (Expected utility for player 0, Expected utility for player 1)
        """
        # Determine whose turn it is based on history length
        # Player 0 acts at positions 0, 2 (after 'cb')
        # Player 1 acts at positions 1 (after 'c' or 'b')
        num_actions = len(history)

        # Check if terminal node
        if self.is_terminal(history):
            payoff = self.get_payoff(cards, history)
            return (payoff, -payoff)

        # Determine current player
        if num_actions == 0:
            player = 0
        elif num_actions == 1:
            player = 1
        elif num_actions == 2 and history == 'cb':
            player = 0
        else:
            raise ValueError(f"Unexpected history: {history}")

        # Get current player's card and infoset
        current_card = cards[player]
        infoset = f"{current_card}:{history}"

        # Get available actions
        actions = self.get_actions(history)

        # Get current strategy
        strategy = self.get_strategy(infoset, actions)

        # Compute action utilities
        action_utilities = {action: [0.0, 0.0] for action in actions}
        for action in actions:
            new_history = history + ('c' if action == 'check' else 'b' if action == 'bet' else action[0])

            # Update reach probabilities
            new_reach_probs = list(reach_probs)
            new_reach_probs[player] *= strategy[action]

            # Recurse
            action_utilities[action] = self.cfr(cards, new_history, tuple(new_reach_probs))

        # Node utility is weighted sum of action utilities
        node_utility = [0.0, 0.0]
        for action in actions:
            node_utility[0] += strategy[action] * action_utilities[action][0]
            node_utility[1] += strategy[action] * action_utilities[action][1]

        # Update regrets and strategy sum for current player
        for action in actions:
            regret = action_utilities[action][player] - node_utility[player]
            opponent_reach = reach_probs[1 - player]
            self.regret_sum[infoset][action] += opponent_reach * regret
            self.strategy_sum[infoset][action] += reach_probs[player] * strategy[action]

        return tuple(node_utility)

    def is_terminal(self, history: str) -> bool:
        """Check if the history represents a terminal node."""
        if len(history) == 0:
            return False

        # Fold
        if history.endswith('bf') or history.endswith('cbf'):
            return True

        # Call (showdown)
        if history.endswith('bc') or history.endswith('cbc'):
            return True

        # Check-check (showdown)
        if history == 'cc':
            return True

        return False

    def get_payoff(self, cards: Tuple[str, str], history: str) -> float:
        """
        Get payoff for player 0 at a terminal node.

        Returns net chips won/lost by player 0.
        """
        p0_card, p1_card = cards

        # Player 1 folds to player 0's bet
        if history == 'bf':
            return 1.0  # Player 0 wins the antes

        # Player 0 folds to player 1's bet
        if history == 'cbf':
            return -1.0  # Player 1 wins the antes

        # Showdown cases
        p0_wins = self.CARD_RANK[p0_card] > self.CARD_RANK[p1_card]

        if history == 'cc':
            # Both checked, pot is 2 (antes only)
            return 1.0 if p0_wins else -1.0

        if history == 'bc':
            # Player 0 bet, player 1 called, pot is 4
            return 2.0 if p0_wins else -2.0

        if history == 'cbc':
            # Player 0 checked, player 1 bet, player 0 called, pot is 4
            return 2.0 if p0_wins else -2.0

        raise ValueError(f"Unexpected terminal history: {history}")

    def get_actions(self, history: str) -> List[str]:
        """Get available actions given the history."""
        if len(history) == 0:
            # Player 0's first action
            return ['check', 'bet']

        if history == 'c':
            # Player 1 after player 0 checks
            return ['check', 'bet']

        if history == 'b':
            # Player 1 after player 0 bets
            return ['fold', 'call']

        if history == 'cb':
            # Player 0 after checking and facing a bet
            return ['fold', 'call']

        raise ValueError(f"Unexpected history: {history}")

    def train(self, iterations: int):
        """Train the CFR algorithm for the specified number of iterations."""
        util_sum = 0

        for i in range(iterations):
            # Sample a random card deal
            cards = random.sample(self.CARDS, 2)

            # Run CFR from the root
            utilities = self.cfr(tuple(cards), '', (1.0, 1.0))
            util_sum += utilities[0]

            self.iterations += 1

            if (i + 1) % 10000 == 0:
                print(f"Iteration {i + 1}/{iterations}, Avg util: {util_sum / (i + 1):.6f}")

        print(f"Training complete: {iterations} iterations")
        print(f"Average utility for player 0: {util_sum / iterations:.6f}")

    def save_strategy(self, filename: str):
        """Save the average strategy to a JSON file."""
        avg_strategy = self.get_average_strategy()

        output = {
            "format": "bluff-strategy-v1",
            "game": "kuhn",
            "producer": "CFR+ solver for Kuhn poker",
            "probs": avg_strategy
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Strategy saved to {filename}")

    def compute_exploitability(self) -> float:
        """Compute exploitability of the average strategy."""
        avg_strategy = self.get_average_strategy()

        # Convert string probs back to floats
        strategy = {}
        for infoset, action_probs in avg_strategy.items():
            strategy[infoset] = {a: eval(p) for a, p in action_probs.items()}

        # Compute best response for each player
        br0_value = self.best_response_value(0, strategy)
        br1_value = self.best_response_value(1, strategy)

        # Exploitability is average of best response values
        exploitability = (br0_value + br1_value) / 2

        return exploitability

    def best_response_value(self, br_player: int, strategy: Dict) -> float:
        """
        Compute the value of the best response for br_player against the opponent's strategy.

        Returns the expected value for br_player.
        """
        total_value = 0
        num_deals = 0

        # Enumerate all possible card deals
        for p0_card in self.CARDS:
            for p1_card in self.CARDS:
                if p0_card == p1_card:
                    continue

                cards = (p0_card, p1_card)
                value = self.br_recursive(cards, '', br_player, strategy)

                # From player 0's perspective, then convert if needed
                if br_player == 0:
                    total_value += value
                else:
                    total_value += -value

                num_deals += 1

        # Average over all deals
        avg_value = total_value / num_deals

        return avg_value

    def br_recursive(self, cards: Tuple[str, str], history: str,
                     br_player: int, strategy: Dict) -> float:
        """
        Recursive function for computing best response value.
        Returns utility for player 0.
        """
        if self.is_terminal(history):
            return self.get_payoff(cards, history)

        # Determine current player
        num_actions = len(history)
        if num_actions == 0:
            player = 0
        elif num_actions == 1:
            player = 1
        elif num_actions == 2 and history == 'cb':
            player = 0
        else:
            raise ValueError(f"Unexpected history: {history}")

        actions = self.get_actions(history)
        current_card = cards[player]
        infoset = f"{current_card}:{history}"

        if player == br_player:
            # Best response player: choose best action
            best_value = float('-inf') if player == 0 else float('inf')

            for action in actions:
                new_history = history + ('c' if action == 'check' else 'b' if action == 'bet' else action[0])
                value = self.br_recursive(cards, new_history, br_player, strategy)

                if player == 0:
                    best_value = max(best_value, value)
                else:
                    best_value = min(best_value, value)

            return best_value
        else:
            # Opponent: follow their strategy
            action_probs = strategy.get(infoset, {a: 1.0 / len(actions) for a in actions})
            expected_value = 0

            for action in actions:
                new_history = history + ('c' if action == 'check' else 'b' if action == 'bet' else action[0])
                value = self.br_recursive(cards, new_history, br_player, strategy)
                expected_value += action_probs[action] * value

            return expected_value

    def compute_strategy_value(self) -> float:
        """Compute the value of the game when both players follow the average strategy."""
        avg_strategy = self.get_average_strategy()

        # Convert string probs back to floats
        strategy = {}
        for infoset, action_probs in avg_strategy.items():
            strategy[infoset] = {a: eval(p) for a, p in action_probs.items()}

        total_value = 0
        num_deals = 0

        # Enumerate all possible card deals
        for p0_card in self.CARDS:
            for p1_card in self.CARDS:
                if p0_card == p1_card:
                    continue

                cards = (p0_card, p1_card)
                value = self.strategy_value_recursive(cards, '', strategy)
                total_value += value
                num_deals += 1

        return total_value / num_deals

    def strategy_value_recursive(self, cards: Tuple[str, str], history: str,
                                  strategy: Dict) -> float:
        """Compute expected value when both players follow strategy. Returns utility for player 0."""
        if self.is_terminal(history):
            return self.get_payoff(cards, history)

        # Determine current player
        num_actions = len(history)
        if num_actions == 0:
            player = 0
        elif num_actions == 1:
            player = 1
        elif num_actions == 2 and history == 'cb':
            player = 0
        else:
            raise ValueError(f"Unexpected history: {history}")

        actions = self.get_actions(history)
        current_card = cards[player]
        infoset = f"{current_card}:{history}"

        action_probs = strategy.get(infoset, {a: 1.0 / len(actions) for a in actions})
        expected_value = 0

        for action in actions:
            new_history = history + ('c' if action == 'check' else 'b' if action == 'bet' else action[0])
            value = self.strategy_value_recursive(cards, new_history, strategy)
            expected_value += action_probs[action] * value

        return expected_value


def main():
    """Main function to run the CFR solver."""
    print("=" * 60)
    print("Kuhn Poker CFR+ Solver")
    print("=" * 60)

    # Create solver
    solver = KuhnPoker()

    # Train
    num_iterations = 2000000
    print(f"\nTraining for {num_iterations} iterations...")
    solver.train(num_iterations)

    # Save strategy
    print("\nSaving strategy...")
    solver.save_strategy('strategy.json')

    # Compute metrics
    print("\nComputing metrics...")
    exploitability = solver.compute_exploitability()
    strategy_value = solver.compute_strategy_value()

    print(f"\nExploitability: {exploitability:.8f}")
    print(f"Strategy value (player 0): {strategy_value:.8f}")

    # Display some key strategies
    print("\n" + "=" * 60)
    print("Sample strategies:")
    print("=" * 60)
    avg_strategy = solver.get_average_strategy()
    for infoset in sorted(avg_strategy.keys()):
        probs_str = ', '.join([f"{a}: {float(p):.4f}" for a, p in avg_strategy[infoset].items()])
        print(f"{infoset:6} -> {probs_str}")

    return exploitability, strategy_value


if __name__ == '__main__':
    exploitability, strategy_value = main()
