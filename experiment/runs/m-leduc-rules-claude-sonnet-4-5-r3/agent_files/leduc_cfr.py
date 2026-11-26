#!/usr/bin/env python3
"""Leduc hold'em CFR solver."""

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class LeducCFR:
    """CFR solver for Leduc hold'em."""

    CARDS = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
    RANKS = {'J': 0, 'Q': 1, 'K': 2}

    def __init__(self):
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def get_strategy(self, infoset: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy using regret matching."""
        regrets = self.regret_sum[infoset]

        positive_regrets = {a: max(0, regrets[a]) for a in actions}
        total = sum(positive_regrets.values())

        if total > 0:
            return {a: positive_regrets[a] / total for a in actions}
        else:
            return {a: 1.0 / len(actions) for a in actions}

    def get_infoset(self, card: str, board: Optional[str], history: str) -> str:
        """Build information set string."""
        if board is None:
            return f"{card}:{history}"
        else:
            round1, round2 = history.split('/')
            return f"{card}|{board}:{round1}/{round2}"

    def get_legal_actions(self, history: str, round_num: int) -> List[str]:
        """Get legal actions given history."""
        # Parse current round
        if '/' in history:
            _, current_round = history.split('/')
        else:
            current_round = history

        actions = ['call']

        # Count raises in current round
        raises = current_round.count('r')
        if raises < 2:
            actions.append('raise')

        # Can fold only if facing a bet
        if len(current_round) > 0 and current_round[-1] == 'r':
            actions.append('fold')

        return actions

    def is_terminal(self, history: str) -> bool:
        """Check if history represents a terminal state."""
        if len(history) == 0:
            return False

        # Fold
        if history[-1] == 'f':
            return True

        # Round ended with call
        if '/' not in history:
            # First round
            if len(history) >= 2 and history[-1] == 'c':
                return False  # Move to next round
        else:
            # Second round
            parts = history.split('/')
            if len(parts[1]) >= 2 and parts[1][-1] == 'c':
                return True  # Showdown

        return False

    def get_payoff(self, cards: List[str], history: str) -> float:
        """Get payoff for player 0 at terminal node."""
        p0_card = cards[0]
        p1_card = cards[1]
        board = cards[2] if len(cards) > 2 else None

        # Check for fold
        if history[-1] == 'f':
            # Last actor folded, opponent wins
            # Determine who acted last
            if '/' in history:
                round1, round2 = history.split('/')
                total_actions = len(round1) + len(round2)
            else:
                total_actions = len(history)

            last_actor = total_actions % 2

            # Calculate pot before fold
            pot = self._calculate_pot(history[:-1])  # Exclude fold action
            if last_actor == 0:
                return -pot[0]
            else:
                return pot[1]

        # Showdown (must have board)
        assert board is not None, "Showdown requires board card"
        pot = self._calculate_pot(history)

        p0_rank = self.RANKS[p0_card[0]]
        p1_rank = self.RANKS[p1_card[0]]

        p0_pair = (p0_card[0] == board[0])
        p1_pair = (p1_card[0] == board[0])

        if p0_pair and not p1_pair:
            return pot[1]
        elif p1_pair and not p0_pair:
            return -pot[0]
        elif p0_rank > p1_rank:
            return pot[1]
        elif p1_rank > p0_rank:
            return -pot[0]
        else:
            return 0  # Tie

    def _calculate_pot(self, history: str) -> Tuple[int, int]:
        """Calculate chips invested by each player."""
        invested = [1, 1]  # Antes

        if '/' in history:
            round1, round2 = history.split('/')
            rounds = [(round1, 2), (round2, 4)]
        else:
            rounds = [(history, 2)]

        for round_history, raise_size in rounds:
            player = 0
            for action in round_history:
                if action == 'r':
                    # Raise: match opponent + raise
                    to_call = invested[1 - player] - invested[player]
                    invested[player] += to_call + raise_size
                elif action == 'c':
                    # Call: match opponent
                    to_call = invested[1 - player] - invested[player]
                    invested[player] += to_call
                player = 1 - player

        return invested

    def next_to_act(self, history: str) -> int:
        """Determine who acts next."""
        if '/' in history:
            round1, round2 = history.split('/')
            total = len(round1) + len(round2)
        else:
            total = len(history)
        return total % 2

    def advance_history(self, history: str, action: str) -> str:
        """Add action to history."""
        action_char = {'fold': 'f', 'call': 'c', 'raise': 'r'}[action]
        return history + action_char

    def should_deal_board(self, history: str) -> bool:
        """Check if we should deal board card after this history."""
        if '/' in history:
            return False
        if len(history) < 2:
            return False
        if history[-1] == 'c':
            return True
        return False

    def cfr(self, cards: List[str], history: str, reach_p0: float, reach_p1: float) -> float:
        """CFR algorithm. Returns expected value for player 0."""

        # Check if terminal
        if self.is_terminal(history):
            return self.get_payoff(cards, history)

        # Check if we need to deal board
        board = cards[2] if len(cards) > 2 else None
        if self.should_deal_board(history) and board is None:
            # Deal board and move to round 2
            new_history = history + '/'
            # Sample one board card (Monte Carlo)
            remaining = [c for c in self.CARDS if c not in cards[:2]]
            board_card = random.choice(remaining)
            new_cards = cards + [board_card]
            return self.cfr(new_cards, new_history, reach_p0, reach_p1)

        player = self.next_to_act(history)
        card = cards[player]

        # Determine round
        round_num = 0 if '/' not in history else 1

        infoset = self.get_infoset(card, board, history)
        actions = self.get_legal_actions(history, round_num)

        strategy = self.get_strategy(infoset, actions)

        # Accumulate strategy
        reach = reach_p0 if player == 0 else reach_p1
        for action in actions:
            self.strategy_sum[infoset][action] += reach * strategy[action]

        # Compute action values
        action_values = {}
        for action in actions:
            new_history = self.advance_history(history, action)

            if player == 0:
                action_values[action] = self.cfr(
                    cards, new_history,
                    reach_p0 * strategy[action], reach_p1
                )
            else:
                action_values[action] = self.cfr(
                    cards, new_history,
                    reach_p0, reach_p1 * strategy[action]
                )

        node_value = sum(strategy[a] * action_values[a] for a in actions)

        # Update regrets
        opp_reach = reach_p1 if player == 0 else reach_p0
        for action in actions:
            regret = action_values[action] - node_value
            self.regret_sum[infoset][action] += opp_reach * regret

        return node_value

    def train(self, iterations: int):
        """Train for specified iterations."""
        for i in range(iterations):
            # Sample initial deal
            cards = random.sample(self.CARDS, 2)

            self.cfr(cards, '', 1.0, 1.0)
            self.iterations += 1

            if (i + 1) % 10000 == 0:
                print(f"Iteration {i + 1}/{iterations}")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Get average strategy."""
        result = {}

        for infoset, action_sums in self.strategy_sum.items():
            total = sum(action_sums.values())
            if total > 0:
                result[infoset] = {
                    action: repr(action_sums[action] / total)
                    for action in action_sums
                }
            else:
                n_actions = len(action_sums)
                result[infoset] = {
                    action: repr(1.0 / n_actions)
                    for action in action_sums
                }

        return result


def main():
    print("Starting Leduc hold'em CFR solver...")

    solver = LeducCFR()

    iterations = 100000
    print(f"Training for {iterations} iterations...")
    solver.train(iterations)

    print("Extracting average strategy...")
    strategy = solver.get_average_strategy()

    output = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": "CFR solver for Leduc hold'em",
        "probs": strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print(f"Strategy written to strategy.json")
    print(f"Total information sets: {len(strategy)}")


if __name__ == '__main__':
    main()
