#!/usr/bin/env python3
"""Evaluate Leduc strategy."""

import json
import random
from typing import Dict, List


class LeducEvaluator:
    """Evaluate Leduc hold'em strategy."""

    CARDS = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
    RANKS = {'J': 0, 'Q': 1, 'K': 2}

    def __init__(self, strategy: Dict[str, Dict[str, float]]):
        self.strategy = strategy

    def get_infoset(self, card: str, board: str, history: str) -> str:
        """Build information set string."""
        if '/' not in history:
            return f"{card}:{history}"
        else:
            round1, round2 = history.split('/')
            return f"{card}|{board}:{round1}/{round2}"

    def get_legal_actions(self, history: str) -> List[str]:
        """Get legal actions."""
        if '/' in history:
            _, current_round = history.split('/')
        else:
            current_round = history

        actions = ['call']

        raises = current_round.count('r')
        if raises < 2:
            actions.append('raise')

        if len(current_round) > 0 and current_round[-1] == 'r':
            actions.append('fold')

        return actions

    def is_terminal(self, history: str) -> bool:
        """Check if terminal."""
        if len(history) == 0:
            return False

        if history[-1] == 'f':
            return True

        if '/' not in history:
            if len(history) >= 2 and history[-1] == 'c':
                return False
        else:
            parts = history.split('/')
            if len(parts[1]) >= 2 and parts[1][-1] == 'c':
                return True

        return False

    def should_deal_board(self, history: str) -> bool:
        """Check if should deal board."""
        if '/' in history:
            return False
        if len(history) < 2:
            return False
        if history[-1] == 'c':
            return True
        return False

    def next_to_act(self, history: str) -> int:
        """Who acts next."""
        if '/' in history:
            round1, round2 = history.split('/')
            total = len(round1) + len(round2)
        else:
            total = len(history)
        return total % 2

    def get_payoff(self, cards: List[str], history: str) -> float:
        """Get payoff for player 0."""
        p0_card = cards[0]
        p1_card = cards[1]
        board = cards[2] if len(cards) > 2 else None

        if history[-1] == 'f':
            if '/' in history:
                round1, round2 = history.split('/')
                total_actions = len(round1) + len(round2)
            else:
                total_actions = len(history)

            last_actor = total_actions % 2
            pot = self._calculate_pot(history[:-1])

            if last_actor == 0:
                return -pot[0]
            else:
                return pot[1]

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
            return 0

    def _calculate_pot(self, history: str) -> List[int]:
        """Calculate pot."""
        invested = [1, 1]

        if '/' in history:
            round1, round2 = history.split('/')
            rounds = [(round1, 2), (round2, 4)]
        else:
            rounds = [(history, 2)]

        for round_history, raise_size in rounds:
            player = 0
            for action in round_history:
                if action == 'r':
                    to_call = invested[1 - player] - invested[player]
                    invested[player] += to_call + raise_size
                elif action == 'c':
                    to_call = invested[1 - player] - invested[player]
                    invested[player] += to_call
                player = 1 - player

        return invested

    def ev_strategy(self, cards: List[str], history: str) -> float:
        """Expected value for player 0 when both players follow strategy."""
        if self.is_terminal(history):
            return self.get_payoff(cards, history)

        board = cards[2] if len(cards) > 2 else None

        if self.should_deal_board(history) and board is None:
            new_history = history + '/'
            remaining = [c for c in self.CARDS if c not in cards[:2]]
            value = 0
            for board_card in remaining:
                new_cards = cards + [board_card]
                value += self.ev_strategy(new_cards, new_history)
            return value / len(remaining)

        acting = self.next_to_act(history)
        card = cards[acting]
        infoset = self.get_infoset(card, board, history)
        actions = self.get_legal_actions(history)

        # Get strategy probabilities
        if infoset in self.strategy:
            probs = self.strategy[infoset]
            # Convert string probs to float
            probs = {a: eval(probs[a]) for a in actions}
        else:
            # Uniform if not in strategy
            probs = {a: 1.0 / len(actions) for a in actions}

        value = 0
        for action in actions:
            action_char = {'fold': 'f', 'call': 'c', 'raise': 'r'}[action]
            new_history = history + action_char
            value += probs[action] * self.ev_strategy(cards, new_history)

        return value

    def best_response_value(self, cards: List[str], history: str, br_player: int) -> float:
        """Best response value for br_player against opponent strategy."""
        if self.is_terminal(history):
            value = self.get_payoff(cards, history)
            return value if br_player == 0 else -value

        board = cards[2] if len(cards) > 2 else None

        if self.should_deal_board(history) and board is None:
            new_history = history + '/'
            remaining = [c for c in self.CARDS if c not in cards[:2]]
            value = 0
            for board_card in remaining:
                new_cards = cards + [board_card]
                value += self.best_response_value(new_cards, new_history, br_player)
            return value / len(remaining)

        acting = self.next_to_act(history)
        actions = self.get_legal_actions(history)

        if acting == br_player:
            # BR player: maximize
            best = float('-inf')
            for action in actions:
                action_char = {'fold': 'f', 'call': 'c', 'raise': 'r'}[action]
                new_history = history + action_char
                value = self.best_response_value(cards, new_history, br_player)
                best = max(best, value)
            return best
        else:
            # Opponent: follow strategy
            card = cards[acting]
            infoset = self.get_infoset(card, board, history)

            if infoset in self.strategy:
                probs = self.strategy[infoset]
                probs = {a: eval(probs[a]) for a in actions}
            else:
                probs = {a: 1.0 / len(actions) for a in actions}

            value = 0
            for action in actions:
                action_char = {'fold': 'f', 'call': 'c', 'raise': 'r'}[action]
                new_history = history + action_char
                value += probs[action] * self.best_response_value(cards, new_history, br_player)
            return value

    def evaluate(self):
        """Evaluate strategy: compute game value and exploitability."""
        print("Evaluating strategy...")

        # Compute game value (EV when both follow strategy)
        total_value = 0
        count = 0

        for i, p0_card in enumerate(self.CARDS):
            for j, p1_card in enumerate(self.CARDS):
                if i != j:
                    cards = [p0_card, p1_card]
                    value = self.ev_strategy(cards, '')
                    total_value += value
                    count += 1

        game_value = total_value / count
        print(f"Game value (player 0): {game_value}")

        # Compute best response values
        br0_total = 0
        br1_total = 0

        for i, p0_card in enumerate(self.CARDS):
            for j, p1_card in enumerate(self.CARDS):
                if i != j:
                    cards = [p0_card, p1_card]
                    br0_total += self.best_response_value(cards, '', 0)
                    br1_total += self.best_response_value(cards, '', 1)

        br0_value = br0_total / count
        br1_value = br1_total / count

        print(f"Player 0 best response value: {br0_value}")
        print(f"Player 1 best response value: {br1_value}")

        exploitability = (br0_value + br1_value) / 2
        print(f"Exploitability: {exploitability}")

        return {
            'game_value': game_value,
            'exploitability': exploitability,
            'br0_value': br0_value,
            'br1_value': br1_value
        }


def main():
    with open('strategy.json', 'r') as f:
        data = json.load(f)

    strategy = data['probs']
    evaluator = LeducEvaluator(strategy)
    results = evaluator.evaluate()

    print("\n" + "="*50)
    print("RESULTS:")
    print(f"  Game value (P0): {results['game_value']:.6f}")
    print(f"  Exploitability: {results['exploitability']:.6f}")
    print("="*50)


if __name__ == '__main__':
    main()
