import json
import itertools
from typing import Dict, Tuple

class KuhnEvaluator:
    """Evaluate a Kuhn poker strategy."""

    CARDS = ['J', 'Q', 'K']
    CARD_RANK = {'J': 0, 'Q': 1, 'K': 2}

    def __init__(self, strategy_file: str):
        with open(strategy_file, 'r') as f:
            data = json.load(f)
        self.strategy = data['probs']

    def get_action_prob(self, infoset: str, action: str) -> float:
        """Get probability of action at infoset."""
        if infoset not in self.strategy:
            return 0.0
        probs = self.strategy[infoset]
        if action not in probs:
            return 0.0
        return float(probs[action])

    def evaluate_game(self) -> Tuple[float, float, float]:
        """
        Evaluate the value of the game and game value.
        Returns: (game_value_p0, game_value_p1, exploitability)
        """
        total_value_p0 = 0.0
        total_weight = 0.0

        # Evaluate all possible hands with equal weight
        for p0_card, p1_card in itertools.permutations(self.CARDS, 2):
            value = self.evaluate_hand(p0_card, p1_card)
            total_value_p0 += value
            total_weight += 1

        # Average value
        avg_value_p0 = total_value_p0 / total_weight if total_weight > 0 else 0
        avg_value_p1 = -avg_value_p0

        # Compute best response value (approximate exploitability)
        br_value_p0 = self.compute_best_response_value(0)
        br_value_p1 = self.compute_best_response_value(1)
        exploitability = (br_value_p0 + br_value_p1) / 2

        return avg_value_p0, avg_value_p1, exploitability

    def evaluate_hand(self, p0_card: str, p1_card: str,
                     p0_reach: float = 1.0, p1_reach: float = 1.0,
                     history: list = None) -> float:
        """Evaluate a single hand."""
        if history is None:
            history = []

        player = len(history) % 2
        my_card = p0_card if player == 0 else p1_card

        # Terminal check
        if self.is_terminal(history):
            return self.get_payoff(history, p0_card, p1_card)

        # Get actions
        actions = self.get_legal_actions(history)
        infoset = self.get_infoset_id(my_card, history)

        # Compute value
        value = 0.0
        for action in actions:
            prob = self.get_action_prob(infoset, action)
            symbol = {'check': 'c', 'bet': 'b', 'fold': 'f', 'call': 'k'}[action]
            new_history = history + [symbol]

            if player == 0:
                new_p0_reach = p0_reach * prob
                action_value = self.evaluate_hand(p0_card, p1_card, new_p0_reach, p1_reach, new_history)
            else:
                new_p1_reach = p1_reach * prob
                action_value = self.evaluate_hand(p0_card, p1_card, p0_reach, new_p1_reach, new_history)
                action_value = -action_value

            value += prob * action_value

        return value if player == 0 else -value

    def compute_best_response_value(self, br_player: int) -> float:
        """Compute best response value for the given player."""
        total_br_value = 0.0
        total_weight = 0.0

        for p0_card, p1_card in itertools.permutations(self.CARDS, 2):
            br_value = self.compute_br_hand(br_player, p0_card, p1_card)
            total_br_value += br_value
            total_weight += 1

        return total_br_value / total_weight if total_weight > 0 else 0

    def compute_br_hand(self, br_player: int, p0_card: str, p1_card: str,
                       history: list = None) -> float:
        """Compute best response value for a single hand."""
        if history is None:
            history = []

        player = len(history) % 2
        my_card = p0_card if player == 0 else p1_card

        # Terminal check
        if self.is_terminal(history):
            return self.get_payoff(history, p0_card, p1_card)

        # Get actions
        actions = self.get_legal_actions(history)
        infoset = self.get_infoset_id(my_card, history)

        if player == br_player:
            # Maximize for BR player
            best_value = float('-inf')
            for action in actions:
                symbol = {'check': 'c', 'bet': 'b', 'fold': 'f', 'call': 'k'}[action]
                new_history = history + [symbol]
                action_value = self.compute_br_hand(br_player, p0_card, p1_card, new_history)
                if player == 1:
                    action_value = -action_value
                best_value = max(best_value, action_value)
            return best_value if player == 0 else -best_value
        else:
            # Play according to strategy for non-BR player
            value = 0.0
            for action in actions:
                prob = self.get_action_prob(infoset, action)
                symbol = {'check': 'c', 'bet': 'b', 'fold': 'f', 'call': 'k'}[action]
                new_history = history + [symbol]
                action_value = self.compute_br_hand(br_player, p0_card, p1_card, new_history)
                if player == 1:
                    action_value = -action_value
                value += prob * action_value
            return value if player == 0 else -value

    def is_terminal(self, history: list) -> bool:
        """Check if terminal."""
        if not history:
            return False
        if history[-1] == 'f':
            return True
        if history[-1] == 'k':
            return True
        if len(history) >= 2 and history[-1] == 'c' and history[-2] == 'c':
            return True
        return False

    def get_legal_actions(self, history: list) -> list:
        """Get legal actions."""
        if not history:
            return ['check', 'bet']
        last = history[-1]
        if last == 'c':
            return ['check', 'bet']
        if last == 'b':
            return ['fold', 'call']
        return []

    def get_payoff(self, history: list, p0_card: str, p1_card: str) -> float:
        """Get payoff for P0."""
        p0_total = 1
        p1_total = 1
        for i, action in enumerate(history):
            player = i % 2
            if action in ['b', 'k']:
                if player == 0:
                    p0_total += 1
                else:
                    p1_total += 1

        if history[-1] == 'f':
            if len(history) % 2 == 1:
                return -p0_total
            else:
                return p1_total

        if self.CARD_RANK[p0_card] > self.CARD_RANK[p1_card]:
            return p1_total
        else:
            return -p0_total

    def get_infoset_id(self, card: str, history: list) -> str:
        """Get infoset ID."""
        filtered = ''.join([a for a in history if a in ['c', 'b']])
        return f"{card}:{filtered}"


if __name__ == '__main__':
    print("Evaluating strategy...")
    evaluator = KuhnEvaluator('strategy.json')

    game_value_p0, game_value_p1, exploitability = evaluator.evaluate_game()

    print(f"\nGame value for Player 0: {game_value_p0:.6f}")
    print(f"Game value for Player 1: {game_value_p1:.6f}")
    print(f"Exploitability (approximation): {exploitability:.6f}")

    # Check if it's close to a known Nash equilibrium value
    # The exact Nash equilibrium value of Kuhn poker is -1/18 ≈ -0.055556 for P0
    print(f"\nExpected Nash value for P0: -0.055556")
    print(f"Difference from expected: {abs(game_value_p0 - (-1/18)):.6f}")

    if abs(exploitability) < 0.01:
        print("✓ Strategy appears to be well-converged!")
    else:
        print("Strategy may need more iterations for convergence")
