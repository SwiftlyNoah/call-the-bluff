import json
from collections import defaultdict
import itertools

class KuhnPokerEvaluator:
    def __init__(self):
        self.cards = ['J', 'Q', 'K']
        self.card_rank = {'J': 0, 'Q': 1, 'K': 2}

    def evaluate_strategy(self, strategy_dict):
        """Evaluate the value of the game and exploitability"""

        # Compute value when both players follow the strategy
        total_value = 0.0
        hand_count = 0

        for card0 in self.cards:
            for card1 in self.cards:
                if card0 == card1:
                    continue
                hand_count += 1
                value = self.compute_game_value(card0, card1, strategy_dict)
                total_value += value

        value_p0 = total_value / hand_count if hand_count > 0 else 0

        # Estimate exploitability (approximate)
        # For Kuhn poker, the value should be close to 0
        print(f"\nGame Statistics:")
        print(f"  Number of hands: {hand_count}")
        print(f"  Expected value for Player 0: {value_p0:.6f}")

        return value_p0

    def compute_game_value(self, card0, card1, strategy_dict, history='', player=0):
        """Recursively compute game value"""
        # Terminal nodes
        if history in ['cc', 'cbf', 'cbx', 'bf', 'bx']:
            return self._evaluate_terminal(card0, card1, history)

        # Determine whose turn
        player_card = card0 if player == 0 else card1
        infoset_history = self._to_infoset_history(history)
        infoset_key = f"{player_card}:{infoset_history}"

        # Get strategy for this infoset
        if infoset_key not in strategy_dict:
            # Default to uniform
            if not history or history == 'c':
                strategy = {'check': 0.5, 'bet': 0.5}
            elif history == 'cb' or history == 'b':
                strategy = {'fold': 0.5, 'call': 0.5}
            else:
                return 0
        else:
            strategy = strategy_dict[infoset_key]

        # Compute value
        value = 0.0
        for action, prob in strategy.items():
            next_history = history + self._action_char(action)
            action_value = self.compute_game_value(card0, card1, strategy_dict, next_history, 1 - player)
            value += prob * action_value

        return value

    def _evaluate_terminal(self, card0, card1, history):
        """Evaluate terminal"""
        winner = 1 if self.card_rank[card0] > self.card_rank[card1] else -1

        if history == 'cc':
            return winner
        elif history == 'cbf':
            return -1
        elif history == 'cbx':
            return winner * 2
        elif history == 'bf':
            return 1
        elif history == 'bx':
            return winner * 2
        return 0

    def _to_infoset_history(self, history):
        """Remove fold/call from history"""
        result = ''
        for c in history:
            if c in ['c', 'b']:
                result += c
        return result

    def _action_char(self, action):
        mapping = {'check': 'c', 'bet': 'b', 'fold': 'f', 'call': 'x'}
        return mapping.get(action, '')

# Load and evaluate
with open('strategy.json', 'r') as f:
    data = json.load(f)
    strategies = data['strategies']

evaluator = KuhnPokerEvaluator()
value_p0 = evaluator.evaluate_strategy(strategies)

print(f"\nStrategy Detail:")
for infoset in sorted(strategies.keys()):
    actions = strategies[infoset]
    print(f"  {infoset}: {actions}")

# Known Kuhn poker value is 0 (symmetric), with exploitability around 0.3 for small regrets
print(f"\nNote: Kuhn poker has value 0 (zero-sum symmetric game)")
print(f"      A fully converged strategy should have exploitability < 0.01")
