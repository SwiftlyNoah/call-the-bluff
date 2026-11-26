import json
from collections import defaultdict

class KuhnPokerCFR:
    def __init__(self):
        self.cards = ['J', 'Q', 'K']
        self.card_rank = {'J': 0, 'Q': 1, 'K': 2}
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

    def whose_turn(self, history):
        """Determine whose turn based on action count parity"""
        return len(history) % 2

    def get_legal_actions(self, history):
        """Get legal actions based on game history"""
        if not history:
            return ['check', 'bet']
        elif history == 'c':
            return ['check', 'bet']
        elif history == 'cb':
            return ['fold', 'call']
        elif history == 'b':
            return ['fold', 'call']
        else:
            return []

    def is_terminal(self, history):
        """Check if a game state is terminal"""
        return history in ['cc', 'cbf', 'cbx', 'bf', 'bx']

    def evaluate_terminal(self, card0, card1, history):
        """Evaluate a terminal node, return value for player 0"""
        winner = 1 if self.card_rank[card0] > self.card_rank[card1] else -1

        if history == 'cc':
            # Both checked: showdown with 1 chip each in pot
            return winner
        elif history == 'cbf':
            # P0 checked, P1 bet, P0 folded - P0 loses their ante
            return -1
        elif history == 'cbx':
            # P0 checked, P1 bet, P0 called - showdown with 2 chips each in pot
            return winner * 2
        elif history == 'bf':
            # P0 bet, P1 folded - P0 wins P1's ante
            return 1
        elif history == 'bx':
            # P0 bet, P1 called - showdown with 2 chips each in pot
            return winner * 2
        return 0

    def get_infoset_key(self, card, history_for_infoset):
        """Get information set key"""
        return f"{card}:{history_for_infoset}"

    def _action_to_history_char(self, action):
        """Convert action to its game history character"""
        mapping = {
            'check': 'c',
            'bet': 'b',
            'fold': 'f',
            'call': 'x'
        }
        return mapping.get(action, '')

    def _history_to_infoset_history(self, history):
        """Convert full game history to infoset history (only check and bet)"""
        infoset_history = ''
        for char in history:
            if char in ['c', 'b']:
                infoset_history += char
        return infoset_history

    def _regret_match_strategy(self, infoset, legal_actions):
        """Compute strategy using regret matching"""
        strategy = {}
        total_positive_regret = 0.0

        for action in legal_actions:
            regret = max(self.regrets[infoset].get(action, 0.0), 0.0)
            total_positive_regret += regret

        if total_positive_regret > 0:
            for action in legal_actions:
                strategy[action] = max(self.regrets[infoset].get(action, 0.0), 0.0) / total_positive_regret
        else:
            uniform_prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                strategy[action] = uniform_prob

        return strategy

    def cfr(self, card0, card1, history, player, prob0, prob1):
        """
        Counterfactual Regret Minimization traversal.

        Returns value of the game from player 0's perspective.
        """
        if self.is_terminal(history):
            return self.evaluate_terminal(card0, card1, history)

        player_card = card0 if player == 0 else card1
        infoset_history = self._history_to_infoset_history(history)
        infoset_key = self.get_infoset_key(player_card, infoset_history)

        legal_actions = self.get_legal_actions(history)

        # Initialize infoset if needed
        for action in legal_actions:
            if action not in self.regrets[infoset_key]:
                self.regrets[infoset_key][action] = 0.0
                self.strategy_sum[infoset_key][action] = 0.0

        # Get strategy via regret matching
        strategy = self._regret_match_strategy(infoset_key, legal_actions)

        # Update strategy sum (weighted by reach probability)
        reach_prob = prob0 if player == 0 else prob1
        for action in legal_actions:
            self.strategy_sum[infoset_key][action] += reach_prob * strategy[action]

        # Compute values for each action
        node_value = 0.0
        values = {}

        for action in legal_actions:
            next_history = history + self._action_to_history_char(action)

            if player == 0:
                new_prob0 = prob0 * strategy[action]
                new_prob1 = prob1
                value = self.cfr(card0, card1, next_history, 1, new_prob0, new_prob1)
            else:
                new_prob0 = prob0
                new_prob1 = prob1 * strategy[action]
                value = self.cfr(card0, card1, next_history, 0, new_prob0, new_prob1)

            values[action] = value
            node_value += strategy[action] * value

        # Update regrets (weighted by opponent reach probability)
        opp_reach = prob1 if player == 0 else prob0
        for action in legal_actions:
            regret = values[action] - node_value
            self.regrets[infoset_key][action] += opp_reach * regret

        return node_value

    def train(self, iterations):
        """Train the CFR algorithm"""
        for iteration in range(iterations):
            for card0 in self.cards:
                for card1 in self.cards:
                    if card0 != card1:
                        self.cfr(card0, card1, '', 0, 1.0, 1.0)

            if (iteration + 1) % 2000 == 0:
                print(f"  Iteration {iteration + 1}/{iterations}")

    def get_average_strategy(self):
        """Compute the average strategy"""
        avg_strategy = {}

        for infoset in self.strategy_sum:
            avg_strategy[infoset] = {}
            total = sum(self.strategy_sum[infoset].values())

            for action in self.strategy_sum[infoset]:
                if total > 0:
                    avg_strategy[infoset][action] = self.strategy_sum[infoset][action] / total
                else:
                    n_actions = len(self.strategy_sum[infoset])
                    avg_strategy[infoset][action] = 1.0 / n_actions

        return avg_strategy

    def get_exploitability(self):
        """Estimate exploitability (lower regret)"""
        total_regret = 0.0
        for infoset in self.regrets:
            for action in self.regrets[infoset]:
                total_regret += max(self.regrets[infoset][action], 0.0)
        return total_regret / 6.0  # Normalize by number of card deals

# Main
print("Training Kuhn Poker CFR solver...")
solver = KuhnPokerCFR()
solver.train(50000)

print("\nComputing average strategy...")
strategy = solver.get_average_strategy()

# Format output according to STRATEGY_FORMAT.md
probs = {}
for infoset, actions in strategy.items():
    probs[infoset] = {action: str(prob) for action, prob in actions.items()}

output = {
    'format': 'bluff-strategy-v1',
    'game': 'kuhn',
    'producer': 'CFR (Counterfactual Regret Minimization) solver, 50000 iterations',
    'probs': probs
}

# Write to file
with open('strategy.json', 'w') as f:
    json.dump(output, f, indent=2, sort_keys=True)

print("Strategy written to strategy.json")
print("\nStrategy summary:")
for infoset in sorted(strategy.keys()):
    print(f"  {infoset}: {strategy[infoset]}")
