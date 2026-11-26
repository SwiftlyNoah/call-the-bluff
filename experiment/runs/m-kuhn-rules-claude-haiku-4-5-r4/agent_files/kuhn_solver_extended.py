import json
import itertools
from typing import Dict, List, Tuple

class KuhnPoker:
    """Kuhn poker solver using counterfactual regret minimization (CFR)."""

    CARDS = ['J', 'Q', 'K']
    CARD_RANK = {'J': 0, 'Q': 1, 'K': 2}

    def __init__(self):
        self.infosets: Dict = {}

    def get_infoset_id(self, card: str, history: List[str]) -> str:
        """Get infoset ID given player's card and action history."""
        filtered = ''.join([a for a in history if a in ['c', 'b']])
        return f"{card}:{filtered}"

    def ensure_infoset(self, infoset_id: str, actions: List[str]):
        """Ensure infoset exists in the dictionary."""
        if infoset_id not in self.infosets:
            self.infosets[infoset_id] = {
                'regret_sum': {action: 0.0 for action in actions},
                'strategy_sum': {action: 0.0 for action in actions},
            }

    def get_legal_actions(self, history: List[str]) -> List[str]:
        """Get legal actions based on action history."""
        if not history:
            return ['check', 'bet']
        last_action = history[-1]
        if last_action == 'c':
            return ['check', 'bet']
        if last_action == 'b':
            return ['fold', 'call']
        return []

    def is_terminal(self, history: List[str]) -> bool:
        """Check if game state is terminal."""
        if not history:
            return False
        if history[-1] == 'f':
            return True
        if history[-1] == 'k':
            return True
        if len(history) >= 2 and history[-1] == 'c' and history[-2] == 'c':
            return True
        return False

    def get_payoff(self, history: List[str], p0_card: str, p1_card: str) -> float:
        """Get payoff for P0 at terminal state."""
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

    def get_strategy(self, infoset_id: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy using regret matching."""
        info = self.infosets[infoset_id]
        strategy = {}
        sum_positive_regret = 0.0

        for action in actions:
            regret = max(0, info['regret_sum'][action])
            strategy[action] = regret
            sum_positive_regret += regret

        if sum_positive_regret > 0:
            for action in actions:
                strategy[action] /= sum_positive_regret
        else:
            prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = prob

        return strategy

    def traverse(self, p0_card: str, p1_card: str, history: List[str],
                 p0_reach: float, p1_reach: float) -> float:
        """Traverse game tree using CFR."""
        player = len(history) % 2
        my_card = p0_card if player == 0 else p1_card

        if self.is_terminal(history):
            return self.get_payoff(history, p0_card, p1_card)

        actions = self.get_legal_actions(history)
        infoset_id = self.get_infoset_id(my_card, history)
        self.ensure_infoset(infoset_id, actions)

        strategy = self.get_strategy(infoset_id, actions)

        node_value = 0.0
        action_values = {}

        action_to_symbol = {
            'check': 'c',
            'bet': 'b',
            'fold': 'f',
            'call': 'k',
        }

        for action in actions:
            symbol = action_to_symbol[action]
            new_history = history + [symbol]

            if player == 0:
                new_p0_reach = p0_reach * strategy[action]
                action_value = self.traverse(p0_card, p1_card, new_history,
                                            new_p0_reach, p1_reach)
            else:
                new_p1_reach = p1_reach * strategy[action]
                action_value = self.traverse(p0_card, p1_card, new_history,
                                            p0_reach, new_p1_reach)
                action_value = -action_value

            action_values[action] = action_value
            node_value += strategy[action] * action_value

        my_reach = p0_reach if player == 0 else p1_reach
        opp_reach = p1_reach if player == 0 else p0_reach

        for action in actions:
            regret = action_values[action] - node_value
            self.infosets[infoset_id]['regret_sum'][action] += opp_reach * regret
            self.infosets[infoset_id]['strategy_sum'][action] += my_reach * strategy[action]

        return node_value if player == 0 else -node_value

    def run_cfr(self, num_iterations: int):
        """Run CFR for the specified number of iterations."""
        for iteration in range(num_iterations):
            for p0_card, p1_card in itertools.permutations(self.CARDS, 2):
                self.traverse(p0_card, p1_card, [], 1.0, 1.0)

            if (iteration + 1) % 2000 == 0:
                print(f"Completed iteration {iteration + 1}/{num_iterations}")

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Compute average strategy from strategy sums."""
        avg_strategy = {}

        for infoset_id, info in self.infosets.items():
            strategy = {}
            sum_strategy = sum(info['strategy_sum'].values())

            if sum_strategy > 0:
                for action, sum_prob in info['strategy_sum'].items():
                    strategy[action] = sum_prob / sum_strategy
            else:
                actions = list(info['strategy_sum'].keys())
                prob = 1.0 / len(actions)
                for action in actions:
                    strategy[action] = prob

            avg_strategy[infoset_id] = strategy

        return avg_strategy

    def to_strategy_json(self) -> Dict:
        """Convert to strategy.json format."""
        avg_strategy = self.get_average_strategy()

        probs = {}
        for infoset_id, strategy in avg_strategy.items():
            probs[infoset_id] = {
                action: str(prob) for action, prob in strategy.items()
            }

        return {
            'format': 'bluff-strategy-v1',
            'game': 'kuhn',
            'producer': 'CFR (Counterfactual Regret Minimization) - 50K iterations',
            'probs': probs
        }


if __name__ == '__main__':
    print("Starting Kuhn poker CFR solver with 50,000 iterations...")
    poker = KuhnPoker()

    poker.run_cfr(50000)
    print("CFR training completed.")

    strategy = poker.to_strategy_json()
    with open('strategy.json', 'w') as f:
        json.dump(strategy, f, indent=2)

    print("Strategy written to strategy.json")
