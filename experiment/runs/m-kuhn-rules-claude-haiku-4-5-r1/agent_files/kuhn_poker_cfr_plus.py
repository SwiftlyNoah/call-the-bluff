import json
from typing import Dict, List


class KuhnPokerCFRPlus:
    def __init__(self):
        self.cards = ['J', 'Q', 'K']
        self.card_values = {'J': 0, 'Q': 1, 'K': 2}
        # Information set -> {action: cumulative regret}
        self.regret_sums = {}
        # Information set -> {action: sum of strategy probabilities}
        self.strategy_sums = {}

    def get_info_set(self, card: str, history: str) -> str:
        """Get information set name from card and history of checks/bets."""
        return f"{card}:{history}"

    def get_available_actions(self, history: str) -> List[str]:
        """Get available actions based on game history."""
        if history == '':
            return ['check', 'bet']
        elif history == 'c':  # P0 checked
            return ['check', 'bet']
        elif history == 'b':  # P0 bet
            return ['fold', 'call']
        elif history == 'cc':  # Both checked
            raise ValueError("Game should be terminal at 'cc'")
        elif history == 'cb':  # P0 checked, P1 bet
            return ['fold', 'call']
        else:
            raise ValueError(f"Unknown history: {history}")

    def get_strategy(self, info_set: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy using regret matching (CFR+)."""
        if info_set not in self.regret_sums:
            self.regret_sums[info_set] = {action: 0.0 for action in actions}
            self.strategy_sums[info_set] = {action: 0.0 for action in actions}

        regrets = self.regret_sums[info_set]
        # CFR+ uses only positive regrets
        positive_regrets = {action: max(regrets[action], 0) for action in actions}
        sum_positive_regrets = sum(positive_regrets.values())

        if sum_positive_regrets > 0:
            strategy = {
                action: positive_regrets[action] / sum_positive_regrets
                for action in actions
            }
        else:
            strategy = {action: 1.0 / len(actions) for action in actions}

        return strategy

    def cfr(self, card0: str, card1: str, history: str, acting_player: int,
            reach_prob_0: float, reach_prob_1: float) -> float:
        """
        CFR+ with reach probabilities.
        Returns utility for player 0.
        """
        # Terminal nodes
        if history == 'cc':
            if self.card_values[card0] > self.card_values[card1]:
                return 1
            else:
                return -1

        if history == 'bf':
            return 1

        if history == 'bk':
            if self.card_values[card0] > self.card_values[card1]:
                return 2
            else:
                return -2

        if history == 'cbf':
            return -1

        if history == 'cbk':
            if self.card_values[card0] > self.card_values[card1]:
                return 2
            else:
                return -2

        # Non-terminal nodes
        if acting_player == 0:
            card = card0
            reach_prob = reach_prob_0
        else:
            card = card1
            reach_prob = reach_prob_1

        actions = self.get_available_actions(history)
        info_set = self.get_info_set(card, history)

        strategy = self.get_strategy(info_set, actions)

        action_to_char = {
            'check': 'c',
            'bet': 'b',
            'fold': 'f',
            'call': 'k'
        }

        # Compute utility for each action
        action_utilities = {}
        for action in actions:
            new_history = history + action_to_char[action]
            if acting_player == 0:
                action_utilities[action] = self.cfr(
                    card0, card1, new_history, 1,
                    reach_prob * strategy[action], reach_prob_1
                )
            else:
                action_utilities[action] = self.cfr(
                    card0, card1, new_history, 0,
                    reach_prob_0, reach_prob * strategy[action]
                )

        # Compute expected utility
        expected_utility = sum(
            strategy[action] * action_utilities[action]
            for action in actions
        )

        # Update regrets and strategy sums (only for the acting player's reach)
        for action in actions:
            regret = action_utilities[action] - expected_utility
            # Weight regret by opponent's reach probability
            if acting_player == 0:
                self.regret_sums[info_set][action] += regret * reach_prob_1
            else:
                self.regret_sums[info_set][action] += regret * reach_prob_0
            self.strategy_sums[info_set][action] += reach_prob * strategy[action]

        return expected_utility

    def solve(self, iterations: int = 10000):
        """Run CFR+ for specified number of iterations."""
        for t in range(iterations):
            for card0 in self.cards:
                for card1 in self.cards:
                    if card0 != card1:
                        self.cfr(card0, card1, '', 0, 1.0, 1.0)

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Compute average strategy from strategy sums."""
        average_strategy = {}
        for info_set, action_sums in self.strategy_sums.items():
            total = sum(action_sums.values())
            if total > 0:
                average_strategy[info_set] = {
                    action: action_sums[action] / total
                    for action in action_sums
                }
            else:
                n_actions = len(action_sums)
                average_strategy[info_set] = {
                    action: 1.0 / n_actions
                    for action in action_sums
                }
        return average_strategy

    def save_strategy(self, filename: str = 'strategy.json'):
        """Save average strategy to JSON file."""
        average_strategy = self.get_average_strategy()

        strategy_data = {
            'game': 'kuhn',
            'strategies': average_strategy
        }

        with open(filename, 'w') as f:
            json.dump(strategy_data, f, indent=2)

        print(f"Strategy saved to {filename}")

    def eval_strategy(self, strategy: Dict[str, Dict[str, float]],
                     card0: str, card1: str, history: str,
                     acting_player: int) -> float:
        """
        Evaluate utility of a strategy pair.
        Returns utility for player 0.
        """
        # Terminal nodes
        if history == 'cc':
            if self.card_values[card0] > self.card_values[card1]:
                return 1
            else:
                return -1

        if history == 'bf':
            return 1

        if history == 'bk':
            if self.card_values[card0] > self.card_values[card1]:
                return 2
            else:
                return -2

        if history == 'cbf':
            return -1

        if history == 'cbk':
            if self.card_values[card0] > self.card_values[card1]:
                return 2
            else:
                return -2

        if acting_player == 0:
            card = card0
        else:
            card = card1

        actions = self.get_available_actions(history)
        info_set = self.get_info_set(card, history)

        if info_set not in strategy:
            uniform = 1.0 / len(actions)
            action_probs = {action: uniform for action in actions}
        else:
            action_probs = strategy[info_set]

        action_to_char = {'check': 'c', 'bet': 'b', 'fold': 'f', 'call': 'k'}

        expected_utility = 0.0
        for action in actions:
            new_history = history + action_to_char[action]
            if acting_player == 0:
                utility = self.eval_strategy(strategy, card0, card1, new_history, 1)
            else:
                utility = self.eval_strategy(strategy, card0, card1, new_history, 0)
            expected_utility += action_probs[action] * utility

        return expected_utility

    def compute_game_value(self, strategy: Dict[str, Dict[str, float]]) -> float:
        """
        Compute expected utility for player 0 when both use the strategy.
        """
        total_utility = 0.0
        hand_count = 0
        for card0 in self.cards:
            for card1 in self.cards:
                if card0 != card1:
                    utility = self.eval_strategy(strategy, card0, card1, '', 0)
                    total_utility += utility
                    hand_count += 1

        return total_utility / hand_count if hand_count > 0 else 0.0


if __name__ == '__main__':
    poker = KuhnPokerCFRPlus()
    print("Running CFR+ solver for Kuhn poker...")
    poker.solve(iterations=10000)

    avg_strategy = poker.get_average_strategy()
    game_value = poker.compute_game_value(avg_strategy)

    poker.save_strategy()

    print(f"\nGame value for P0: {game_value:.6f}")
    print("\nAll information set strategies:")
    for info_set in sorted(avg_strategy.keys()):
        print(f"{info_set}: {avg_strategy[info_set]}")
