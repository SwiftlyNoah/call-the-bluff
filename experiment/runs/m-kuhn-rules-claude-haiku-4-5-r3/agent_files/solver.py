#!/usr/bin/env python3
"""Kuhn poker solver using Counterfactual Regret Minimization."""

import json
from collections import defaultdict
from typing import Dict, Tuple

CARDS = ['J', 'Q', 'K']
CARD_RANK = {'J': 0, 'Q': 1, 'K': 2}


class KuhnPokerCFR:
    def __init__(self):
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.node_map = {}

    def _action_to_char(self, action: str) -> str:
        """Convert action name to history character."""
        if action in ['check', 'call']:
            return 'c'
        elif action == 'bet':
            return 'b'
        elif action == 'fold':
            return 'f'
        raise ValueError(f"Unknown action: {action}")

    def _get_actions(self, history: str) -> list:
        """Get legal actions based on history."""
        if not history:
            return ['check', 'bet']

        acting_player = len(history) % 2
        last = history[-1] if history else ''

        if 'b' not in history:
            # No bets yet - can check or bet
            return ['check', 'bet']

        # Has at least one bet
        if history.count('b') == 1:
            if last == 'b':
                # Facing a bet - fold or call
                return ['fold', 'call']
            else:  # last == 'c'
                # Other player checked after we checked
                return ['check', 'bet']

        if history.count('b') == 2:
            if last == 'b':
                # Facing another bet
                return ['fold', 'call']

        return []

    def _get_info_set(self, card: str, history: str) -> str:
        """Get information set name."""
        return f"{card}:{history}"

    def _is_terminal(self, history: str) -> Tuple[bool, float]:
        """Check if state is terminal and return payoff for P0 if so."""
        if history.endswith('cc'):
            # Both checked - no more betting
            return (True, 0)  # Showdown, determined by cards
        elif history.endswith('bc'):
            # P0 bet, P1 called - showdown
            return (True, 0)
        elif history.endswith('cbc'):
            # P0 checked, P1 bet, P0 called - showdown
            return (True, 0)
        elif history.endswith('bf'):
            # P0 bet, P1 folded
            return (True, 1)
        elif history.endswith('cbf'):
            # P0 checked, P1 bet, P0 folded
            return (True, -1)

        return (False, 0)

    def _get_showdown_payoff(self, card_p0: str, card_p1: str,
                             history: str) -> float:
        """Get payoff for P0 at showdown."""
        # Determine pot size from history
        num_bets = history.count('b')

        if CARD_RANK[card_p0] > CARD_RANK[card_p1]:
            # P0 wins
            if num_bets == 0:  # Both checked (cc)
                return 1.0
            else:  # At least one bet (bc or cbc)
                return 2.0
        else:
            # P1 wins
            if num_bets == 0:
                return -1.0
            else:
                return -2.0

    def cfr(self, card_p0: str, card_p1: str, history: str,
            probs: Tuple[float, float], target_player: int) -> float:
        """
        Compute counterfactual value for target_player.
        probs = (prob_p0_reach, prob_p1_reach)
        Returns value from target_player's perspective.
        """
        p0_reach, p1_reach = probs

        # Check terminal states
        is_term, fold_payoff = self._is_terminal(history)
        if is_term:
            if history.endswith('f'):
                # Fold - return fold payoff
                return fold_payoff if target_player == 0 else -fold_payoff
            else:
                # Showdown
                payoff_p0 = self._get_showdown_payoff(card_p0, card_p1, history)
                return payoff_p0 if target_player == 0 else -payoff_p0

        # Determine acting player
        acting = len(history) % 2  # 0 for P0, 1 for P1
        acting_card = card_p0 if acting == 0 else card_p1

        # Get information set and actions
        info_set = self._get_info_set(acting_card, history)
        actions = self._get_actions(history)

        # Get current strategy
        strategy = self._get_strategy(info_set, actions)

        # Compute values for each action
        values = {}
        node_value = 0.0

        for action in actions:
            new_history = history + self._action_to_char(action)

            # Update reaching probabilities
            if acting == 0:
                new_probs = (p0_reach * strategy[action], p1_reach)
            else:
                new_probs = (p0_reach, p1_reach * strategy[action])

            # Recurse
            child_value = self.cfr(card_p0, card_p1, new_history, new_probs, target_player)
            values[action] = child_value
            node_value += strategy[action] * child_value

        # Update regrets and strategy sum if this is target player's turn
        if acting == target_player:
            opp_reach = p1_reach if acting == 0 else p0_reach
            for action in actions:
                regret = opp_reach * (values[action] - node_value)
                self.regrets[info_set][action] += regret

            # Update strategy sum
            reach_prob = p0_reach if acting == 0 else p1_reach
            for action in actions:
                self.strategy_sum[info_set][action] += reach_prob * strategy[action]

        return node_value

    def _get_strategy(self, info_set: str, actions: list) -> dict:
        """Get strategy using regret matching."""
        strategy = {}
        regrets_sum = self.regrets[info_set]

        # Regret matching
        pos_regret_sum = sum(max(0, regrets_sum[a]) for a in actions)

        if pos_regret_sum > 0:
            for action in actions:
                strategy[action] = max(0, regrets_sum[action]) / pos_regret_sum
        else:
            # Uniform if no positive regrets
            prob = 1.0 / len(actions)
            for action in actions:
                strategy[action] = prob

        return strategy

    def solve(self, iterations: int):
        """Run CFR."""
        for it in range(iterations):
            if it % 100000 == 0 and it > 0:
                print(f"  Iteration {it}")

            # For each player
            for player in range(2):
                # For each deal
                for card_p0 in CARDS:
                    for card_p1 in CARDS:
                        if card_p0 != card_p1:
                            self.cfr(card_p0, card_p1, '', (1.0, 1.0), player)

    def get_strategy(self) -> dict:
        """Extract average strategy."""
        strategy = {}

        for info_set, action_sums in self.strategy_sum.items():
            card = info_set[0]
            history = info_set[2:]  # Remove "X:"
            actions = self._get_actions(history)

            total = sum(action_sums[a] for a in actions)
            strategy[info_set] = {}

            if total > 0:
                for action in actions:
                    prob = action_sums[action] / total
                    strategy[info_set][action] = repr(prob)
            else:
                # Uniform fallback
                for action in actions:
                    strategy[info_set][action] = repr(1.0 / len(actions))

        return strategy


def main():
    solver = KuhnPokerCFR()
    print("Running CFR for 1000000 iterations...")
    solver.solve(1000000)

    strategy = solver.get_strategy()

    output = {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": "CFR solver",
        "probs": strategy
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nWrote {len(strategy)} information sets")

    # Show summary
    for card in ['J', 'Q', 'K']:
        info_set = f"{card}:"
        if info_set in strategy:
            probs = strategy[info_set]
            print(f"{card}: check={float(probs['check']):.3f}, bet={float(probs['bet']):.3f}")


if __name__ == '__main__':
    main()
