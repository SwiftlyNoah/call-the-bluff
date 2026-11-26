#!/usr/bin/env python3
"""Leduc hold'em solver using Counterfactual Regret Minimization."""

import json
import random
from typing import Dict, List
from itertools import combinations
from collections import defaultdict


RANKS = ['J', 'Q', 'K']
SUITS = ['s', 'h']
DECK = [f'{r}{s}' for r in RANKS for s in SUITS]


class GameState:
    """Represents a Leduc hold'em game state."""

    def __init__(self, p0_card: str, p1_card: str, public_card: str):
        self.p0_card = p0_card
        self.p1_card = p1_card
        self.public_card = public_card

        # Game progression
        self.round_num = 0
        self.r1_actions = []
        self.r2_actions = []

        # Current state
        self.player_to_move = 0
        self.bets_this_round = [0, 0]
        self.pot = 2  # Both antes
        self.total_contribution = [1, 1]  # Track total chips contributed
        self.raises_this_round = 0

    def copy(self) -> 'GameState':
        """Deep copy the state."""
        new_state = GameState(self.p0_card, self.p1_card, self.public_card)
        new_state.round_num = self.round_num
        new_state.r1_actions = self.r1_actions.copy()
        new_state.r2_actions = self.r2_actions.copy()
        new_state.player_to_move = self.player_to_move
        new_state.bets_this_round = self.bets_this_round.copy()
        new_state.pot = self.pot
        new_state.total_contribution = self.total_contribution.copy()
        new_state.raises_this_round = self.raises_this_round
        return new_state

    def is_terminal(self) -> bool:
        """Check if the game is over."""
        all_actions = self.r1_actions + self.r2_actions
        if 'f' in all_actions:
            return True

        if self.round_num == 1 and self._round_complete():
            return True

        return False

    def _round_complete(self) -> bool:
        """Check if the current round is complete."""
        actions = self.r2_actions if self.round_num == 1 else self.r1_actions

        if len(actions) == 0:
            return False

        return self.bets_this_round[0] == self.bets_this_round[1]

    def get_legal_actions(self) -> List[str]:
        """Get legal actions for the current player."""
        actions = ['c']

        if self.bets_this_round[self.player_to_move] < self.bets_this_round[1 - self.player_to_move]:
            actions.append('f')

        if self.raises_this_round < 2:
            actions.append('r')

        return actions

    def apply_action(self, action: str) -> 'GameState':
        """Apply an action and return new state."""
        new_state = self.copy()

        if new_state.round_num == 0:
            new_state.r1_actions.append(action)
        else:
            new_state.r2_actions.append(action)

        raise_size = 2 if new_state.round_num == 0 else 4
        player = new_state.player_to_move

        if action == 'c':
            opponent_bet = new_state.bets_this_round[1 - player]
            my_current_bet = new_state.bets_this_round[player]
            amount_to_bet = opponent_bet - my_current_bet
            new_state.pot += amount_to_bet
            new_state.total_contribution[player] += amount_to_bet
            new_state.bets_this_round[player] = opponent_bet

        elif action == 'r':
            opponent_bet = new_state.bets_this_round[1 - player]
            my_current_bet = new_state.bets_this_round[player]
            new_bet_level = opponent_bet + raise_size
            amount_to_bet = new_bet_level - my_current_bet
            new_state.pot += amount_to_bet
            new_state.total_contribution[player] += amount_to_bet
            new_state.bets_this_round[player] = new_bet_level
            new_state.raises_this_round += 1

        elif action == 'f':
            pass

        new_state.player_to_move = 1 - player

        if action == 'c' and new_state._round_complete():
            if new_state.round_num == 0:
                new_state.round_num = 1
                new_state.bets_this_round = [0, 0]
                new_state.raises_this_round = 0
                new_state.player_to_move = 0

        return new_state

    def get_info_set_name(self) -> str:
        """Get the information set name for the current player."""
        player = self.player_to_move
        my_card = self.p0_card if player == 0 else self.p1_card

        if self.round_num == 0:
            actions_str = ''.join(self.r1_actions)
            return f"{my_card}:{actions_str}"
        else:
            r1_str = ''.join(self.r1_actions)
            r2_str = ''.join(self.r2_actions)
            return f"{my_card}|{self.public_card}:{r1_str}/{r2_str}"

    def get_payoff_p0(self) -> float:
        """Get payoff for player 0."""
        all_actions = self.r1_actions + self.r2_actions
        if 'f' in all_actions:
            # Find who folded and determine winner
            if 'f' in self.r1_actions:
                fold_idx = self.r1_actions.index('f')
                p0_folded = (fold_idx % 2 == 0)
            else:
                fold_idx = self.r2_actions.index('f')
                p0_folded = (fold_idx % 2 == 0)

            if p0_folded:
                return -(self.pot - self.total_contribution[1])
            else:
                return self.pot - self.total_contribution[0]

        # Showdown
        p0_rank_idx = RANKS.index(self.p0_card[0])
        p1_rank_idx = RANKS.index(self.p1_card[0])
        pub_rank_idx = RANKS.index(self.public_card[0])

        if p0_rank_idx == pub_rank_idx and p1_rank_idx != pub_rank_idx:
            winner = 0
        elif p1_rank_idx == pub_rank_idx and p0_rank_idx != pub_rank_idx:
            winner = 1
        elif p0_rank_idx == pub_rank_idx and p1_rank_idx == pub_rank_idx:
            winner = None
        else:
            winner = 0 if p0_rank_idx > p1_rank_idx else 1

        if winner is None:
            return 0
        elif winner == 0:
            return self.pot - self.total_contribution[0]
        else:
            return -(self.pot - self.total_contribution[1])


class CFRSolver:
    """CFR solver for Leduc hold'em."""

    def __init__(self):
        self.regrets: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.strategy_sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def get_strategy(self, info_set: str, actions: List[str]) -> Dict[str, float]:
        """Get regret-matched strategy."""
        strategy = {}
        sum_positive_regret = 0.0

        for action in actions:
            regret = max(0.0, self.regrets[info_set].get(action, 0.0))
            strategy[action] = regret
            sum_positive_regret += regret

        if sum_positive_regret <= 0:
            prob = 1.0 / len(actions)
            return {a: prob for a in actions}

        return {a: strategy[a] / sum_positive_regret for a in actions}

    def cfr(self, state: GameState, reach_p0: float, reach_p1: float) -> float:
        """Recursive CFR. Returns payoff for player 0."""

        if state.is_terminal():
            return state.get_payoff_p0()

        player = state.player_to_move
        info_set = state.get_info_set_name()
        actions = state.get_legal_actions()

        strategy = self.get_strategy(info_set, actions)

        for action in actions:
            self.strategy_sum[info_set][action] += strategy[action]

        node_values = {}
        node_value = 0.0

        for action in actions:
            next_state = state.apply_action(action)

            if player == 0:
                new_reach_p0 = reach_p0 * strategy[action]
                new_reach_p1 = reach_p1
            else:
                new_reach_p0 = reach_p0
                new_reach_p1 = reach_p1 * strategy[action]

            value = self.cfr(next_state, new_reach_p0, new_reach_p1)
            node_values[action] = value
            node_value += strategy[action] * value

        for action in actions:
            regret = node_values[action] - node_value
            if player == 0:
                self.regrets[info_set][action] += reach_p1 * regret
            else:
                self.regrets[info_set][action] += reach_p0 * regret

        return node_value

    def solve(self, iterations: int):
        """Run CFR for specified iterations."""
        total_value = 0.0
        hand_combos = list(combinations(DECK, 2))

        for iteration in range(iterations):
            if (iteration + 1) % 500 == 0:
                print(f"Iteration {iteration + 1}/{iterations}")

            # Random hand and board
            p0_card, p1_card = random.choice(hand_combos)
            remaining = [c for c in DECK if c not in (p0_card, p1_card)]
            pub_card = random.choice(remaining)

            state = GameState(p0_card, p1_card, pub_card)
            value = self.cfr(state, 1.0, 1.0)
            total_value += value

        avg_value = total_value / iterations
        print(f"Average value for player 0: {avg_value}")

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get average strategy."""
        strategies = {}

        for info_set, action_dict in self.strategy_sum.items():
            total = sum(action_dict.values())
            if total > 0:
                strategies[info_set] = {a: v / total for a, v in action_dict.items()}
            else:
                actions = list(action_dict.keys())
                strategies[info_set] = {a: 1.0 / len(actions) for a in actions}

        return strategies

    def save_strategy(self, filename: str):
        """Save strategy to JSON."""
        strategies = self.get_average_strategy()

        output = {
            'game': 'leduc',
            'strategies': strategies
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Strategy saved to {filename}")


def main():
    solver = CFRSolver()
    solver.solve(iterations=10000)
    solver.save_strategy('strategy.json')


if __name__ == '__main__':
    main()
