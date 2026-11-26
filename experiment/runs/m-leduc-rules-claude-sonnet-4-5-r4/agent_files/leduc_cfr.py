#!/usr/bin/env python3
"""
Leduc Hold'em CFR Solver
"""

import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import itertools


class LeducState:
    """Represents a state in Leduc Hold'em."""

    def __init__(self):
        self.deck = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
        self.players_cards = [None, None]  # Private cards for each player
        self.public_card = None
        self.pot = [1, 1]  # Ante of 1 each
        self.bets = [0, 0]  # Current round bets
        self.round = 1  # 1 or 2
        self.history = ""  # Action history
        self.acting_player = 0
        self.num_raises = 0

    def deal(self, cards: List[str]):
        """Deal cards (for deterministic setup in CFR)."""
        self.players_cards = [cards[0], cards[1]]
        if len(cards) > 2:
            self.public_card = cards[2]

    def is_terminal(self) -> bool:
        """Check if state is terminal."""
        # Someone folded
        if 'f' in self.history:
            return True

        # Completed both rounds
        if self.round == 2:
            # Round 2 ends with check-check or call after raise
            round2_actions = self.history.split('/')[1] if '/' in self.history else ""
            if len(round2_actions) >= 2:
                if round2_actions[-1] == 'c' and round2_actions[-2] in ['c', 'r']:
                    return True

        return False

    def get_payoff(self, player: int) -> float:
        """Get payoff for player at terminal state."""
        # Fold case
        if 'f' in self.history:
            # Count actions (not '/') to determine who folded
            actions = self.history.replace('/', '')
            folder_pos = actions.index('f')
            folder = folder_pos % 2
            winner = 1 - folder
            total_pot = sum(self.pot) + sum(self.bets)
            if player == winner:
                return self.pot[1 - player] + self.bets[1 - player]
            else:
                return -(self.pot[player] + self.bets[player])

        # Showdown
        p0_card = self.players_cards[0]
        p1_card = self.players_cards[1]
        public = self.public_card

        # Check for pairs
        p0_rank = p0_card[0]
        p1_rank = p1_card[0]
        public_rank = public[0]

        p0_has_pair = (p0_rank == public_rank)
        p1_has_pair = (p1_rank == public_rank)

        rank_order = {'J': 0, 'Q': 1, 'K': 2}

        winner = None
        if p0_has_pair and not p1_has_pair:
            winner = 0
        elif p1_has_pair and not p0_has_pair:
            winner = 1
        elif p0_has_pair and p1_has_pair:
            winner = None  # Both have pairs, split
        else:
            # Compare ranks
            if rank_order[p0_rank] > rank_order[p1_rank]:
                winner = 0
            elif rank_order[p1_rank] > rank_order[p0_rank]:
                winner = 1
            else:
                winner = None  # Tie

        total_pot = sum(self.pot) + sum(self.bets)

        if winner is None:
            return 0.0  # Split pot
        elif player == winner:
            return self.pot[1 - player] + self.bets[1 - player]
        else:
            return -(self.pot[player] + self.bets[player])

    def get_legal_actions(self) -> List[str]:
        """Get legal actions for current player."""
        actions = []

        # Can fold if facing a bet
        if self.bets[self.acting_player] < self.bets[1 - self.acting_player]:
            actions.append('fold')

        # Can always call/check
        actions.append('call')

        # Can raise if not at limit
        if self.num_raises < 2:
            actions.append('raise')

        return actions

    def apply_action(self, action: str) -> 'LeducState':
        """Apply action and return new state."""
        new_state = LeducState()
        new_state.deck = self.deck[:]
        new_state.players_cards = self.players_cards[:]
        new_state.public_card = self.public_card
        new_state.pot = self.pot[:]
        new_state.bets = self.bets[:]
        new_state.round = self.round
        new_state.history = self.history
        new_state.acting_player = self.acting_player
        new_state.num_raises = self.num_raises

        raise_size = 2 if new_state.round == 1 else 4

        if action == 'fold':
            new_state.history += 'f'
        elif action == 'call':
            # Match opponent's bet
            to_call = new_state.bets[1 - new_state.acting_player] - new_state.bets[new_state.acting_player]
            new_state.bets[new_state.acting_player] += to_call
            new_state.history += 'c'
        elif action == 'raise':
            # Match + raise
            to_call = new_state.bets[1 - new_state.acting_player] - new_state.bets[new_state.acting_player]
            new_state.bets[new_state.acting_player] += to_call + raise_size
            new_state.history += 'r'
            new_state.num_raises += 1

        # Check if round should advance
        if not new_state.is_terminal():
            round_actions = new_state.history.split('/')[-1]

            # Round ends on check-check or call after raise
            if len(round_actions) >= 2:
                if round_actions[-1] == 'c' and round_actions[-2] in ['c', 'r']:
                    # Round complete
                    if new_state.round == 1:
                        new_state.round = 2
                        new_state.pot[0] += new_state.bets[0]
                        new_state.pot[1] += new_state.bets[1]
                        new_state.bets = [0, 0]
                        new_state.num_raises = 0
                        new_state.history += '/'
                        new_state.acting_player = 0
                        return new_state

            # Next player acts
            new_state.acting_player = 1 - new_state.acting_player

        return new_state

    def get_infoset_key(self, player: int) -> str:
        """Get information set key for player."""
        card = self.players_cards[player]

        if self.round == 1:
            return f"{card}:{self.history}"
        else:
            parts = self.history.split('/')
            round1 = parts[0]
            round2 = parts[1] if len(parts) > 1 else ""
            return f"{card}|{self.public_card}:{round1}/{round2}"


class CFRSolver:
    """CFR+ solver for Leduc Hold'em."""

    def __init__(self):
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def get_strategy(self, infoset: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy for infoset using regret matching."""
        strategy = {}
        regrets = self.regret_sum[infoset]

        # Use positive regrets only
        positive_regret_sum = sum(max(0, regrets.get(a, 0)) for a in actions)

        for action in actions:
            if positive_regret_sum > 0:
                strategy[action] = max(0, regrets.get(action, 0)) / positive_regret_sum
            else:
                strategy[action] = 1.0 / len(actions)

        return strategy

    def cfr(self, state: LeducState, reach_probs: List[float]) -> float:
        """CFR recursive algorithm."""
        player = state.acting_player

        if state.is_terminal():
            return state.get_payoff(player)

        infoset = state.get_infoset_key(player)
        actions = state.get_legal_actions()
        strategy = self.get_strategy(infoset, actions)

        # Compute action utilities
        action_utils = {}
        for action in actions:
            next_state = state.apply_action(action)
            # Update reach probabilities for next state
            new_reach = reach_probs[:]
            new_reach[player] *= strategy[action]
            action_utils[action] = -self.cfr(next_state, new_reach)

        # Node utility
        util = sum(strategy[a] * action_utils[a] for a in actions)

        # Update regrets and strategy sum
        for action in actions:
            regret = action_utils[action] - util
            self.regret_sum[infoset][action] += reach_probs[1 - player] * regret

            # CFR+: floor regrets at 0
            if self.regret_sum[infoset][action] < 0:
                self.regret_sum[infoset][action] = 0

        # Update strategy sum
        for action in actions:
            self.strategy_sum[infoset][action] += reach_probs[player] * strategy[action]

        return util

    def train(self, iterations: int):
        """Train CFR for given iterations."""
        cards = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']

        # Enumerate all possible deals
        all_deals = list(itertools.permutations(cards, 3))

        for i in range(iterations):
            # Iterate through all deals
            for deal in all_deals:
                state = LeducState()
                state.deal(list(deal))

                self.cfr(state, [1.0, 1.0])
                self.iterations += 1

            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{iterations}")

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get average strategy across all iterations."""
        avg_strategy = {}

        for infoset in self.strategy_sum:
            total = sum(self.strategy_sum[infoset].values())
            if total > 0:
                avg_strategy[infoset] = {
                    action: prob / total
                    for action, prob in self.strategy_sum[infoset].items()
                }
            else:
                # Uniform if never visited
                actions = list(self.strategy_sum[infoset].keys())
                avg_strategy[infoset] = {a: 1.0 / len(actions) for a in actions}

        return avg_strategy

    def compute_exploitability(self) -> float:
        """Compute exploitability of current average strategy."""
        avg_strat = self.get_average_strategy()

        # Compute best response value for each player
        br_values = []
        for player in [0, 1]:
            br_value = self.compute_best_response_value(player, avg_strat)
            br_values.append(br_value)

        # Exploitability is average of BR values
        return (br_values[0] + br_values[1]) / 2

    def compute_best_response_value(self, player: int, strategy: Dict) -> float:
        """Compute best response value for player against given strategy."""
        cards = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
        total_value = 0
        num_deals = 0

        # Enumerate all possible deals
        for deal in itertools.permutations(cards, 3):
            state = LeducState()
            state.deal(list(deal))
            value = self.best_response_util(state, player, strategy, player)
            total_value += value
            num_deals += 1

        return total_value / num_deals

    def best_response_util(self, state: LeducState, br_player: int, strategy: Dict, perspective: int) -> float:
        """Compute best response utility recursively."""
        if state.is_terminal():
            return state.get_payoff(perspective)

        player = state.acting_player
        infoset = state.get_infoset_key(player)
        actions = state.get_legal_actions()

        if player == br_player:
            # Best response: choose best action
            best_value = float('-inf')
            for action in actions:
                next_state = state.apply_action(action)
                value = self.best_response_util(next_state, br_player, strategy, perspective)
                best_value = max(best_value, value)
            return best_value
        else:
            # Opponent: follow strategy
            if infoset not in strategy:
                # Uniform if not in strategy
                probs = {a: 1.0 / len(actions) for a in actions}
            else:
                probs = strategy[infoset]

            value = 0
            for action in actions:
                next_state = state.apply_action(action)
                action_value = self.best_response_util(next_state, br_player, strategy, perspective)
                value += probs.get(action, 0) * action_value
            return value

    def compute_game_value(self) -> float:
        """Compute expected value for player 0 under average strategy."""
        avg_strat = self.get_average_strategy()
        cards = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']

        total_value = 0
        num_deals = 0

        for deal in itertools.permutations(cards, 3):
            state = LeducState()
            state.deal(list(deal))
            value = self.evaluate_strategy(state, avg_strat, 0)
            total_value += value
            num_deals += 1

        return total_value / num_deals

    def evaluate_strategy(self, state: LeducState, strategy: Dict, perspective: int) -> float:
        """Evaluate expected value under given strategy."""
        if state.is_terminal():
            return state.get_payoff(perspective)

        player = state.acting_player
        infoset = state.get_infoset_key(player)
        actions = state.get_legal_actions()

        if infoset not in strategy:
            probs = {a: 1.0 / len(actions) for a in actions}
        else:
            probs = strategy[infoset]

        value = 0
        for action in actions:
            next_state = state.apply_action(action)
            action_value = self.evaluate_strategy(next_state, strategy, perspective)
            value += probs.get(action, 0) * action_value

        return value


def save_strategy(solver: CFRSolver, filename: str):
    """Save strategy to JSON file in required format."""
    avg_strategy = solver.get_average_strategy()

    # Build strategy in required format
    strategy_data = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": "CFR+ solver, 10k iterations x 120 deals",
        "probs": {}
    }

    for infoset, action_probs in avg_strategy.items():
        # Convert probabilities to strings
        strategy_data["probs"][infoset] = {
            action: repr(prob) for action, prob in action_probs.items()
        }

    with open(filename, 'w') as f:
        json.dump(strategy_data, f, indent=2)


def main():
    print("Starting Leduc Hold'em CFR solver...")

    solver = CFRSolver()

    # Train
    num_iterations = 10000
    print(f"Training for {num_iterations} iterations (each processes all 120 deals)...")
    solver.train(num_iterations)

    print("\nComputing metrics...")

    # Compute exploitability
    exploitability = solver.compute_exploitability()
    print(f"Exploitability: {exploitability:.6f}")

    # Compute game value
    game_value = solver.compute_game_value()
    print(f"Game value (P0): {game_value:.6f}")

    # Save strategy
    print("\nSaving strategy to strategy.json...")
    save_strategy(solver, "strategy.json")

    print("Done!")

    return exploitability, game_value


if __name__ == "__main__":
    main()
