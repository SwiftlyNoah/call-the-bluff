#!/usr/bin/env python3
"""Leduc hold'em solver using CFR"""

import json
from collections import defaultdict
from typing import Dict, Tuple
import itertools

CARDS = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']

def card_rank(card: str) -> int:
    return {'J': 0, 'Q': 1, 'K': 2}[card[0]]


class LeducState:
    """Game state for Leduc hold'em"""

    def __init__(self, p0_card: str, p1_card: str, public_card: str = None):
        self.p0_card = p0_card
        self.p1_card = p1_card
        self.public_card = public_card
        self.bets = [[1, 1], [0, 0]]  # Bets in round 0 and 1
        self.history = []  # List of actions
        self.round = 0

    def copy(self):
        s = LeducState(self.p0_card, self.p1_card, self.public_card)
        s.bets = [row[:] for row in self.bets]
        s.history = self.history[:]
        s.round = self.round
        return s

    def legal_actions(self) -> list:
        """Get legal actions for current player"""
        to_move = len(self.history) % 2
        raise_limit = 2 if self.round == 0 else 2

        # Count raises in this round
        actions_this_round = []
        for a in self.history:
            if a == 'next_round':
                actions_this_round = []
            else:
                actions_this_round.append(a)

        raises = sum(1 for a in actions_this_round if a == 'raise')

        actions = []

        # Can always call (check if no bet)
        actions.append('call')

        # Can fold if opponent bet more
        opp = 1 - to_move
        if self.bets[self.round][opp] > self.bets[self.round][to_move]:
            actions.append('fold')

        # Can raise if limit not reached
        if raises < raise_limit:
            actions.append('raise')

        return actions

    def apply(self, action: str) -> None:
        """Apply an action"""
        to_move = len(self.history) % 2
        raise_size = 2 if self.round == 0 else 4

        self.history.append(action)

        if action == 'raise':
            self.bets[self.round][to_move] += raise_size
        elif action == 'call':
            opp = 1 - to_move
            needed = self.bets[self.round][opp] - self.bets[self.round][to_move]
            self.bets[self.round][to_move] += needed

            # Check if round should end
            if self.bets[self.round][0] == self.bets[self.round][1] and len([a for a in self.history if a != 'next_round']) >= 2:
                # Round ends - move to next
                if self.round == 0:
                    self.round = 1
                    self.history.append('next_round')

    def is_terminal(self) -> bool:
        """Check if game is over"""
        if 'fold' in self.history:
            return True

        # Check if both rounds completed
        actions_no_round = [a for a in self.history if a != 'next_round']

        if self.round == 0:
            # Need at least 2 actions in round 0 ending with call and matched bets
            if len(actions_no_round) >= 2 and actions_no_round[-1] == 'call':
                if self.bets[0][0] == self.bets[0][1]:
                    return False  # Round ends, but game continues

        if self.round == 1:
            # Find actions in round 1
            if 'next_round' in self.history:
                r1_start = self.history.index('next_round') + 1
                r1_actions = self.history[r1_start:]
            else:
                r1_actions = []

            if len(r1_actions) >= 2 and r1_actions[-1] == 'call':
                if self.bets[1][0] == self.bets[1][1]:
                    return True  # Game over

        return False

    def payoff(self, player: int) -> float:
        """Get payoff for player"""
        # Someone folded
        if 'fold' in self.history:
            folder = self.history.index('fold') % 2
            winner = 1 - folder
            pot = sum(self.bets[0]) + sum(self.bets[1])
            if player == winner:
                return pot - sum(self.bets[player])
            else:
                return -sum(self.bets[player])

        # Showdown
        p0_rank = card_rank(self.p0_card)
        p1_rank = card_rank(self.p1_card)
        pub_rank = card_rank(self.public_card)

        p0_match = (p0_rank == pub_rank)
        p1_match = (p1_rank == pub_rank)

        if p0_match and p1_match:
            winner = None  # Split pot
        elif p0_match:
            winner = 0
        elif p1_match:
            winner = 1
        else:
            if p0_rank > p1_rank:
                winner = 0
            elif p1_rank > p0_rank:
                winner = 1
            else:
                winner = None  # Split pot

        pot = sum(self.bets[0]) + sum(self.bets[1])
        if winner is None:
            return 0
        elif player == winner:
            return pot - sum(self.bets[player])
        else:
            return -sum(self.bets[player])


class CFRSolver:
    """CFR solver for Leduc hold'em"""

    def __init__(self):
        self.regrets = defaultdict(lambda: defaultdict(float))
        self.strategies = defaultdict(lambda: defaultdict(float))
        self.action_map = {}

    def infoset(self, state: LeducState, player: int) -> str:
        """Get information set ID"""
        card = state.p0_card if player == 0 else state.p1_card

        if state.round == 0:
            # First round: card:actions
            actions = ''.join(['f' if a == 'fold' else 'c' if a == 'call' else 'r'
                              for a in state.history if a != 'next_round'])
            return f"{card}:{actions}"
        else:
            # Second round: card|public:r1actions/r2actions
            r1_end = state.history.index('next_round') if 'next_round' in state.history else len(state.history)
            r1_actions = state.history[:r1_end]
            r2_actions = state.history[r1_end + 1:] if 'next_round' in state.history else []

            r1_str = ''.join(['f' if a == 'fold' else 'c' if a == 'call' else 'r' for a in r1_actions])
            r2_str = ''.join(['f' if a == 'fold' else 'c' if a == 'call' else 'r' for a in r2_actions])
            return f"{card}|{state.public_card}:{r1_str}/{r2_str}"

    def strategy(self, infoset: str, actions: list) -> dict:
        """Get strategy for infoset using regret matching"""
        pos_regrets = [max(0, self.regrets[infoset][a]) for a in actions]
        total = sum(pos_regrets)

        if total <= 0:
            return {a: 1.0 / len(actions) for a in actions}

        return {a: pr / total for a, pr in zip(actions, pos_regrets)}

    def avg_strategy(self, infoset: str, actions: list) -> dict:
        """Get average strategy"""
        total = sum(self.strategies[infoset][a] for a in actions)

        if total <= 0:
            return {a: 1.0 / len(actions) for a in actions}

        return {a: self.strategies[infoset][a] / total for a in actions}

    def cfr(self, state: LeducState, player: int, reach: Tuple[float, float]) -> float:
        """CFR traversal"""
        if state.is_terminal():
            return state.payoff(player)

        to_move = len([a for a in state.history if a != 'next_round']) % 2
        actions = state.legal_actions()
        iset = self.infoset(state, to_move)

        if iset not in self.action_map:
            self.action_map[iset] = actions

        strat = self.strategy(iset, actions)

        if to_move != player:
            # Opponent's turn
            value = 0.0
            for a in actions:
                s = state.copy()
                s.apply(a)
                new_reach = (reach[0] * strat[a], reach[1] * strat[a]) if to_move == 1 else (reach[0] * strat[a], reach[1])
                if to_move == 1:
                    new_reach = (reach[0], reach[1] * strat[a])
                value += strat[a] * self.cfr(s, player, new_reach)
            return value
        else:
            # Our turn
            value = 0.0
            values = {}

            for a in actions:
                s = state.copy()
                s.apply(a)
                new_reach = list(reach)
                new_reach[player] *= strat[a]
                new_reach = tuple(new_reach)

                av = self.cfr(s, player, new_reach)
                values[a] = av
                value += strat[a] * av

            # Update regrets
            for a in actions:
                r = values[a] - value
                self.regrets[iset][a] += r * reach[1 - player]
                self.strategies[iset][a] += strat[a] * reach[player]

            return value

    def solve(self, iterations: int = 1000):
        """Solve the game"""
        all_combos = list(itertools.combinations(CARDS, 2))

        for it in range(iterations):
            if it % 100 == 0:
                print(f"Iteration {it}")

            for p0_card, p1_card in all_combos:
                remaining = [c for c in CARDS if c not in [p0_card, p1_card]]

                for pub_card in remaining:
                    state = LeducState(p0_card, p1_card, pub_card)

                    # Compute probability of this deal
                    prob = 1.0 / (len(all_combos) * len(remaining))

                    for player in [0, 1]:
                        self.cfr(state, player, (prob, prob))

        # Extract strategy
        result = {}
        for iset in self.action_map:
            result[iset] = self.avg_strategy(iset, self.action_map[iset])

        return result


if __name__ == '__main__':
    print("Solving Leduc hold'em...")
    solver = CFRSolver()
    strategy = solver.solve(iterations=5000)

    # Convert probabilities to strings
    probs = {}
    for infoset, strat in strategy.items():
        probs[infoset] = {action: repr(prob) for action, prob in strat.items()}

    output = {
        'format': 'bluff-strategy-v1',
        'game': 'leduc',
        'producer': 'CFR solver for Leduc hold\'em',
        'probs': probs
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(strategy)} infosets to strategy.json")
