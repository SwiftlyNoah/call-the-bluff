#!/usr/bin/env python3
"""
Leduc hold'em solver using Counterfactual Regret Minimization (CFR+)
"""
import json
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

RANKS = ['J', 'Q', 'K']
SUITS = ['s', 'h']
DECK = [r + s for r in RANKS for s in SUITS]
RANK_ORDER = {'J': 0, 'Q': 1, 'K': 2}

class LeducSolver:
    def __init__(self):
        self.regret_sums = defaultdict(lambda: defaultdict(float))
        self.strategy_sums = defaultdict(lambda: defaultdict(float))

    def card_rank(self, card: str) -> int:
        return RANK_ORDER[card[0]]

    def showdown(self, card0: str, card1: str, public_card: str) -> int:
        """Return 1 if p0 wins, -1 if p1 wins, 0 if tie"""
        rank0 = self.card_rank(card0)
        rank1 = self.card_rank(card1)
        pub_rank = self.card_rank(public_card)

        p0_paired = rank0 == pub_rank
        p1_paired = rank1 == pub_rank

        if p0_paired and not p1_paired:
            return 1
        if p1_paired and not p0_paired:
            return -1
        if p0_paired and p1_paired:
            if rank0 > rank1:
                return 1
            elif rank1 > rank0:
                return -1
            else:
                return 0

        # Neither paired
        if rank0 > rank1:
            return 1
        elif rank1 > rank0:
            return -1
        else:
            return 0

    def get_legal_actions(self, action_history: str) -> List[str]:
        """Get legal actions given action history of current round"""
        if not action_history:
            return ['call', 'raise']

        last_action = action_history[-1]
        num_raises = action_history.count('r')

        actions = ['call']
        if last_action == 'r':
            actions.append('fold')
        if num_raises < 2:
            actions.append('raise')

        return actions

    def infoset_name(self, card: str, public_card: Optional[str],
                     r1_actions: str, r2_actions: str) -> str:
        """Generate information set name"""
        if public_card is None:
            return f"{card}:{r1_actions}"
        else:
            return f"{card}|{public_card}:{r1_actions}/{r2_actions}"

    def cfr_recursive(self, card0: str, card1: str, public_card: Optional[str],
                      r1_actions: str, r2_actions: str, actor: int,
                      p0_bet: int, p1_bet: int, reach: Tuple[float, float]) -> float:
        """
        Recursively compute value using CFR.
        Returns utility to player 0.
        p0_bet, p1_bet are the total amounts each player has contributed so far.
        """
        # Terminal: someone folded in round 1
        if r1_actions and r1_actions[-1] == 'f':
            actors_r1 = [i % 2 for i in range(len(r1_actions))]
            folder = actors_r1[-1]
            if folder == 0:
                return -p0_bet
            else:
                return p1_bet

        # Transition from round 1 to round 2
        if len(r1_actions) >= 2 and r1_actions[-1] == 'c':
            if public_card is None:
                # Deal public card and transition to round 2
                dealt = {card0, card1}
                remaining = [c for c in DECK if c not in dealt]
                expected_value = 0.0
                for pub_card in remaining:
                    value = self.cfr_recursive(card0, card1, pub_card, r1_actions, "",
                                              0, p0_bet, p1_bet, reach)
                    expected_value += value / len(remaining)
                return expected_value

        # Terminal: someone folded in round 2
        if r2_actions and r2_actions[-1] == 'f':
            actors_r2 = [i % 2 for i in range(len(r2_actions))]
            folder = actors_r2[-1]
            if folder == 0:
                return -p0_bet
            else:
                return p1_bet

        # Terminal: showdown
        if len(r2_actions) >= 2 and r2_actions[-1] == 'c':
            winner = self.showdown(card0, card1, public_card)
            if winner == 1:
                return p1_bet
            elif winner == -1:
                return -p0_bet
            else:
                return 0

        # Non-terminal: decision node
        is_r1 = public_card is None
        current_history = r1_actions if is_r1 else r2_actions
        my_card = card0 if actor == 0 else card1

        legal_actions = self.get_legal_actions(current_history)
        iset = self.infoset_name(my_card, public_card, r1_actions, r2_actions)

        # Get strategy
        strategy = self.get_strategy(iset, legal_actions)

        # Compute values for each action
        action_values = {}
        node_value = 0.0
        my_reach = reach[actor]
        opp_reach = reach[1 - actor]

        for action in legal_actions:
            if action == 'call':
                new_r1 = r1_actions + 'c' if is_r1 else r1_actions
                new_r2 = r2_actions + 'c' if not is_r1 else ""
                new_p0_bet = p0_bet
                new_p1_bet = p1_bet
            elif action == 'raise':
                bet_increase = 2 if is_r1 else 4
                new_r1 = r1_actions + 'r' if is_r1 else r1_actions
                new_r2 = r2_actions + 'r' if not is_r1 else ""
                if actor == 0:
                    new_p0_bet = p0_bet + bet_increase
                    new_p1_bet = p1_bet
                else:
                    new_p0_bet = p0_bet
                    new_p1_bet = p1_bet + bet_increase
            elif action == 'fold':
                new_r1 = r1_actions + 'f' if is_r1 else r1_actions
                new_r2 = r2_actions + 'f' if not is_r1 else ""
                new_p0_bet = p0_bet
                new_p1_bet = p1_bet

            new_reach = (reach[0] * strategy[action], reach[1] * strategy[action])
            action_values[action] = self.cfr_recursive(card0, card1, public_card,
                                                      new_r1, new_r2, 1 - actor,
                                                      new_p0_bet, new_p1_bet, new_reach)
            node_value += strategy[action] * action_values[action]

        # Update strategy sums and regrets
        for action in legal_actions:
            self.strategy_sums[iset][action] += my_reach * strategy[action]
            regret = action_values[action] - node_value
            self.regret_sums[iset][action] += opp_reach * regret

        return node_value

    def get_strategy(self, iset: str, legal_actions: List[str]) -> Dict[str, float]:
        """Get regret-matched strategy"""
        strategy = {}
        sum_positive_regret = 0.0

        for action in legal_actions:
            regret = max(self.regret_sums[iset].get(action, 0), 0)
            sum_positive_regret += regret

        if sum_positive_regret > 0:
            for action in legal_actions:
                strategy[action] = max(self.regret_sums[iset].get(action, 0), 0) / sum_positive_regret
        else:
            uniform = 1.0 / len(legal_actions)
            for action in legal_actions:
                strategy[action] = uniform

        return strategy

    def solve(self, iterations: int = 1000):
        """Run CFR"""
        print(f"Running CFR for {iterations} iterations...")

        for t in range(1, iterations + 1):
            # Iterate over all card combinations
            for cards in combinations(DECK, 2):
                card0, card1 = cards
                # Both players ante 1
                self.cfr_recursive(card0, card1, None, "", "", 0, 1, 1, (1.0, 1.0))

            if t % max(1, iterations // 10) == 0:
                print(f"  Iteration {t}/{iterations}")

        print("CFR complete!")

    def get_average_strategy(self) -> Dict[str, Dict[str, str]]:
        """Compute average strategy"""
        avg_strategy = {}

        for iset, action_counts in self.strategy_sums.items():
            total = sum(action_counts.values())
            if total > 0:
                avg_strategy[iset] = {
                    action: str(count / total)
                    for action, count in action_counts.items()
                }

        return avg_strategy

    def save_strategy(self, filename: str = "strategy.json"):
        """Save strategy to JSON"""
        avg_strategy = self.get_average_strategy()

        output = {
            "format": "bluff-strategy-v1",
            "game": "leduc",
            "producer": "CFR solver for Leduc hold'em",
            "probs": avg_strategy
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Strategy saved to {filename}")
        print(f"Number of information sets: {len(avg_strategy)}")

def main():
    solver = LeducSolver()
    solver.solve(iterations=10000)
    solver.save_strategy()

if __name__ == "__main__":
    main()
