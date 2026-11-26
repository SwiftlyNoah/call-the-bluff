#!/usr/bin/env python3
"""
Leduc hold'em CFR solver.
"""

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple


CARDS = ['Js', 'Jh', 'Qs', 'Qh', 'Ks', 'Kh']
RANKS = {'J': 0, 'Q': 1, 'K': 2}


class CFRSolver:
    """CFR solver for Leduc hold'em."""

    def __init__(self):
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
        self.iterations = 0

    def get_strategy(self, info_set: str, actions: List[str]) -> Dict[str, float]:
        """Get current strategy using regret matching."""
        regrets = self.regret_sum[info_set]

        positive_regret_sum = 0.0
        strategy = {}

        for action in actions:
            strategy[action] = max(0.0, regrets[action])
            positive_regret_sum += strategy[action]

        if positive_regret_sum > 0:
            for action in actions:
                strategy[action] /= positive_regret_sum
        else:
            for action in actions:
                strategy[action] = 1.0 / len(actions)

        return strategy

    def cfr(self, cards: List[str], history: str, public: str, reach_probs: List[float]) -> float:
        """
        Run CFR recursively.

        History: sequence of actions where each action is 'f', 'c', or 'r'
        Round 1 and 2 are separated by '/'
        """

        # Parse current state
        if '/' in history:
            parts = history.split('/')
            round1_hist = parts[0]
            round2_hist = parts[1]
            is_round2 = True
            current_hist = round2_hist
        else:
            round1_hist = history
            round2_hist = ''
            is_round2 = False
            current_hist = history

        # Determine acting player (alternates each action)
        acting_player = len(current_hist) % 2

        # Check terminal: fold
        if current_hist and current_hist[-1] == 'f':
            return self.payoff_fold(cards, history, public)

        # Check if round is complete
        if self.round_complete(current_hist):
            if not is_round2:
                # Move to round 2 with public card
                remaining = [c for c in CARDS if c not in cards]
                total_util = 0.0
                for pub_card in remaining:
                    util = self.cfr(cards, history + '/', pub_card, reach_probs)
                    total_util += util
                return total_util / len(remaining)
            else:
                # Showdown
                return self.payoff_showdown(cards, history, public)

        # Build info set
        if is_round2:
            info_set = f"{cards[acting_player]}|{public}:{round1_hist}/{round2_hist}"
        else:
            info_set = f"{cards[acting_player]}:{history}"

        # Get legal actions
        actions = self.get_legal_actions(current_hist)

        # Get strategy
        strategy = self.get_strategy(info_set, actions)

        # Recursively compute utilities
        action_utils = {}
        for action in actions:
            new_reach = reach_probs.copy()
            new_reach[acting_player] *= strategy[action]

            new_history = history + action[0]
            action_utils[action] = self.cfr(cards, new_history, public, new_reach)

        # Compute node utility
        util = sum(strategy[a] * action_utils[a] for a in actions)

        # Update regrets
        for action in actions:
            regret = action_utils[action] - util
            self.regret_sum[info_set][action] += reach_probs[1 - acting_player] * regret

        # Update strategy sum
        for action in actions:
            self.strategy_sum[info_set][action] += reach_probs[acting_player] * strategy[action]

        return util

    def round_complete(self, hist: str) -> bool:
        """
        Check if a betting round is complete.
        A round is complete when both players have acted and the last action
        was a call (or check) with bets matched.
        """
        if len(hist) == 0:
            return False

        # Last action must be 'c' (call or check)
        if hist[-1] != 'c':
            return False

        # Both players must have acted (at least 2 actions if started with check,
        # or appropriate length for raises)
        raises = hist.count('r')

        # Possible complete sequences:
        # "cc" - both check
        # "rc" - raise-call
        # "crc" - check-raise-call
        # "rrc" - raise-raise-call
        # "crrc" - check-raise-raise-call

        # After 0 raises: need "cc" (both checked)
        if raises == 0:
            return hist == "cc"

        # After 1 raise: need raise-call or check-raise-call
        if raises == 1:
            return hist == "rc" or hist == "crc"

        # After 2 raises: need rrc or crrc
        if raises == 2:
            return hist == "rrc" or hist == "crrc"

        return False

    def get_legal_actions(self, hist: str) -> List[str]:
        """Get legal actions in current betting round."""
        raises = hist.count('r')

        actions = ['call']

        # Can raise if less than 2 raises this round
        if raises < 2:
            actions.append('raise')

        # Can fold only if facing a bet (opponent's last action was raise)
        if len(hist) > 0 and hist[-1] == 'r':
            actions.append('fold')

        return actions

    def calculate_pot(self, history: str) -> List[int]:
        """Calculate pot contributions for each player."""
        pot = [1, 1]  # antes

        # Split into rounds
        if '/' in history:
            parts = history.split('/')
            round1_hist = parts[0]
            round2_hist = parts[1]
        else:
            round1_hist = history
            round2_hist = ''

        # Process round 1 (raise size 2)
        round_pot = [0, 0]
        for i, action in enumerate(round1_hist):
            player = i % 2
            if action == 'r':
                # Match + raise
                to_call = round_pot[1 - player] - round_pot[player]
                round_pot[player] += to_call + 2
                pot[player] += to_call + 2
            elif action == 'c':
                # Match
                to_call = round_pot[1 - player] - round_pot[player]
                round_pot[player] += to_call
                pot[player] += to_call

        # Process round 2 (raise size 4)
        round_pot = [0, 0]
        for i, action in enumerate(round2_hist):
            player = i % 2
            if action == 'r':
                to_call = round_pot[1 - player] - round_pot[player]
                round_pot[player] += to_call + 4
                pot[player] += to_call + 4
            elif action == 'c':
                to_call = round_pot[1 - player] - round_pot[player]
                round_pot[player] += to_call
                pot[player] += to_call

        return pot

    def payoff_fold(self, cards: List[str], history: str, public: str) -> float:
        """Payoff when someone folded (returns P0's payoff)."""
        # Remove the fold action to calculate pot
        pot = self.calculate_pot(history[:-1])

        # Find who folded
        if '/' in history:
            current_hist = history.split('/')[1]
        else:
            current_hist = history

        folded_player = (len(current_hist) - 1) % 2

        if folded_player == 0:
            return -pot[0]
        else:
            return pot[1]

    def payoff_showdown(self, cards: List[str], history: str, public: str) -> float:
        """Payoff at showdown (returns P0's payoff)."""
        pot = self.calculate_pot(history)

        p0_rank = RANKS[cards[0][0]]
        p1_rank = RANKS[cards[1][0]]
        public_rank = RANKS[public[0]]

        p0_pair = (p0_rank == public_rank)
        p1_pair = (p1_rank == public_rank)

        # Pairs beat non-pairs
        if p0_pair and not p1_pair:
            return pot[1]
        elif p1_pair and not p0_pair:
            return -pot[0]
        # Both have pairs or neither has pair
        elif p0_rank > p1_rank:
            return pot[1]
        elif p1_rank > p0_rank:
            return -pot[0]
        else:
            return 0.0  # Tie

    def train(self, iterations: int):
        """Train CFR."""
        print(f"Training CFR for {iterations} iterations...")

        for i in range(iterations):
            if (i + 1) % 10000 == 0:
                print(f"  Iteration {i + 1}/{iterations}")

            deck = CARDS.copy()
            random.shuffle(deck)
            cards = [deck[0], deck[1]]

            self.cfr(cards, '', None, [1.0, 1.0])
            self.iterations += 1

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get average strategy."""
        avg_strategy = {}

        for info_set, action_sums in self.strategy_sum.items():
            total = sum(action_sums.values())
            if total > 0:
                avg_strategy[info_set] = {
                    action: prob / total
                    for action, prob in action_sums.items()
                }
            else:
                actions = list(action_sums.keys())
                if len(actions) > 0:
                    avg_strategy[info_set] = {
                        action: 1.0 / len(actions)
                        for action in actions
                    }

        return avg_strategy

    def compute_exploitability(self) -> Tuple[float, float]:
        """Compute exploitability and game value."""
        strategy = self.get_average_strategy()

        br0 = self.best_response_value(strategy, 0)
        br1 = self.best_response_value(strategy, 1)
        game_value = self.game_value(strategy)

        exploitability = (br0 + br1) / 2
        return exploitability, game_value

    def best_response_value(self, strategy: Dict, br_player: int) -> float:
        """Compute best response value for a player."""
        total = 0.0
        count = 0

        for p0_card in CARDS:
            for p1_card in CARDS:
                if p0_card == p1_card:
                    continue
                cards = [p0_card, p1_card]
                value = self.br_recursive(strategy, cards, '', None, br_player)
                total += value
                count += 1

        return total / count

    def br_recursive(self, strategy: Dict, cards: List[str], history: str,
                    public: str, br_player: int) -> float:
        """Best response recursive calculation."""

        if '/' in history:
            parts = history.split('/')
            round1_hist = parts[0]
            round2_hist = parts[1]
            is_round2 = True
            current_hist = round2_hist
        else:
            round1_hist = history
            round2_hist = ''
            is_round2 = False
            current_hist = history

        acting_player = len(current_hist) % 2

        if current_hist and current_hist[-1] == 'f':
            payoff = self.payoff_fold(cards, history, public)
            return payoff if br_player == 0 else -payoff

        if self.round_complete(current_hist):
            if not is_round2:
                remaining = [c for c in CARDS if c not in cards]
                total = 0.0
                for pub_card in remaining:
                    total += self.br_recursive(strategy, cards, history + '/', pub_card, br_player)
                return total / len(remaining)
            else:
                payoff = self.payoff_showdown(cards, history, public)
                return payoff if br_player == 0 else -payoff

        if is_round2:
            info_set = f"{cards[acting_player]}|{public}:{round1_hist}/{round2_hist}"
        else:
            info_set = f"{cards[acting_player]}:{history}"

        actions = self.get_legal_actions(current_hist)

        if acting_player == br_player:
            # Best response: choose best action
            best = None
            for action in actions:
                new_history = history + action[0]
                value = self.br_recursive(strategy, cards, new_history, public, br_player)
                if best is None or value > best:
                    best = value
            return best
        else:
            # Opponent plays strategy
            if info_set in strategy:
                probs = strategy[info_set]
            else:
                probs = {a: 1.0 / len(actions) for a in actions}

            expected = 0.0
            for action in actions:
                prob = probs.get(action, 1.0 / len(actions))
                new_history = history + action[0]
                value = self.br_recursive(strategy, cards, new_history, public, br_player)
                expected += prob * value
            return expected

    def game_value(self, strategy: Dict) -> float:
        """Compute expected game value for P0."""
        total = 0.0
        count = 0

        for p0_card in CARDS:
            for p1_card in CARDS:
                if p0_card == p1_card:
                    continue
                cards = [p0_card, p1_card]
                value = self.ev_recursive(strategy, cards, '', None)
                total += value
                count += 1

        return total / count

    def ev_recursive(self, strategy: Dict, cards: List[str], history: str,
                    public: str) -> float:
        """Expected value recursive calculation."""

        if '/' in history:
            parts = history.split('/')
            round1_hist = parts[0]
            round2_hist = parts[1]
            is_round2 = True
            current_hist = round2_hist
        else:
            round1_hist = history
            round2_hist = ''
            is_round2 = False
            current_hist = history

        acting_player = len(current_hist) % 2

        if current_hist and current_hist[-1] == 'f':
            return self.payoff_fold(cards, history, public)

        if self.round_complete(current_hist):
            if not is_round2:
                remaining = [c for c in CARDS if c not in cards]
                total = 0.0
                for pub_card in remaining:
                    total += self.ev_recursive(strategy, cards, history + '/', pub_card)
                return total / len(remaining)
            else:
                return self.payoff_showdown(cards, history, public)

        if is_round2:
            info_set = f"{cards[acting_player]}|{public}:{round1_hist}/{round2_hist}"
        else:
            info_set = f"{cards[acting_player]}:{history}"

        actions = self.get_legal_actions(current_hist)

        if info_set in strategy:
            probs = strategy[info_set]
        else:
            probs = {a: 1.0 / len(actions) for a in actions}

        expected = 0.0
        for action in actions:
            prob = probs.get(action, 1.0 / len(actions))
            new_history = history + action[0]
            value = self.ev_recursive(strategy, cards, new_history, public)
            expected += prob * value

        return expected


def main():
    """Main function."""
    random.seed(42)

    solver = CFRSolver()
    solver.train(200000)

    print("\nComputing exploitability...")
    exploitability, value_p0 = solver.compute_exploitability()
    print(f"Exploitability: {exploitability:.6f}")
    print(f"Expected value for P0: {value_p0:.6f}")

    avg_strategy = solver.get_average_strategy()

    output = {
        "format": "bluff-strategy-v1",
        "game": "leduc",
        "producer": "CFR solver for Leduc hold'em",
        "probs": {}
    }

    for info_set, probs in sorted(avg_strategy.items()):
        output["probs"][info_set] = {
            action: f"{prob:.10f}"
            for action, prob in sorted(probs.items())
        }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nStrategy written to strategy.json")
    print(f"Total information sets: {len(avg_strategy)}")

    return exploitability, value_p0


if __name__ == '__main__':
    exploitability, value_p0 = main()

    print("\n" + "="*60)
    print(json.dumps({
        "claim_converged": exploitability < 0.01,
        "claimed_exploitability": round(exploitability, 6),
        "claimed_value_p0": round(value_p0, 6),
        "confidence": "high" if exploitability < 0.01 else "medium",
        "notes": f"Trained with 200,000 CFR iterations. Exploitability: {exploitability:.6f}"
    }))
