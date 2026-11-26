#!/usr/bin/env python3
"""
Counterfactual Regret Minimization solver for Kuhn poker.
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple


class KuhnCFR:
    """CFR solver for Kuhn poker."""

    CARDS = ['J', 'Q', 'K']
    CARD_RANK = {'J': 0, 'Q': 1, 'K': 2}

    def __init__(self):
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))

    def get_strategy(self, infoset: str) -> Dict[str, float]:
        """Get current strategy from regret matching."""
        actions = self.get_actions(infoset)
        regrets = self.regret_sum[infoset]

        normalizing_sum = sum(max(0, regrets[a]) for a in actions)

        strategy = {}
        if normalizing_sum > 0:
            for a in actions:
                strategy[a] = max(0, regrets[a]) / normalizing_sum
        else:
            for a in actions:
                strategy[a] = 1.0 / len(actions)

        return strategy

    def get_actions(self, infoset: str) -> List[str]:
        """Get available actions for an information set."""
        history = infoset.split(':')[1]

        if history == '':
            return ['check', 'bet']
        elif history == 'c':
            return ['check', 'bet']
        elif history == 'b':
            return ['fold', 'call']
        elif history == 'cb':
            return ['fold', 'call']

        raise ValueError(f"Unknown infoset: {infoset}")

    def is_terminal(self, history: str) -> bool:
        """Check if history represents a terminal state."""
        return history in ['cc', 'bc', 'bf', 'cbc', 'cbf']

    def get_payoff(self, history: str, card0: str, card1: str) -> int:
        """Get payoff for player 0 at terminal node."""
        if history == 'cc':
            return 1 if self.CARD_RANK[card0] > self.CARD_RANK[card1] else -1
        elif history == 'bc':
            return 2 if self.CARD_RANK[card0] > self.CARD_RANK[card1] else -2
        elif history == 'bf':
            return 1
        elif history == 'cbc':
            return 2 if self.CARD_RANK[card0] > self.CARD_RANK[card1] else -2
        elif history == 'cbf':
            return -1

        raise ValueError(f"Unknown terminal history: {history}")

    def get_current_player(self, history: str) -> int:
        """Get the player to act."""
        if history in ['', 'cb']:
            return 0
        elif history in ['c', 'b']:
            return 1
        raise ValueError(f"Unknown history: {history}")

    def action_to_history(self, action: str) -> str:
        """Convert action to history character."""
        return {'check': 'c', 'bet': 'b', 'call': 'c', 'fold': 'f'}[action]

    def cfr(self, cards: Tuple[str, str], history: str,
            pi_0: float, pi_1: float, update_player: int) -> float:
        """
        Run CFR recursively for a specific player.
        Returns expected value for player 0.
        """
        card0, card1 = cards

        if self.is_terminal(history):
            return float(self.get_payoff(history, card0, card1))

        player = self.get_current_player(history)
        card = card0 if player == 0 else card1
        infoset = f"{card}:{history}"

        strategy = self.get_strategy(infoset)
        actions = self.get_actions(infoset)

        # Compute action values
        action_utils = {}
        for action in actions:
            next_history = history + self.action_to_history(action)
            if player == 0:
                action_utils[action] = self.cfr(cards, next_history,
                                                pi_0 * strategy[action], pi_1,
                                                update_player)
            else:
                action_utils[action] = self.cfr(cards, next_history,
                                                pi_0, pi_1 * strategy[action],
                                                update_player)

        util = sum(strategy[a] * action_utils[a] for a in actions)

        # Only update if this is the update player's node
        if player == update_player:
            reach_prob = pi_1 if player == 0 else pi_0
            my_reach = pi_0 if player == 0 else pi_1

            for action in actions:
                regret = action_utils[action] - util
                self.regret_sum[infoset][action] += reach_prob * regret
                self.strategy_sum[infoset][action] += my_reach * strategy[action]

        return util

    def train(self, iterations: int):
        """Train CFR for specified iterations."""
        deals = [(c0, c1) for c0 in self.CARDS for c1 in self.CARDS if c0 != c1]

        for i in range(iterations):
            for cards in deals:
                # Update player 0
                self.cfr(cards, '', 1.0, 1.0, 0)
                # Update player 1
                self.cfr(cards, '', 1.0, 1.0, 1)

            if (i + 1) % 10000 == 0:
                print(f"Iteration {i + 1}/{iterations}")

    def get_average_strategy(self) -> Dict[str, Dict[str, float]]:
        """Get time-averaged strategy."""
        avg_strategy = {}

        for infoset, action_sums in self.strategy_sum.items():
            actions = self.get_actions(infoset)
            total = sum(action_sums.values())

            avg_strategy[infoset] = {}
            if total > 0:
                for action in actions:
                    avg_strategy[infoset][action] = action_sums[action] / total
            else:
                for action in actions:
                    avg_strategy[infoset][action] = 1.0 / len(actions)

        return avg_strategy

    def compute_best_response_value(self, strategy: Dict[str, Dict[str, float]],
                                     br_player: int) -> float:
        """Compute best response value for a player."""
        deals = [(c0, c1) for c0 in self.CARDS for c1 in self.CARDS if c0 != c1]
        total = 0.0

        for cards in deals:
            value = self._br_traverse(cards, '', strategy, br_player)
            total += value

        return total / len(deals)

    def _br_traverse(self, cards: Tuple[str, str], history: str,
                    strategy: Dict[str, Dict[str, float]], br_player: int) -> float:
        """Traverse tree for best response calculation."""
        card0, card1 = cards

        if self.is_terminal(history):
            payoff = float(self.get_payoff(history, card0, card1))
            return payoff if br_player == 0 else -payoff

        player = self.get_current_player(history)
        card = card0 if player == 0 else card1
        infoset = f"{card}:{history}"
        actions = self.get_actions(infoset)

        if player == br_player:
            return max(self._br_traverse(cards, history + self.action_to_history(a),
                                        strategy, br_player) for a in actions)
        else:
            if infoset in strategy:
                strat = strategy[infoset]
            else:
                strat = {a: 1.0/len(actions) for a in actions}

            return sum(strat[a] * self._br_traverse(cards,
                                                    history + self.action_to_history(a),
                                                    strategy, br_player)
                      for a in actions)

    def compute_exploitability(self, strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute exploitability of strategy."""
        br0 = self.compute_best_response_value(strategy, 0)
        br1 = self.compute_best_response_value(strategy, 1)
        return (br0 + br1) / 2

    def compute_strategy_value(self, strategy: Dict[str, Dict[str, float]]) -> float:
        """Compute expected value for player 0."""
        deals = [(c0, c1) for c0 in self.CARDS for c1 in self.CARDS if c0 != c1]
        total = 0.0

        for cards in deals:
            total += self._strategy_traverse(cards, '', strategy)

        return total / len(deals)

    def _strategy_traverse(self, cards: Tuple[str, str], history: str,
                          strategy: Dict[str, Dict[str, float]]) -> float:
        """Traverse tree with both players following strategy."""
        card0, card1 = cards

        if self.is_terminal(history):
            return float(self.get_payoff(history, card0, card1))

        player = self.get_current_player(history)
        card = card0 if player == 0 else card1
        infoset = f"{card}:{history}"
        actions = self.get_actions(infoset)

        if infoset in strategy:
            strat = strategy[infoset]
        else:
            strat = {a: 1.0/len(actions) for a in actions}

        return sum(strat[a] * self._strategy_traverse(cards,
                                                      history + self.action_to_history(a),
                                                      strategy)
                  for a in actions)


def main():
    print("Solving Kuhn poker with CFR...")

    solver = KuhnCFR()
    solver.train(100000)

    print("\nComputing average strategy...")
    strategy = solver.get_average_strategy()

    print("\nAverage strategy:")
    for infoset in sorted(strategy.keys()):
        print(f"  {infoset}:")
        for action, prob in strategy[infoset].items():
            print(f"    {action}: {prob:.6f}")

    print("\nComputing metrics...")
    exploitability = solver.compute_exploitability(strategy)
    value_p0 = solver.compute_strategy_value(strategy)

    print(f"Exploitability: {exploitability:.8f}")
    print(f"Expected value for player 0: {value_p0:.8f}")

    print("\nSaving strategy to strategy.json...")
    output = {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": "CFR solver, 100000 iterations",
        "probs": {
            infoset: {action: repr(prob) for action, prob in probs.items()}
            for infoset, probs in strategy.items()
        }
    }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print("Done!")

    return exploitability, value_p0


if __name__ == '__main__':
    exp, val = main()

    result = {
        "claim_converged": exp < 0.001,
        "claimed_exploitability": exp,
        "claimed_value_p0": val,
        "confidence": "high" if exp < 0.0001 else "medium" if exp < 0.001 else "low",
        "notes": "CFR with separate player traversals, 100000 iterations"
    }
    print("\n" + json.dumps(result))
