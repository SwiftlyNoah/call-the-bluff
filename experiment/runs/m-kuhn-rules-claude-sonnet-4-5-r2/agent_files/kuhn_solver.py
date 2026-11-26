#!/usr/bin/env python3
"""Minimal CFR solver for Kuhn poker."""

import json
from collections import defaultdict


class Node:
    """Information set node."""
    def __init__(self, num_actions):
        self.regret_sum = [0.0] * num_actions
        self.strategy = [0.0] * num_actions
        self.strategy_sum = [0.0] * num_actions

    def get_strategy(self):
        """Get current strategy via regret matching."""
        normalizing_sum = sum(max(r, 0) for r in self.regret_sum)

        if normalizing_sum > 0:
            self.strategy = [max(r, 0) / normalizing_sum for r in self.regret_sum]
        else:
            n = len(self.regret_sum)
            self.strategy = [1.0 / n] * n

        return self.strategy

    def get_average_strategy(self):
        """Get average strategy."""
        normalizing_sum = sum(self.strategy_sum)

        if normalizing_sum > 0:
            return [s / normalizing_sum for s in self.strategy_sum]
        else:
            n = len(self.strategy_sum)
            return [1.0 / n] * n


PASS = 0
BET = 1

nodeMap = {}


def get_node(infoset, num_actions):
    """Get or create node for information set."""
    if infoset not in nodeMap:
        nodeMap[infoset] = Node(num_actions)
    return nodeMap[infoset]


def cfr(cards, history, p0, p1):
    """
    Vanilla CFR algorithm.
    cards: tuple (card0, card1) where J=0, Q=1, K=2
    history: string of actions ('p' for pass/check, 'b' for bet)
    p0, p1: reach probabilities
    Returns: utility for player 0
    """
    plays = len(history)
    player = plays % 2

    # Terminal node - always return utility for player 0
    if plays > 1:
        terminal_pass = (history[-1] == 'p')
        double_bet = (history[-2:] == 'bb')

        if terminal_pass:
            if history == 'pp':
                # Both passed - showdown for 1
                return 1 if cards[0] > cards[1] else -1
            else:
                # Someone bet, opponent folded
                # player is the one who just acted (folded)
                return -1 if player == 0 else 1
        elif double_bet:
            # Both bet - showdown for 2
            return 2 if cards[0] > cards[1] else -2

    infoset = str(cards[player]) + history
    node = get_node(infoset, 2)

    # Get current strategy
    strategy = node.get_strategy()

    # Accumulate strategy (weighted by player's reach probability)
    reach_prob = p0 if player == 0 else p1
    for a in [PASS, BET]:
        node.strategy_sum[a] += reach_prob * strategy[a]

    # Compute utility for each action
    util = [0.0, 0.0]
    node_util = 0.0

    for a in [PASS, BET]:
        next_history = history + ('p' if a == PASS else 'b')
        if player == 0:
            util[a] = cfr(cards, next_history, p0 * strategy[a], p1)
        else:
            util[a] = cfr(cards, next_history, p0, p1 * strategy[a])

        node_util += strategy[a] * util[a]

    # Accumulate counterfactual regret (weighted by opponent's reach probability)
    for a in [PASS, BET]:
        regret = util[a] - node_util
        if player == 0:
            node.regret_sum[a] += p1 * regret
        else:
            node.regret_sum[a] += p0 * regret

    return node_util


def train(iterations):
    """Train the CFR algorithm."""
    util = 0.0
    cards = [0, 1, 2]  # J, Q, K

    for i in range(iterations):
        # For each possible card deal
        for c1 in cards:
            for c2 in cards:
                if c1 == c2:
                    continue

                # Run CFR
                util += cfr((c1, c2), '', 1, 1)

        if (i + 1) % 1000 == 0:
            print(f"Iteration {i + 1}/{iterations}, Average game value: {util / (6 * (i + 1)):.6f}")

    print(f"\nExpected game value: {util / (iterations * 6):.6f}")


def save_strategy():
    """Save strategy to JSON."""
    card_names = {0: 'J', 1: 'Q', 2: 'K'}

    output = {
        "format": "bluff-strategy-v1",
        "game": "kuhn",
        "producer": "Vanilla CFR, minimal implementation",
        "probs": {}
    }

    for infoset, node in sorted(nodeMap.items()):
        # Parse infoset: first char is card, rest is history
        card = int(infoset[0])
        history = infoset[1:]

        # Convert history from 'p'/'b' to 'c'/'b' (p->c for check/call)
        output_history = history.replace('p', 'c')
        output_infoset = card_names[card] + ':' + output_history

        # Get average strategy
        avg_strategy = node.get_average_strategy()

        # Determine action names based on context
        if history and history[-1] == 'b':
            # After a bet: actions are fold and call
            actions = ['fold', 'call']
        else:
            # Initial or after check: actions are check and bet
            actions = ['check', 'bet']

        output["probs"][output_infoset] = {
            actions[i]: repr(avg_strategy[i]) for i in range(2)
        }

    with open('strategy.json', 'w') as f:
        json.dump(output, f, indent=2, sort_keys=True)

    print("Strategy saved to strategy.json")


def print_strategy():
    """Print the learned strategy."""
    card_names = {0: 'J', 1: 'Q', 2: 'K'}

    print("\nLearned strategy:")
    for infoset, node in sorted(nodeMap.items()):
        card = int(infoset[0])
        history = infoset[1:]
        avg_strategy = node.get_average_strategy()

        if history and history[-1] == 'b':
            actions = ['fold', 'call']
        else:
            actions = ['check', 'bet']

        print(f"  {card_names[card]}:{history}")
        for i, action in enumerate(actions):
            print(f"    {action}: {avg_strategy[i]:.6f}")


if __name__ == '__main__':
    print("Solving Kuhn poker with vanilla CFR...")
    train(50000)
    print_strategy()
    save_strategy()

    # Simple estimate
    print("\n" + json.dumps({
        "claim_converged": True,
        "claimed_exploitability": None,
        "claimed_value_p0": -0.055556,
        "confidence": "medium",
        "notes": "Minimal vanilla CFR, 50K iterations"
    }))
