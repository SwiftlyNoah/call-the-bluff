#!/usr/bin/env python3
"""Independent check of strategy.json for Leduc hold'em, in exact rational arithmetic.

Shares no code with solver.py: the game is re-implemented here as a plain history-based
extensive-form game (one state per deal and action sequence), and best responses are found by
grouping states into the best responder's information sets.

Usage: python3 verify.py [strategy.json]
"""
import json
import sys
from fractions import Fraction
from itertools import permutations

DECK = ["Js", "Jh", "Qs", "Qh", "Ks", "Kh"]
ORDER = {"J": 0, "Q": 1, "K": 2}
BET = {1: 2, 2: 4}
LABEL = {"f": "fold", "c": "call", "r": "raise"}


def round_over(seq):
    """A round ends on a fold, or on a check/call that is not the round's first action."""
    return seq.endswith("f") or (len(seq) >= 2 and seq.endswith("c"))


def chips(first, second):
    """Chips each player has put in after the given action sequences."""
    put = [1, 1]
    for rnd, seq in ((1, first), (2, second)):
        for i, a in enumerate(seq):
            who = i % 2
            owe = put[1 - who] - put[who]
            if a == "c":
                put[who] += owe
            elif a == "r":
                put[who] += owe + BET[rnd]
    return put


class State:
    """cards = (player 0's card, player 1's card, public card); first/second = round actions."""

    def __init__(self, cards, first="", second=""):
        self.cards, self.first, self.second = cards, first, second

    def in_round_two(self):
        return round_over(self.first) and not self.first.endswith("f")

    def is_terminal(self):
        return self.first.endswith("f") or round_over(self.second)

    def seq(self):
        return self.second if self.in_round_two() else self.first

    def to_act(self):
        return len(self.seq()) % 2

    def legal(self):
        seq = self.seq()
        put = chips(self.first, self.second)
        who = self.to_act()
        acts = []
        if put[1 - who] > put[who]:
            acts.append("f")
        acts.append("c")
        if seq.count("r") < 2:
            acts.append("r")
        return acts

    def play(self, a):
        if self.in_round_two():
            return State(self.cards, self.first, self.second + a)
        return State(self.cards, self.first + a, self.second)

    def infoset(self):
        mine = self.cards[self.to_act()]
        if self.in_round_two():
            return f"{mine}|{self.cards[2]}:{self.first}/{self.second}"
        return f"{mine}:{self.first}"

    def payoff0(self):
        put = chips(self.first, self.second)
        seq = self.seq()
        if seq.endswith("f"):
            folder = (len(seq) - 1) % 2
            return -put[0] if folder == 0 else put[1]
        assert put[0] == put[1]
        a, b, pub = (ORDER[c[0]] for c in self.cards)
        if a == pub and b != pub:
            return put[1]
        if b == pub and a != pub:
            return -put[0]
        if a > b:
            return put[1]
        if b > a:
            return -put[0]
        return 0


def all_deals():
    deals = list(permutations(DECK, 3))
    return [(State(d), Fraction(1, len(deals))) for d in deals]


def enumerate_infosets():
    found = {}
    stack = [s for s, _ in all_deals()]
    while stack:
        s = stack.pop()
        if s.is_terminal():
            continue
        acts = s.legal()
        prev = found.setdefault(s.infoset(), acts)
        assert prev == acts
        stack.extend(s.play(a) for a in acts)
    return found


def load(path, infosets):
    with open(path) as f:
        doc = json.load(f)
    assert doc["format"] == "bluff-strategy-v1", doc["format"]
    assert doc["game"] == "leduc", doc["game"]
    probs = doc["probs"]
    missing = sorted(set(infosets) - set(probs))
    extra = sorted(set(probs) - set(infosets))
    assert not missing, f"missing information sets: {missing[:5]}"
    assert not extra, f"unknown information sets: {extra[:5]}"
    sigma = {}
    for name, acts in infosets.items():
        entry = probs[name]
        assert sorted(entry) == sorted(LABEL[a] for a in acts), (name, entry)
        row = {a: Fraction(entry[LABEL[a]]) for a in acts}
        assert all(isinstance(v, str) for v in entry.values()), name
        assert all(p >= 0 for p in row.values()), name
        assert sum(row.values()) == 1, (name, sum(row.values()))
        sigma[name] = row
    return sigma


def value(s, sigma):
    """Player 0's expected payoff from s when both follow sigma."""
    if s.is_terminal():
        return Fraction(s.payoff0())
    row = sigma[s.infoset()]
    return sum(p * value(s.play(a), sigma) for a, p in row.items() if p)


def advance(pairs, me, sigma):
    """Push weighted states forward through the opponent's moves until `me` acts or play ends.

    Returns (terminal payoff to me, {information set of me: [(state, weight), ...]})."""
    total = Fraction(0)
    groups = {}
    stack = list(pairs)
    while stack:
        s, w = stack.pop()
        if s.is_terminal():
            p0 = s.payoff0()
            total += w * (p0 if me == 0 else -p0)
        elif s.to_act() == me:
            groups.setdefault(s.infoset(), []).append((s, w))
        else:
            for a, p in sigma[s.infoset()].items():
                if p:
                    stack.append((s.play(a), w * p))
    return total, groups


def best_response_value(me, sigma):
    def solve(pairs):
        total, groups = advance(pairs, me, sigma)
        for group in groups.values():
            acts = group[0][0].legal()
            total += max(solve([(s.play(a), w) for s, w in group]) for a in acts)
        return total

    return solve(all_deals())


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "strategy.json"
    infosets = enumerate_infosets()
    sigma = load(path, infosets)
    v0 = sum(w * value(s, sigma) for s, w in all_deals())
    br0 = best_response_value(0, sigma)
    br1 = best_response_value(1, sigma)
    expl = (br0 + br1) / 2
    print(f"information sets: {len(infosets)} (all present, all rows sum to exactly 1)")
    print(f"value to player 0 under the profile: {float(v0):+.9f}")
    print(f"best response value, player 0: {float(br0):+.9f}")
    print(f"best response value, player 1: {float(br1):+.9f}")
    print(f"exploitability (br0 + br1) / 2: {float(expl):.9e}")
    assert br0 >= v0 and br1 >= -v0


if __name__ == "__main__":
    main()
