#!/usr/bin/env python3
"""Independent check of strategy.json for Leduc hold'em.

Shares no code with solve_leduc.py. It walks the full game tree deal by deal (no public-tree
vectors), checks the file's format and coverage, and computes the value of the profile and each
player's best-response value by backward induction over information sets.

Usage: python3 verify_leduc.py [strategy path]
"""
import json
import sys
from collections import defaultdict
from fractions import Fraction
from itertools import permutations

DECK = ["Js", "Jh", "Qs", "Qh", "Ks", "Kh"]
ORDER = {"J": 0, "Q": 1, "K": 2}
LETTER = {"fold": "f", "call": "c", "raise": "r"}
BET = (2, 4)


def legal(history):
    """Legal action labels after the current round's `history` (a string of f/c/r)."""
    raises = history.count("r")
    facing = history.endswith("r")
    out = []
    if facing:
        out.append("fold")
    out.append("call")
    if raises < 2:
        out.append("raise")
    return out


def round_over(history):
    return len(history) >= 2 and history[-1] == "c"


def chips(h1, h2):
    """Chips each player has put in, given both rounds' action strings."""
    put = [1, 1]
    for size, hist in zip(BET, (h1, h2)):
        for k, letter in enumerate(hist):
            who = k % 2
            if letter == "c":
                put[who] = put[1 - who]
            elif letter == "r":
                put[who] = put[1 - who] + size
    return put


def payoff0(c0, c1, pub, h1, h2):
    """Player 0's net payoff at a terminal history."""
    put = chips(h1, h2)
    last = h2 if h2 else h1
    if last.endswith("f"):
        folder = (len(last) - 1) % 2
        return -put[0] if folder == 0 else put[1]
    hit0 = c0[0] == pub[0]
    hit1 = c1[0] == pub[0]
    if hit0 != hit1:
        return put[1] if hit0 else -put[0]
    if ORDER[c0[0]] == ORDER[c1[0]]:
        return 0
    return put[1] if ORDER[c0[0]] > ORDER[c1[0]] else -put[0]


def infoset(card, pub, h1, h2, second):
    return f"{card}|{pub}:{h1}/{h2}" if second else f"{card}:{h1}"


def walk(c0, c1, h1, h2, pub, second, visit):
    """Depth-first walk of one private deal; `visit` handles decision and terminal states.

    Returns the value produced by `visit`'s combination rule.
    """
    hist = h2 if second else h1
    if hist.endswith("f"):
        return visit.terminal(c0, c1, pub, h1, h2)
    if round_over(hist):
        if second:
            return visit.terminal(c0, c1, pub, h1, h2)
        rest = [c for c in DECK if c not in (c0, c1)]
        return visit.chance(
            [(Fraction(1, len(rest)), lambda p=p: walk(c0, c1, h1, "", p, True, visit)) for p in rest]
        )
    who = len(hist) % 2
    card = c0 if who == 0 else c1
    name = infoset(card, pub, h1, h2, second)
    acts = legal(hist)

    def child(action):
        if second:
            return walk(c0, c1, h1, h2 + LETTER[action], pub, True, visit)
        return walk(c0, c1, h1 + LETTER[action], h2, pub, False, visit)

    return visit.decision(who, name, acts, child)


class Collect:
    """Collects every information set with its legal actions."""

    def __init__(self):
        self.sets = {}

    def terminal(self, *args):
        return None

    def chance(self, branches):
        for _, go in branches:
            go()

    def decision(self, who, name, acts, child):
        previous = self.sets.setdefault(name, (who, acts))
        assert previous == (who, acts), name
        for a in acts:
            child(a)


class ProfileValue:
    """Expected payoff to player 0 when both follow the strategy."""

    def __init__(self, probs):
        self.probs = probs

    def terminal(self, c0, c1, pub, h1, h2):
        return float(payoff0(c0, c1, pub, h1, h2))

    def chance(self, branches):
        return sum(float(p) * go() for p, go in branches)

    def decision(self, who, name, acts, child):
        return sum(self.probs[name][a] * child(a) for a in acts if self.probs[name][a] > 0)


def best_response_value(probs, hero, sets):
    """Value to `hero` of a best response to the other player's strategy in `probs`.

    States are grouped by hero information set with weight (chance * opponent reach). Information
    sets are then resolved deepest first: each picks the action with the highest weighted value,
    where values below it use the choices already made.
    """
    choice = {}
    states = defaultdict(list)  # infoset -> list of (weight, {action: thunk})

    class Gather:
        def __init__(self):
            self.weight = 1.0

        def terminal(self, *args):
            return None

        def chance(self, branches):
            saved = self.weight
            for p, go in branches:
                self.weight = saved * float(p)
                go()
            self.weight = saved

        def decision(self, who, name, acts, child):
            saved = self.weight
            if who == hero:
                states[name].append((saved, child))
                for a in acts:
                    child(a)
            else:
                for a in acts:
                    if probs[name][a] > 0:
                        self.weight = saved * probs[name][a]
                        child(a)
            self.weight = saved

    class Follow:
        """Value to hero below a state, using hero choices made so far."""

        def terminal(self, c0, c1, pub, h1, h2):
            v = float(payoff0(c0, c1, pub, h1, h2))
            return v if hero == 0 else -v

        def chance(self, branches):
            return sum(float(p) * go() for p, go in branches)

        def decision(self, who, name, acts, child):
            if who == hero:
                return child(choice[name])
            return sum(probs[name][a] * child(a) for a in acts if probs[name][a] > 0)

    deals = list(permutations(DECK, 2))
    deal_weight = 1.0 / len(deals)

    # The walk's child thunks are bound to a visitor, so gather with one visitor and re-walk with
    # another: store the state's coordinates instead of the thunk.
    coords = defaultdict(list)

    class GatherCoords(Gather):
        def decision(self, who, name, acts, child):
            if who == hero:
                coords[name].append((self.weight, self.where))
            saved = self.weight
            saved_where = self.where
            for a in acts:
                if who != hero:
                    if probs[name][a] <= 0:
                        continue
                    self.weight = saved * probs[name][a]
                self.where = saved_where + (a,)
                child(a)
            self.weight = saved
            self.where = saved_where

    follow = Follow()

    def replay(c0, c1, path, pubs):
        """Value to hero after following `path` (actions and public cards) from the deal."""
        h1, h2, pub, second = "", "", None, False
        for step in path:
            if step in DECK:
                pub, second = step, True
            elif second:
                h2 += LETTER[step]
            else:
                h1 += LETTER[step]
        return walk(c0, c1, h1, h2, pub, second, follow)

    for c0, c1 in deals:
        gather = GatherCoords()
        gather.weight = deal_weight
        gather.where = (c0, c1)
        original_chance = gather.chance

        def chance(branches, gather=gather, c0=c0, c1=c1):
            rest = [c for c in DECK if c not in (c0, c1)]
            saved = gather.weight
            saved_where = gather.where
            for (p, go), card in zip(branches, rest):
                gather.weight = saved * float(p)
                gather.where = saved_where + (card,)
                go()
            gather.weight = saved
            gather.where = saved_where

        gather.chance = chance
        walk(c0, c1, "", "", None, False, gather)

    def depth(name):
        return len(name.split(":")[1]) + (10 if "|" in name else 0)

    for name in sorted(coords, key=depth, reverse=True):
        acts = sets[name][1]
        totals = {a: 0.0 for a in acts}
        for weight, where in coords[name]:
            c0, c1, path = where[0], where[1], where[2:]
            for a in acts:
                totals[a] += weight * replay(c0, c1, path + (a,), None)
        choice[name] = max(acts, key=lambda a: totals[a])

    # Hero information sets the opponent never allows still need a choice for the final walk.
    for name, (who, acts) in sets.items():
        if who == hero and name not in choice:
            choice[name] = acts[0]

    total = 0.0
    for c0, c1 in deals:
        total += deal_weight * walk(c0, c1, "", "", None, False, follow)
    return total


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "strategy.json"
    with open(path) as handle:
        document = json.load(handle)
    assert document["format"] == "bluff-strategy-v1", document["format"]
    assert document["game"] == "leduc", document["game"]

    collect = Collect()
    for c0, c1 in permutations(DECK, 2):
        walk(c0, c1, "", "", None, False, collect)
    sets = collect.sets

    raw = document["probs"]
    missing = sorted(set(sets) - set(raw))
    extra = sorted(set(raw) - set(sets))
    assert not missing, f"missing information sets: {missing[:5]}"
    assert not extra, f"unknown information sets: {extra[:5]}"
    probs = {}
    for name, (who, acts) in sets.items():
        entry = raw[name]
        assert list(entry) == acts, (name, list(entry), acts)
        exact = {a: Fraction(entry[a]) for a in acts}
        assert all(isinstance(entry[a], str) for a in acts), name
        assert all(v >= 0 for v in exact.values()), name
        assert sum(exact.values()) == 1, (name, sum(exact.values()))
        probs[name] = {a: float(v) for a, v in exact.items()}
    print(f"format ok: {len(sets)} information sets, all present, each sums to exactly 1")

    deals = list(permutations(DECK, 2))
    value0 = sum(
        walk(c0, c1, "", "", None, False, ProfileValue(probs)) for c0, c1 in deals
    ) / len(deals)
    br0 = best_response_value(probs, 0, sets)
    br1 = best_response_value(probs, 1, sets)
    print(f"value_p0           {value0:+.10f}")
    print(f"best response p0   {br0:+.10f}")
    print(f"best response p1   {br1:+.10f}")
    print(f"exploitability     {(br0 + br1) / 2:.6e}")


if __name__ == "__main__":
    main()
