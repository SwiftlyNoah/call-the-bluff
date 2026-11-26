#!/usr/bin/env python3
"""Independent exact check of strategy.json against game.json, in rational arithmetic.

Verifies the file's shape, checks perfect recall, and prints the exact exploitability and
the value to player 0 when both players follow the file.
"""
import json
import sys
from fractions import Fraction

from solve import DECISION, Game, expected_value_p0, exploitability


def main():
    game_path = sys.argv[1] if len(sys.argv) > 1 else "game.json"
    strategy_path = sys.argv[2] if len(sys.argv) > 2 else "strategy.json"
    game = Game(game_path)
    with open(strategy_path) as f:
        doc = json.load(f)

    assert doc["format"] == "bluff-strategy-v1", doc["format"]
    assert doc["game"] == game.name, doc["game"]
    probs = doc["probs"]
    assert set(probs) == set(game.iset_names), "infoset names differ from the game's"

    strategy = []
    worst_sum_error = Fraction(0)
    for name, labels in zip(game.iset_names, game.iset_actions):
        row = probs[name]
        assert set(row) == set(labels) and len(row) == len(labels), name
        values = [Fraction(row[label]) for label in labels]
        assert all(v >= 0 for v in values), name
        total = sum(values)
        worst_sum_error = max(worst_sum_error, abs(total - 1))
        strategy.append([v / total for v in values])
    print("infosets: %d, worst |sum - 1| as written: %.3e" % (len(strategy), float(worst_sum_error)))

    # Perfect recall: every node of an infoset has the same sequence of own (infoset, action) pairs.
    parent = {}
    for i in range(game.n):
        for j, c in enumerate(game.children[i]):
            parent[c] = (i, j)
    history = {}
    for i in range(game.n):
        if game.kind[i] != DECISION:
            continue
        p = game.player[i]
        seq = []
        node = i
        while node in parent:
            node, j = parent[node]
            if game.kind[node] == DECISION and game.player[node] == p:
                seq.append((game.infoset[node], j))
        seq = tuple(seq)
        k = game.infoset[i]
        assert history.setdefault(k, seq) == seq, "imperfect recall at %r" % game.iset_names[k]
    print("perfect recall: yes")

    expl, br0, br1 = exploitability(game, strategy, exact=True)
    value = expected_value_p0(game, strategy, exact=True)
    print("br value p0 vs p1's strategy: %.12f" % float(br0))
    print("br value p1 vs p0's strategy: %.12f" % float(br1))
    print("exploitability (exact, as float): %.12e" % float(expl))
    print("value_p0 when both follow the file: %.12f" % float(value))
    print("game value is bracketed by [%.12f, %.12f]" % (float(-br1), float(br0)))


if __name__ == "__main__":
    main()
