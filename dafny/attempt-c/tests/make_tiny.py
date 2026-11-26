"""Write attempt C's own tiny test games and strategies into tests/data/.

uv run dafny/attempt-c/tests/make_tiny.py
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"


def game(name, nodes, **extra):
    return {"format": "bluff-game-v1", "name": name, "root": 0, "nodes": nodes, **extra}


def strat(name, probs):
    return {"format": "bluff-strategy-v1", "game": name, "producer": "attempt-c tests", "probs": probs}


def matching_pennies():
    # p0 picks H/T, p1 picks without seeing; p0 wins 1 on a match, loses 1 otherwise.
    nodes = [
        {"id": 0, "kind": "decision", "player": 0, "infoset": "P0", "actions": [
            {"label": "H", "child": 1}, {"label": "T", "child": 2}]},
        {"id": 1, "kind": "decision", "player": 1, "infoset": "P1", "actions": [
            {"label": "H", "child": 3}, {"label": "T", "child": 4}]},
        {"id": 2, "kind": "decision", "player": 1, "infoset": "P1", "actions": [
            {"label": "H", "child": 5}, {"label": "T", "child": 6}]},
        {"id": 3, "kind": "terminal", "payoff_p0": "1"},
        {"id": 4, "kind": "terminal", "payoff_p0": "-1"},
        {"id": 5, "kind": "terminal", "payoff_p0": "-1"},
        {"id": 6, "kind": "terminal", "payoff_p0": "1"},
    ]
    return game("mp", nodes, known_value_p0="0", known_value_source="symmetry")


def two_card_poker():
    # Chance deals p0 a card H or L (1/2 each). p0 bets or checks. After a bet p1 (who sees
    # nothing) calls or folds. Check: showdown for 1. Bet-call: showdown for 2. Fold: p0 wins 1.
    nodes = [{"id": 0, "kind": "chance", "outcomes": [
        {"label": "H", "prob": "1/2", "child": 1}, {"label": "L", "prob": "1/2", "child": 2}]}]
    nid = 3
    for card, node, sign in (("H", 1, 1), ("L", 2, -1)):
        check_t, bet = nid, nid + 1
        call_t, fold_t = nid + 2, nid + 3
        nid += 4
        nodes.append({"id": node, "kind": "decision", "player": 0, "infoset": card, "actions": [
            {"label": "check", "child": check_t}, {"label": "bet", "child": bet}]})
        nodes.append({"id": check_t, "kind": "terminal", "payoff_p0": str(sign)})
        nodes.append({"id": bet, "kind": "decision", "player": 1, "infoset": "bet", "actions": [
            {"label": "call", "child": call_t}, {"label": "fold", "child": fold_t}]})
        nodes.append({"id": call_t, "kind": "terminal", "payoff_p0": str(2 * sign)})
        nodes.append({"id": fold_t, "kind": "terminal", "payoff_p0": "1"})
    return game("two_card", nodes)


def forgetful():
    # NOT perfect recall: p0 moves L/R, then moves again at one infoset X that forgets the
    # first move. Payoff 1 iff the second move repeats the first. Any pure strategy gets 1
    # only by (L,l) or (R,r); a per-sequence optimizer would claim 1 for both branches.
    nodes = [
        {"id": 0, "kind": "decision", "player": 0, "infoset": "R0", "actions": [
            {"label": "L", "child": 1}, {"label": "R", "child": 2}]},
        {"id": 1, "kind": "decision", "player": 0, "infoset": "X", "actions": [
            {"label": "l", "child": 3}, {"label": "r", "child": 4}]},
        {"id": 2, "kind": "decision", "player": 0, "infoset": "X", "actions": [
            {"label": "l", "child": 5}, {"label": "r", "child": 6}]},
        {"id": 3, "kind": "terminal", "payoff_p0": "1"},
        {"id": 4, "kind": "terminal", "payoff_p0": "0"},
        {"id": 5, "kind": "terminal", "payoff_p0": "0"},
        {"id": 6, "kind": "terminal", "payoff_p0": "1"},
    ]
    return game("forgetful", nodes)


def main():
    OUT.mkdir(exist_ok=True)
    files = {
        "mp.game.json": matching_pennies(),
        "mp.eq.strategy.json": strat("mp", {"P0": {"H": "1/2", "T": "1/2"}, "P1": {"H": "1/2", "T": "1/2"}}),
        "mp.skew.strategy.json": strat("mp", {"P0": {"H": "0.7", "T": "0.3"}, "P1": {"H": "0.4", "T": "0.6"}}),
        "two_card.game.json": two_card_poker(),
        "two_card.eq.strategy.json": strat("two_card", {
            "H": {"check": "0", "bet": "1"}, "L": {"check": "2/3", "bet": "1/3"},
            "bet": {"call": "2/3", "fold": "1/3"}}),
        "two_card.naive.strategy.json": strat("two_card", {
            "H": {"check": "0", "bet": "1"}, "L": {"check": "1", "bet": "0"},
            "bet": {"call": "1/2", "fold": "1/2"}}),
        "forgetful.game.json": forgetful(),
        "forgetful.strategy.json": strat("forgetful", {"R0": {"L": "1/2", "R": "1/2"}, "X": {"l": "1/2", "r": "1/2"}}),
        "mp.badrow.strategy.json": strat("mp", {"P0": {"H": "0.5", "T": "0.6"}, "P1": {"H": "1/2", "T": "1/2"}}),
    }
    for name, doc in files.items():
        (OUT / name).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {len(files)} files to {OUT}")


if __name__ == "__main__":
    main()
