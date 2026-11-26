"""Tiny fallback builders for Kuhn and Leduc game files (CONTRACTS.md section 1).

The zoo prefers the poker agent's files in games/out/. These exist only so the zoo can run
before those land, and as a second, independent encoding to compare against them.

    uv run zoo/fallback_games.py            # writes zoo/out/games/{kuhn,leduc}.game.json
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out" / "games"
RANK = {"J": 0, "Q": 1, "K": 2}


class Builder:
    def __init__(self) -> None:
        self.nodes: list[dict] = []

    def add(self, node: dict) -> int:
        node["id"] = len(self.nodes)
        self.nodes.append(node)
        return node["id"]


def frac(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def kuhn() -> dict:
    b = Builder()
    root = b.add({"kind": "chance", "outcomes": []})
    cards = "JQK"

    def play(c0: str, c1: str, hist: str) -> int:
        # returns node id for the state after `hist`
        payoff = None
        win = 1 if RANK[c0] > RANK[c1] else -1
        if hist in ("cc",):
            payoff = win * 1
        elif hist in ("bc", "cbc"):
            payoff = win * 2
        elif hist == "bf":
            payoff = 1
        elif hist == "cbf":
            payoff = -1
        if payoff is not None:
            return b.add({"kind": "terminal", "payoff_p0": str(payoff)})
        player = len(hist) % 2
        card = c0 if player == 0 else c1
        nid = b.add({"kind": "decision", "player": player, "infoset": f"{card}:{hist}", "actions": []})
        facing = hist.endswith("b")
        acts = [("fold", "f"), ("call", "c")] if facing else [("check", "c"), ("bet", "b")]
        b.nodes[nid]["actions"] = [{"label": lab, "child": play(c0, c1, hist + ch)} for lab, ch in acts]
        return nid

    for c0 in cards:
        for c1 in cards:
            if c0 != c1:
                b.nodes[root]["outcomes"].append({"label": c0 + c1, "prob": "1/6", "child": play(c0, c1, "")})
    return {
        "format": "bluff-game-v1",
        "name": "kuhn",
        "description": "3-card Kuhn poker, ante 1, bet 1 (zoo fallback builder)",
        "known_value_p0": "-1/18",
        "known_value_source": "Kuhn 1950",
        "root": root,
        "nodes": b.nodes,
    }


def leduc() -> dict:
    """Leduc hold'em: JJQQKK, ante 1, raise 2 then 4, at most 2 raises per round, p0 acts first."""
    b = Builder()
    deck = {"J": 2, "Q": 2, "K": 2}

    def showdown(c0: str, c1: str, board: str) -> int:
        if c0 == board:
            return 1
        if c1 == board:
            return -1
        return (RANK[c0] > RANK[c1]) - (RANK[c0] < RANK[c1])

    def betting(c0, c1, board, rnd, r1, hist, contrib, raises) -> int:
        player = len(hist) % 2
        card = c0 if player == 0 else c1
        label = f"{card}:{hist}" if rnd == 1 else f"{card}:{r1}|{board}:{hist}"
        size = 2 if rnd == 1 else 4
        facing = contrib[0] != contrib[1]
        acts = []
        if facing:
            acts = [("fold", "f"), ("call", "c")] + ([("raise", "r")] if raises < 2 else [])
        else:
            acts = [("check", "k"), ("raise", "r")]
        nid = b.add({"kind": "decision", "player": player, "infoset": label, "actions": []})
        children = []
        for lab, ch in acts:
            h = hist + ch
            c = list(contrib)
            if ch == "f":
                pay = -c[0] if player == 0 else c[1]
                children.append({"label": lab, "child": b.add({"kind": "terminal", "payoff_p0": str(pay)})})
                continue
            if ch == "c":
                c[player] = c[1 - player]
            if ch == "r":
                c[player] = c[1 - player] + size
            round_over = ch == "c" or (ch == "k" and len(h) >= 2 and h[-2] == "k")
            if not round_over:
                children.append({"label": lab, "child": betting(c0, c1, board, rnd, r1, h, c, raises + (ch == "r"))})
            elif rnd == 1:
                children.append({"label": lab, "child": deal_board(c0, c1, h, c)})
            else:
                pay = showdown(c0, c1, board) * c[0]
                children.append({"label": lab, "child": b.add({"kind": "terminal", "payoff_p0": str(pay)})})
        b.nodes[nid]["actions"] = children
        return nid

    def deal_board(c0, c1, r1, contrib) -> int:
        nid = b.add({"kind": "chance", "outcomes": []})
        left = dict(deck)
        left[c0] -= 1
        left[c1] -= 1
        total = sum(left.values())
        outs = []
        for card, n in left.items():
            if n:
                outs.append({"label": card, "prob": frac(Fraction(n, total)),
                             "child": betting(c0, c1, card, 2, r1, "", contrib, 0)})
        b.nodes[nid]["outcomes"] = outs
        return nid

    root = b.add({"kind": "chance", "outcomes": []})
    outs = []
    for c0, n0 in deck.items():
        for c1, n1 in deck.items():
            n1e = n1 - (c0 == c1)
            p = Fraction(n0, 6) * Fraction(n1e, 5)
            if p:
                outs.append({"label": c0 + c1, "prob": frac(p), "child": betting(c0, c1, None, 1, "", "", [1, 1], 0)})
    b.nodes[root]["outcomes"] = outs
    return {
        "format": "bluff-game-v1",
        "name": "leduc",
        "description": "Leduc hold'em, JJQQKK, ante 1, raises 2/4, 2 raises per round (zoo fallback builder)",
        "root": root,
        "nodes": b.nodes,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for g in (kuhn(), leduc()):
        path = OUT / f"{g['name']}.game.json"
        path.write_text(json.dumps(g, separators=(",", ":")) + "\n")
        print(path, len(g["nodes"]), "nodes")


if __name__ == "__main__":
    main()
