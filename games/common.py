"""Shared helpers for the game generators in games/.

A generator builds a tree with GameBuilder and writes a CONTRACTS.md game file
(format "bluff-game-v1"). Node ids are assigned in pre-order (root is 0), every
number is written as an exact rational string, and nothing here knows about any
particular game.

Also here: a small structural checker (`check_game`) used by the generators'
self-check and by games/tests/test_poker.py. It is not the project's validator
(that is reference/validate_game.py, owned by the spec task); it exists so a
generator can refuse to write a malformed file on its own.
"""

from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

FORMAT = "bluff-game-v1"


def q(x: Fraction | int) -> str:
    """Exact rational string: "3", "-2", "1/6"."""
    f = Fraction(x)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


class GameBuilder:
    """Builds a game tree in pre-order. Call the add_* methods depth first."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []

    def _new(self, node: dict[str, Any]) -> int:
        node["id"] = len(self.nodes)
        self.nodes.append(node)
        return node["id"]

    def chance(self, outcomes: list[tuple[str, Fraction, Any]]) -> int:
        """outcomes: (label, prob, build_child) where build_child() returns a node id."""
        nid = self._new({"kind": "chance", "outcomes": []})
        node = self.nodes[nid]
        for label, prob, build in outcomes:
            child = build()
            node["outcomes"].append({"label": label, "prob": q(prob), "child": child})
        return nid

    def decision(self, player: int, infoset: str, actions: list[tuple[str, Any]]) -> int:
        nid = self._new({"kind": "decision", "player": player, "infoset": infoset, "actions": []})
        node = self.nodes[nid]
        for label, build in actions:
            node["actions"].append({"label": label, "child": build()})
        return nid

    def terminal(self, payoff_p0: Fraction | int) -> int:
        return self._new({"kind": "terminal", "payoff_p0": q(payoff_p0)})


def game_doc(name: str, description: str, nodes: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {"format": FORMAT, "name": name, "description": description}
    doc.update(extra)
    doc["root"] = 0
    # Key order inside a node: id first, for readability.
    doc["nodes"] = [{"id": n["id"], **{k: v for k, v in n.items() if k != "id"}} for n in nodes]
    return doc


def write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")


# ---------------------------------------------------------------------------
# Structural self-check


def check_game(doc: dict[str, Any]) -> dict[str, int]:
    """Raise AssertionError on a malformed game; return counts.

    Checks: format, ids are 0..n-1, tree shape (every non-root node is a child
    exactly once, all reachable from the root), exact chance sums, infoset
    consistency (same player, same action labels in the same order) and perfect
    recall (every node of an infoset has the same sequence of the acting
    player's own (infoset, action) pairs on its path).
    """
    assert doc["format"] == FORMAT
    nodes = doc["nodes"]
    for i, n in enumerate(nodes):
        assert n["id"] == i, f"node ids must be 0..n-1 in order, got {n['id']} at {i}"
    parent_count: Counter[int] = Counter()
    for n in nodes:
        kids = [o["child"] for o in n.get("outcomes", [])] + [a["child"] for a in n.get("actions", [])]
        parent_count.update(kids)
        if n["kind"] == "chance":
            probs = [Fraction(o["prob"]) for o in n["outcomes"]]
            assert all(0 < p <= 1 for p in probs), f"bad prob at node {n['id']}"
            assert sum(probs) == 1, f"chance probs at node {n['id']} sum to {sum(probs)}"
            labels = [o["label"] for o in n["outcomes"]]
            assert len(set(labels)) == len(labels)
        elif n["kind"] == "decision":
            assert n["player"] in (0, 1)
            labels = [a["label"] for a in n["actions"]]
            assert labels and len(set(labels)) == len(labels)
        else:
            assert n["kind"] == "terminal"
            Fraction(n["payoff_p0"])
    root = doc["root"]
    assert parent_count[root] == 0
    for i in range(len(nodes)):
        if i != root:
            assert parent_count[i] == 1, f"node {i} has {parent_count[i]} parents"

    infoset_sig: dict[str, tuple[int, tuple[str, ...]]] = {}
    infoset_hist: dict[str, tuple] = {}
    reached = 0
    stack: list[tuple[int, tuple, tuple]] = [(root, (), ())]  # node, p0 seq, p1 seq
    while stack:
        nid, s0, s1 = stack.pop()
        reached += 1
        n = nodes[nid]
        if n["kind"] == "chance":
            for o in n["outcomes"]:
                stack.append((o["child"], s0, s1))
        elif n["kind"] == "decision":
            p, iset = n["player"], n["infoset"]
            sig = (p, tuple(a["label"] for a in n["actions"]))
            own = s0 if p == 0 else s1
            if iset in infoset_sig:
                assert infoset_sig[iset] == sig, f"infoset {iset} inconsistent"
                assert infoset_hist[iset] == own, f"perfect recall fails at infoset {iset}"
            else:
                infoset_sig[iset], infoset_hist[iset] = sig, own
            for a in n["actions"]:
                step = ((iset, a["label"]),)
                stack.append((a["child"], s0 + step if p == 0 else s0, s1 + step if p == 1 else s1))
    assert reached == len(nodes), "unreachable nodes"

    kinds = Counter(n["kind"] for n in nodes)
    per_player = Counter(sig[0] for sig in infoset_sig.values())
    return {
        "nodes": len(nodes),
        "chance": kinds["chance"],
        "decision": kinds["decision"],
        "terminal": kinds["terminal"],
        "infosets": len(infoset_sig),
        "infosets_p0": per_player[0],
        "infosets_p1": per_player[1],
    }
