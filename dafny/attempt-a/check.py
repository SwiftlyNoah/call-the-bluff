"""Thin trusted wrapper around the verified core (attempt A).

Usage: uv run dafny/attempt-a/check.py GAME.game.json STRATEGY.strategy.json

Reads a bluff-game-v1 game and a bluff-strategy-v1 profile, builds the Dafny
values (exact rationals throughout), calls the compiled Checker.Check, and prints
a bluff-verdict-v1 JSON verdict. Exit 2 on a malformed game or strategy, with
the reason on stderr.

This file is in the trusted base: JSON parsing, the infoset and action
numbering, and the strategy normalization (each row must sum to 1 within 1e-6
and is then divided by its exact sum) are not verified. Everything after the
call to Check is the compiled Dafny core, whose guarantees are CheckSound in
Check.dfy.
"""

from __future__ import annotations

import json
import sys
import threading
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build" / "checker-py"

TOL = Fraction(1, 10**6)


class Malformed(Exception):
    pass


def load_core():
    if not (BUILD / "Checker.py").exists():
        raise SystemExit(f"compiled core missing: run {HERE / 'build.sh'} first")
    sys.path.insert(0, str(BUILD))
    import _dafny  # noqa: E402
    import Checker  # noqa: E402
    import Game  # noqa: E402

    return _dafny, Checker, Game


def no_dup_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise Malformed(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def edges(n: dict, nid: int, key: str) -> list[dict]:
    """The outcome or action list of a node, each entry a dict with a unique string label."""
    es = n.get(key)
    if not isinstance(es, list) or not all(isinstance(e, dict) for e in es):
        raise Malformed(f"game: node {nid} {key} is not a list of objects")
    labels = [e.get("label") for e in es]
    if not all(isinstance(x, str) for x in labels) or len(set(labels)) != len(labels):
        raise Malformed(f"game: node {nid} {key} labels are missing or not unique: {labels!r}")
    return es


def rat(s, what: str) -> Fraction:
    if not isinstance(s, str):
        raise Malformed(f"{what}: expected an exact rational string, got {s!r}")
    try:
        return Fraction(s.strip())
    except (ValueError, ZeroDivisionError) as e:
        raise Malformed(f"{what}: not a rational: {s!r}") from e


def fmt(q: Fraction) -> str:
    q = Fraction(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def build_game(game: dict, D, G):
    """Return (root Node, arity list, infoset names, action labels per infoset)."""
    if game.get("format") != "bluff-game-v1":
        raise Malformed("game: format is not bluff-game-v1")
    nodes = {}
    for n in game.get("nodes", []):
        nid = n.get("id")
        if not is_int(nid) or nid < 0 or nid in nodes:
            raise Malformed(f"game: bad or duplicate node id {nid!r}")
        nodes[nid] = n
    root_id = game.get("root")
    if root_id not in nodes:
        raise Malformed("game: root is not a node id")

    infoset_ids: dict[str, int] = {}
    labels: list[list[str]] = []
    owner: list[int] = []
    seen: set[int] = set()

    # Iterative post-order build (no Python recursion on deep trees).
    built: dict[int, object] = {}
    stack = [(root_id, False)]
    while stack:
        nid, done = stack.pop()
        n = nodes.get(nid)
        if not is_int(nid) or n is None:
            raise Malformed(f"game: child {nid} is not a node")
        kind = n.get("kind")
        if kind == "terminal":
            if nid in seen:
                raise Malformed(f"game: node {nid} appears twice (not a tree)")
            seen.add(nid)
            built[nid] = G.Node_Terminal(D.BigRational(rat(n.get("payoff_p0"), f"node {nid} payoff")))
            continue
        if kind == "chance":
            kids = [o.get("child") for o in edges(n, nid, "outcomes")]
        elif kind == "decision":
            kids = [a.get("child") for a in edges(n, nid, "actions")]
        else:
            raise Malformed(f"game: node {nid} has unknown kind {kind!r}")
        if not done:
            if nid in seen:
                raise Malformed(f"game: node {nid} appears twice (not a tree)")
            seen.add(nid)
            stack.append((nid, True))
            for c in reversed(kids):
                stack.append((c, False))
            continue
        kid_nodes = D.SeqWithoutIsStrInference([built.pop(c) for c in kids])
        if kind == "chance":
            qs = [rat(o.get("prob"), f"node {nid} prob") for o in n["outcomes"]]
            if not all(0 < q <= 1 for q in qs):
                raise Malformed(f"game: node {nid} has a chance probability outside (0, 1]")
            probs = [D.BigRational(q) for q in qs]
            built[nid] = G.Node_Chance(D.SeqWithoutIsStrInference(probs), kid_nodes)
        else:
            player = n.get("player")
            if not is_int(player) or player not in (0, 1):
                raise Malformed(f"game: node {nid} player {player!r} is not 0 or 1")
            name = n.get("infoset")
            if not isinstance(name, str):
                raise Malformed(f"game: node {nid} has no infoset string")
            acts = [a.get("label") for a in n["actions"]]
            if name not in infoset_ids:
                infoset_ids[name] = len(labels)
                labels.append(acts)
                owner.append(player)
            I = infoset_ids[name]
            if labels[I] != acts:
                raise Malformed(f"game: infoset {name!r} lists different actions at node {nid}")
            if owner[I] != player:
                raise Malformed(f"game: infoset {name!r} has nodes of both players")
            built[nid] = G.Node_Decision(player, I, kid_nodes)
    if len(seen) != len(nodes):
        raise Malformed(f"game: {len(nodes) - len(seen)} node(s) are not reachable from the root")
    names = [None] * len(labels)
    for name, I in infoset_ids.items():
        names[I] = name
    return built[root_id], [len(a) for a in labels], names, labels, owner


def build_profile(strat: dict, names, labels, D):
    if strat.get("format") != "bluff-strategy-v1":
        raise Malformed("strategy: format is not bluff-strategy-v1")
    probs = strat.get("probs")
    if not isinstance(probs, dict):
        raise Malformed("strategy: probs is not an object")
    rows = []
    for name, acts in zip(names, labels):
        row = probs.get(name)
        if not isinstance(row, dict):
            raise Malformed(f"strategy: infoset {name!r} is missing")
        extra = set(row) - set(acts)
        if extra:
            raise Malformed(f"strategy: infoset {name!r} has unknown actions {sorted(extra)}")
        vals = []
        for a in acts:
            if a not in row:
                raise Malformed(f"strategy: infoset {name!r} is missing action {a!r}")
            q = rat(row[a], f"strategy {name!r}/{a!r}")
            if q < 0:
                raise Malformed(f"strategy: infoset {name!r} action {a!r} is negative")
            vals.append(q)
        total = sum(vals, Fraction(0))
        if abs(total - 1) > TOL:
            raise Malformed(f"strategy: infoset {name!r} sums to {float(total)!r}, not 1 within 1e-6")
        rows.append(D.SeqWithoutIsStrInference([D.BigRational(v / total) for v in vals]))
    return D.SeqWithoutIsStrInference(rows)


def run(game_path: str, strat_path: str) -> dict:
    D, C, G = load_core()
    try:
        game = json.loads(Path(game_path).read_text(), object_pairs_hook=no_dup_keys)
        strat = json.loads(Path(strat_path).read_text(), object_pairs_hook=no_dup_keys)
    except (OSError, json.JSONDecodeError) as e:
        raise Malformed(f"cannot read input: {e}") from e
    if not isinstance(game, dict) or not isinstance(strat, dict):
        raise Malformed("game and strategy must be JSON objects")
    try:
        root, arity, names, labels, owner = build_game(game, D, G)
        prof = build_profile(strat, names, labels, D)
    except (AttributeError, TypeError, KeyError) as e:
        # A structurally wrong document (a number where an object belongs, ...)
        # is malformed input: exit 2, never a traceback.
        raise Malformed(f"malformed input: {type(e).__name__}: {e}") from e

    res = C.default__.Check(C.GameT_GameT(root, D.SeqWithoutIsStrInference(arity)), prof)
    if res.is_Fail:
        raise Malformed("core rejected input: " + "".join(res.reason))
    v = res.verdict

    def br_map(br, player):
        return {names[I]: labels[I][br[I]] for I in range(len(names)) if owner[I] == player}

    return {
        "format": "bluff-verdict-v1",
        "game": game.get("name"),
        "value_p0": fmt(v.value__p0),
        "br_value_p0": fmt(v.br__value__p0),
        "br_value_p1": fmt(v.br__value__p1),
        "nash_conv": fmt(v.nash__conv),
        "exploitability": fmt(v.exploitability),
        "exploitability_float": float(v.exploitability),
        "best_response_p0": br_map(v.br0, 0),
        "best_response_p1": br_map(v.br1, 1),
        "verified": True,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: check.py GAME.game.json STRATEGY.strategy.json", file=sys.stderr)
        return 2
    out: dict = {}

    def work():
        try:
            out["verdict"] = run(argv[1], argv[2])
        except Malformed as e:
            out["error"] = str(e)

    # The compiled core recurses once per list step; give it a deep stack.
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(1 << 29)
    t = threading.Thread(target=work)
    t.start()
    t.join()
    if "error" in out:
        print(out["error"], file=sys.stderr)
        return 2
    if "verdict" not in out:
        return 1
    print(json.dumps(out["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
