"""Thin, TRUSTED adapter for attempt-b's verified core.

Usage: uv run dafny/attempt-b/check.py GAME.game.json STRATEGY.strategy.json

Reads a CONTRACTS.md game file and strategy file, builds the Dafny values (flat node table,
heights, infoset table, a normalized exact profile), calls the compiled, verified Checker.Check,
and prints a bluff-verdict-v1 JSON. Exit 2 on a malformed game or strategy, reason on stderr.

What this file does (and is trusted for): parsing JSON and rational strings, numbering nodes and
infosets, checking that every node of an infoset lists the same action labels, computing node
heights, normalizing strategy rows (sum within 1e-6 of 1, then divided by the exact sum), and
printing. Everything else (well-formedness of the table, the profile being a distribution, every
number in the verdict) comes from the verified core.
"""

import json
import os
import subprocess
import sys
import threading
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build", "bluffcore-py")
TOL = Fraction(1, 10**6)


class Malformed(Exception):
    pass


def build_is_stale():
    """True when the compiled core is missing or older than any .dfy source."""
    marker = os.path.join(BUILD, "Checker.py")
    if not os.path.exists(marker):
        return True
    built = os.path.getmtime(marker)
    return any(os.path.getmtime(os.path.join(HERE, f)) > built
               for f in os.listdir(HERE) if f.endswith(".dfy"))


def load_core():
    # Never run a stale or unverified build: rebuild WITH verification (no --no-verify) whenever
    # any source is newer than the compiled core, so "verified": true always means the code that
    # ran is the code Dafny just verified.
    if build_is_stale():
        subprocess.run(["dafny", "build", "--target:py", "Check.dfy", "--cores", "2",
                        "-o", "build/bluffcore"], cwd=HERE, check=True,
                       stdout=subprocess.DEVNULL)
    sys.path.insert(0, BUILD)
    import _dafny  # noqa: E402
    import GameModel  # noqa: E402
    import Checker  # noqa: E402
    return _dafny, GameModel, Checker


def rat(s, what):
    if not isinstance(s, str):
        raise Malformed(f"{what}: expected an exact rational string, got {s!r}")
    try:
        return Fraction(s.strip())
    except (ValueError, ZeroDivisionError):
        raise Malformed(f"{what}: not a rational: {s!r}")


def parse_game(game):
    if game.get("format") != "bluff-game-v1":
        raise Malformed("game: format is not bluff-game-v1")
    raw = game.get("nodes")
    if not isinstance(raw, list) or not raw:
        raise Malformed("game: no nodes")
    ids = [nd.get("id") for nd in raw]
    if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in ids):
        raise Malformed("game: node ids must be non-negative integers")
    if len(set(ids)) != len(ids):
        raise Malformed("game: duplicate node id")
    index = {i: k for k, i in enumerate(sorted(ids))}
    by_index = [None] * len(ids)
    for nd in raw:
        by_index[index[nd["id"]]] = nd
    root = game.get("root")
    if root not in index:
        raise Malformed("game: root is not a node id")

    def child(c, where):
        if not isinstance(c, int) or isinstance(c, bool) or c not in index:
            raise Malformed(f"{where}: child {c!r} is not a node id")
        return index[c]

    infoset_index, infoset_player, infoset_labels, infoset_names = {}, [], [], []
    nodes = []  # (kind, payload) in index order
    for k, nd in enumerate(by_index):
        where = f"node {nd['id']}"
        kind = nd.get("kind")
        if kind == "terminal":
            nodes.append(("T", rat(nd.get("payoff_p0"), where)))
        elif kind == "chance":
            outs = nd.get("outcomes") or []
            probs = [rat(o.get("prob"), where) for o in outs]
            kids = [child(o.get("child"), where) for o in outs]
            nodes.append(("C", probs, kids))
        elif kind == "decision":
            player, name = nd.get("player"), nd.get("infoset")
            if type(player) is not int or player not in (0, 1) or not isinstance(name, str):
                raise Malformed(f"{where}: bad player or infoset")
            acts = nd.get("actions") or []
            labels = [a.get("label") for a in acts]
            if name not in infoset_index:
                infoset_index[name] = len(infoset_player)
                infoset_player.append(player)
                infoset_labels.append(labels)
                infoset_names.append(name)
            iset = infoset_index[name]
            if infoset_player[iset] != player or infoset_labels[iset] != labels:
                raise Malformed(f"{where}: infoset {name!r} has inconsistent player or actions")
            if len(set(labels)) != len(labels):
                raise Malformed(f"{where}: duplicate action labels")
            kids = [child(a.get("child"), where) for a in acts]
            nodes.append(("D", player, iset, kids))
        else:
            raise Malformed(f"{where}: unknown kind {kind!r}")

    # Heights (longest path to a leaf), iteratively; a cycle is rejected.
    n = len(nodes)
    height = [None] * n
    state = [0] * n  # 0 new, 1 on stack, 2 done
    for start in range(n):
        if state[start]:
            continue
        stack = [(start, 0)]
        state[start] = 1
        while stack:
            v, i = stack[-1]
            kids = [] if nodes[v][0] == "T" else nodes[v][-1]
            if i < len(kids):
                stack[-1] = (v, i + 1)
                c = kids[i]
                if state[c] == 1:
                    raise Malformed("game: the node graph has a cycle")
                if state[c] == 0:
                    state[c] = 1
                    stack.append((c, 0))
            else:
                height[v] = 1 + max((height[c] for c in kids), default=-1)
                state[v] = 2
                stack.pop()
    return {
        "name": game.get("name", ""), "nodes": nodes, "root": index[root], "height": height,
        "infoset_player": infoset_player, "infoset_labels": infoset_labels,
        "infoset_names": infoset_names, "infoset_index": infoset_index,
    }


def parse_strategy(strat, G):
    if strat.get("format") != "bluff-strategy-v1":
        raise Malformed("strategy: format is not bluff-strategy-v1")
    probs = strat.get("probs")
    if not isinstance(probs, dict):
        raise Malformed("strategy: no probs object")
    rows = []
    for name, labels in zip(G["infoset_names"], G["infoset_labels"]):
        row = probs.get(name)
        if not isinstance(row, dict):
            raise Malformed(f"strategy: infoset {name!r} missing")
        if set(row) != set(labels):
            raise Malformed(f"strategy: infoset {name!r} actions {sorted(row)} != {sorted(labels)}")
        vals = [rat(row[a], f"strategy {name!r}") for a in labels]
        if any(v < 0 for v in vals):
            raise Malformed(f"strategy: infoset {name!r} has a negative probability")
        total = sum(vals)
        if abs(total - 1) > TOL:
            raise Malformed(f"strategy: infoset {name!r} sums to {float(total)}, not 1 within 1e-6")
        rows.append([v / total for v in vals])
    return rows


def fmt(q):
    q = Fraction(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def run(game_path, strat_path):
    with open(game_path) as f:
        game = json.load(f)
    with open(strat_path) as f:
        strat = json.load(f)
    try:
        G = parse_game(game)
        rows = parse_strategy(strat, G)
    except (AttributeError, TypeError, KeyError, IndexError) as e:
        # JSON of the wrong shape (a node that is not an object, a missing key, ...) is malformed
        # input, exit 2 per CONTRACTS.md, not an internal error.
        raise Malformed(f"malformed input ({type(e).__name__}: {e})")
    _dafny, GM, CK = load_core()
    R = lambda q: _dafny.BigRational(q)  # noqa: E731
    S = _dafny.SeqWithoutIsStrInference
    dnodes = []
    for nd in G["nodes"]:
        if nd[0] == "T":
            dnodes.append(GM.Node_Terminal(R(nd[1])))
        elif nd[0] == "C":
            dnodes.append(GM.Node_Chance(S([R(p) for p in nd[1]]), S(nd[2])))
        else:
            dnodes.append(GM.Node_Decision(nd[1], nd[2], S(nd[3])))
    g = GM.Game_Game(S(dnodes), G["root"], S(G["height"]), S(G["infoset_player"]),
                     S([len(labs) for labs in G["infoset_labels"]]))
    sigma = S([S([R(p) for p in row]) for row in rows])
    res = CK.default__.Check(g, sigma)
    if res.is_Err:
        raise Malformed("core rejected the input: " + "".join(res.msg))
    v = res.v

    def br(pi, p):
        return {G["infoset_names"][k]: G["infoset_labels"][k][pi[k]]
                for k in range(len(pi)) if G["infoset_player"][k] == p}

    return {
        "format": "bluff-verdict-v1",
        "game": G["name"] or strat.get("game", ""),
        "value_p0": fmt(v.valueP0),
        "br_value_p0": fmt(v.brValue0),
        "br_value_p1": fmt(v.brValue1),
        "nash_conv": fmt(v.nashConv),
        "exploitability": fmt(v.exploitability),
        "exploitability_float": float(Fraction(v.exploitability)),
        "best_response_p0": br(list(v.bestResponse0), 0),
        "best_response_p1": br(list(v.bestResponse1), 1),
        "verified": True,
    }


def main():
    if len(sys.argv) != 3:
        print("usage: check.py GAME.game.json STRATEGY.strategy.json", file=sys.stderr)
        sys.exit(2)
    out = {}

    def work():
        try:
            out["verdict"] = run(sys.argv[1], sys.argv[2])
        except (Malformed, json.JSONDecodeError, OSError) as e:
            out["error"] = str(e)

    # The compiled core recurses once per node on the pending list; give it room.
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(1 << 29)
    t = threading.Thread(target=work)
    t.start()
    t.join()
    if "error" in out:
        print(out["error"], file=sys.stderr)
        sys.exit(2)
    if "verdict" not in out:
        sys.exit(1)
    print(json.dumps(out["verdict"], indent=1))


if __name__ == "__main__":
    main()
