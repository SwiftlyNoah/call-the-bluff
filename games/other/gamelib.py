"""Shared helpers for the games in games/other/.

Everything here is exact (fractions.Fraction) except the LP in `lp_bracket`, which only proposes a
profile; the bracket it reports is computed exactly by `best_response` on the rounded profile.

This module is deliberately self-contained so the other-games task does not depend on files other
agents are writing at the same time (reference/validate_game.py, reference/br.py, reference/lp.py).
Those are the project's real oracles; this is a local sanity layer.
"""

from __future__ import annotations

import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

FORMAT = "bluff-game-v1"
OUT_DIR = Path(__file__).resolve().parents[1] / "out" / "other"


# ---------------------------------------------------------------- building


class Tree:
    """Builds a node list in preorder, so the root is node 0."""

    def __init__(self) -> None:
        self.nodes: list[dict | None] = []

    def _reserve(self) -> int:
        self.nodes.append(None)
        return len(self.nodes) - 1

    def terminal(self, payoff_p0: Fraction | int) -> int:
        nid = self._reserve()
        self.nodes[nid] = {"id": nid, "kind": "terminal", "payoff_p0": fstr(payoff_p0)}
        return nid

    def chance(self, outcomes) -> int:
        """outcomes: iterable of (label, prob, build_child) where build_child() returns a node id."""
        nid = self._reserve()
        outs = []
        for label, prob, build in outcomes:
            outs.append({"label": label, "prob": fstr(prob), "child": build()})
        self.nodes[nid] = {"id": nid, "kind": "chance", "outcomes": outs}
        return nid

    def decision(self, player: int, infoset: str, actions) -> int:
        """actions: iterable of (label, build_child)."""
        nid = self._reserve()
        acts = []
        for label, build in actions:
            acts.append({"label": label, "child": build()})
        self.nodes[nid] = {
            "id": nid,
            "kind": "decision",
            "player": player,
            "infoset": infoset,
            "actions": acts,
        }
        return nid


def fstr(x: Fraction | int) -> str:
    return str(Fraction(x))


def game_doc(name: str, description: str, tree: Tree, known_value=None, source=None) -> dict:
    doc = {"format": FORMAT, "name": name, "description": description}
    if known_value is not None:
        assert source, "known_value_source is required with known_value_p0"
        doc["known_value_p0"] = fstr(known_value)
        doc["known_value_source"] = source
    doc["root"] = 0
    doc["nodes"] = tree.nodes
    return doc


def write_game(doc: dict, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{doc['name']}.game.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


def load_game(path) -> dict:
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- validation and stats


def by_id(game: dict) -> dict[int, dict]:
    return {n["id"]: n for n in game["nodes"]}


def validate(game: dict) -> dict:
    """Checks the CONTRACTS.md rules this task relies on. Returns stats. Raises AssertionError."""
    assert game["format"] == FORMAT
    nodes = by_id(game)
    assert len(nodes) == len(game["nodes"]), "duplicate node ids"
    seen_child: set[int] = set()
    infosets: dict[str, tuple[int, tuple[str, ...]]] = {}
    own_seq: dict[str, tuple] = {}
    counts = defaultdict(int)
    max_depth = 0
    chance_depths: set[int] = set()

    def walk(nid: int, depth: int, seqs: tuple[tuple, tuple]) -> None:
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        n = nodes[nid]
        counts[n["kind"]] += 1
        if n["kind"] == "terminal":
            Fraction(n["payoff_p0"])
            return
        kids = n["outcomes"] if n["kind"] == "chance" else n["actions"]
        assert kids, f"node {nid} has no children"
        labels = [k["label"] for k in kids]
        assert len(set(labels)) == len(labels), f"node {nid} repeats a label"
        for k in kids:
            assert k["child"] not in seen_child, f"child {k['child']} appears twice"
            seen_child.add(k["child"])
        if n["kind"] == "chance":
            chance_depths.add(depth)
            probs = [Fraction(k["prob"]) for k in kids]
            assert all(0 < p <= 1 for p in probs), f"node {nid} prob out of range"
            assert sum(probs) == 1, f"node {nid} probs sum to {sum(probs)}"
            for k in kids:
                walk(k["child"], depth + 1, seqs)
            return
        p, iset = n["player"], n["infoset"]
        assert p in (0, 1)
        sig = (p, tuple(labels))
        assert infosets.setdefault(iset, sig) == sig, f"infoset {iset} inconsistent"
        # perfect recall: every node of an infoset has the same own (infoset, action) history
        assert own_seq.setdefault(iset, seqs[p]) == seqs[p], f"perfect recall fails at {iset}"
        for k in kids:
            new = list(seqs)
            new[p] = seqs[p] + ((iset, k["label"]),)
            walk(k["child"], depth + 1, tuple(new))

    walk(game["root"], 0, ((), ()))
    assert seen_child | {game["root"]} == set(nodes), "unreachable nodes"
    if "known_value_p0" in game:
        Fraction(game["known_value_p0"])
        assert game.get("known_value_source")
    return {
        "nodes": len(nodes),
        "terminal": counts["terminal"],
        "chance": counts["chance"],
        "decision": counts["decision"],
        "infosets_p0": sum(1 for _, (p, _) in infosets.items() if p == 0),
        "infosets_p1": sum(1 for _, (p, _) in infosets.items() if p == 1),
        "max_depth": max_depth,
        "chance_depths": sorted(chance_depths),
    }


def infoset_actions(game: dict) -> dict[str, tuple[int, list[str]]]:
    out = {}
    for n in game["nodes"]:
        if n["kind"] == "decision":
            out[n["infoset"]] = (n["player"], [a["label"] for a in n["actions"]])
    return out


def uniform_profile(game: dict) -> dict:
    return {
        iset: {a: Fraction(1, len(acts)) for a in acts} for iset, (_, acts) in infoset_actions(game).items()
    }


# ---------------------------------------------------------------- exact evaluation


def expected_value(game: dict, profile: dict) -> Fraction:
    nodes = by_id(game)

    def ev(nid: int) -> Fraction:
        n = nodes[nid]
        if n["kind"] == "terminal":
            return Fraction(n["payoff_p0"])
        if n["kind"] == "chance":
            return sum((Fraction(o["prob"]) * ev(o["child"]) for o in n["outcomes"]), Fraction(0))
        row = profile[n["infoset"]]
        return sum(
            (Fraction(row[a["label"]]) * ev(a["child"]) for a in n["actions"] if row[a["label"]]),
            Fraction(0),
        )

    return ev(game["root"])


def best_response(game: dict, profile: dict, p: int) -> tuple[Fraction, dict[str, str]]:
    """Exact best-response value for player p (in p's own utility) against the other player's
    part of `profile`, and a pure best response. Needs perfect recall."""
    nodes = by_id(game)
    sign = 1 if p == 0 else -1
    members: dict[str, list[tuple[int, Fraction]]] = defaultdict(list)

    # pass 1: reach probability of chance and the opponent to every node of p
    def reach(nid: int, w: Fraction) -> None:
        n = nodes[nid]
        if n["kind"] == "terminal" or w == 0:
            return
        if n["kind"] == "chance":
            for o in n["outcomes"]:
                reach(o["child"], w * Fraction(o["prob"]))
        elif n["player"] == p:
            members[n["infoset"]].append((nid, w))
            for a in n["actions"]:
                reach(a["child"], w)
        else:
            row = profile[n["infoset"]]
            for a in n["actions"]:
                reach(a["child"], w * Fraction(row[a["label"]]))

    reach(game["root"], Fraction(1))

    best: dict[str, str] = {}
    memo: dict[int, Fraction] = {}

    # value(nid): expected utility to p of the subtree, weighted by chance and opponent reach
    # INSIDE the subtree only (the caller multiplies by reach above it).
    def value(nid: int) -> Fraction:
        if nid in memo:
            return memo[nid]
        n = nodes[nid]
        if n["kind"] == "terminal":
            v = sign * Fraction(n["payoff_p0"])
        elif n["kind"] == "chance":
            v = sum((Fraction(o["prob"]) * value(o["child"]) for o in n["outcomes"]), Fraction(0))
        elif n["player"] != p:
            row = profile[n["infoset"]]
            v = sum(
                (Fraction(row[a["label"]]) * value(a["child"]) for a in n["actions"] if row[a["label"]]),
                Fraction(0),
            )
        else:
            a = choose(n["infoset"])
            v = value(next(x["child"] for x in n["actions"] if x["label"] == a))
        memo[nid] = v
        return v

    def choose(iset: str) -> str:
        if iset in best:
            return best[iset]
        labels = None
        totals: dict[str, Fraction] = {}
        for nid, w in members[iset]:
            n = nodes[nid]
            labels = [a["label"] for a in n["actions"]]
            for a in n["actions"]:
                totals[a["label"]] = totals.get(a["label"], Fraction(0)) + w * value(a["child"])
        if labels is None:  # infoset unreachable by chance and opponent: any action is a BR
            labels = infoset_actions(game)[iset][1]
            best[iset] = labels[0]
            return best[iset]
        best[iset] = max(labels, key=lambda a: totals[a])  # first maximizer on ties
        return best[iset]

    v = value(game["root"])
    for iset, (q, _) in infoset_actions(game).items():
        if q == p:
            choose(iset)
    return v, best


def exploitability(game: dict, profile: dict) -> dict:
    b0, _ = best_response(game, profile, 0)
    b1, _ = best_response(game, profile, 1)
    return {"br0": b0, "br1": b1, "nash_conv": b0 + b1, "exploitability": (b0 + b1) / 2}


# ---------------------------------------------------------------- sequence-form LP (float) with exact bracket


def _sequences(game: dict):
    nodes = by_id(game)
    seq_index = [{(): 0}, {(): 0}]
    parent = [{}, {}]
    payoff: dict[tuple[int, int], float] = defaultdict(float)

    def walk(nid: int, s: list, w: Fraction) -> None:
        n = nodes[nid]
        if n["kind"] == "terminal":
            payoff[(seq_index[0][s[0]], seq_index[1][s[1]])] += float(w * Fraction(n["payoff_p0"]))
            return
        if n["kind"] == "chance":
            for o in n["outcomes"]:
                walk(o["child"], s, w * Fraction(o["prob"]))
            return
        p, iset = n["player"], n["infoset"]
        parent[p].setdefault(iset, s[p])
        for a in n["actions"]:
            key = (iset, a["label"])
            seq_index[p].setdefault(key, len(seq_index[p]))
            t = list(s)
            t[p] = key
            walk(a["child"], t, w)

    walk(game["root"], [(), ()], Fraction(1))
    return seq_index, parent, payoff


def lp_solve(game: dict) -> tuple[float, dict, dict]:
    """Float sequence-form LP (scipy HiGHS). Returns (value_p0, behavioral p0, behavioral p1)."""
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, hstack

    seq_index, parent, payoff = _sequences(game)
    n0, n1 = len(seq_index[0]), len(seq_index[1])
    acts = infoset_actions(game)

    def constraint_matrix(p: int):
        isets = [iset for iset, (q, _) in acts.items() if q == p]
        rows, cols, vals = [0], [0], [1.0]
        for r, iset in enumerate(isets, start=1):
            for a in acts[iset][1]:
                rows.append(r)
                cols.append(seq_index[p][(iset, a)])
                vals.append(1.0)
            rows.append(r)
            cols.append(seq_index[p][parent[p][iset]])
            vals.append(-1.0)
        m = len(isets) + 1
        rhs = np.zeros(m)
        rhs[0] = 1.0
        return coo_matrix((vals, (rows, cols)), shape=(m, len(seq_index[p]))).tocsr(), rhs

    E, e = constraint_matrix(0)
    F, f = constraint_matrix(1)
    keys = list(payoff)
    A = coo_matrix(
        ([payoff[k] for k in keys], ([k[0] for k in keys], [k[1] for k in keys])), shape=(n0, n1)
    ).tocsr()

    # p0: max q0  s.t.  F^T q - A^T x <= 0,  E x = e,  x >= 0
    m1 = F.shape[0]
    c = np.zeros(n0 + m1)
    c[n0] = -1.0
    A_ub = hstack([-A.T, F.T]).tocsr()
    A_eq = hstack([E, coo_matrix((E.shape[0], m1))]).tocsr()
    bounds = [(0, None)] * n0 + [(None, None)] * m1
    r0 = linprog(c, A_ub=A_ub, b_ub=np.zeros(n1), A_eq=A_eq, b_eq=e, bounds=bounds, method="highs")
    assert r0.status == 0, r0.message
    x = r0.x[:n0]

    # p1: min p0  s.t.  -E^T p + A y <= 0,  F y = f,  y >= 0
    m0 = E.shape[0]
    c = np.zeros(n1 + m0)
    c[n1] = 1.0
    A_ub = hstack([A, -E.T]).tocsr()
    A_eq = hstack([F, coo_matrix((F.shape[0], m0))]).tocsr()
    bounds = [(0, None)] * n1 + [(None, None)] * m0
    r1 = linprog(c, A_ub=A_ub, b_ub=np.zeros(n0), A_eq=A_eq, b_eq=f, bounds=bounds, method="highs")
    assert r1.status == 0, r1.message
    y = r1.x[:n1]

    def behavioral(p: int, z) -> dict:
        out = {}
        for iset, (q, labels) in acts.items():
            if q != p:
                continue
            par = z[seq_index[p][parent[p][iset]]]
            if par > 1e-12:
                out[iset] = {a: max(0.0, z[seq_index[p][(iset, a)]] / par) for a in labels}
            else:
                out[iset] = {a: 1.0 / len(labels) for a in labels}
        return out

    return -r0.fun, behavioral(0, x), behavioral(1, y)


def rationalize(beh: dict, max_den: int) -> dict:
    out = {}
    for iset, row in beh.items():
        r = {a: Fraction(v).limit_denominator(max_den) for a, v in row.items()}
        r = {a: max(Fraction(0), v) for a, v in r.items()}
        s = sum(r.values())
        if s == 0:
            r = {a: Fraction(1, len(r)) for a in r}
        else:
            r = {a: v / s for a, v in r.items()}
        out[iset] = r
    return out


def lp_bracket(game: dict, dens=(12, 100, 1000, 10**4, 10**6, 10**9)) -> dict:
    """Solves the LP in floats, rounds the profile to rationals and brackets the value EXACTLY:
    value_p0 is in [lower, upper] with lower = -BR_1(x) and upper = BR_0(y). When lower == upper the
    rounded profile is an exact equilibrium and the value is certified by exact best response."""
    v, b0, b1 = lp_solve(game)
    best = None
    for d in dens:
        prof = {**rationalize(b0, d), **rationalize(b1, d)}
        lower = -best_response(game, prof, 1)[0]
        upper = best_response(game, prof, 0)[0]
        cand = {"lp_float": v, "lower": lower, "upper": upper, "max_den": d, "profile": prof}
        if best is None or upper - lower < best["upper"] - best["lower"]:
            best = cand
        if lower == upper:
            break
    return best
