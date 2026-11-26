"""Sequence-form LP oracle with an exactly certified rational equilibrium.

Independent of the Dafny core and of reference/br.py. For a CONTRACTS.md game file
(bluff-game-v1) this module:

1. builds the sequence-form payoff matrix A and constraint matrices E, F
   (Koller, Megiddo and von Stengel 1996) from the tree,
2. solves both players' zero-sum LPs with scipy's HiGHS in floating point,
3. rounds each player's realization plan to rationals (Fraction.limit_denominator with
   increasing bounds, plus an exact vertex reconstruction from the LP's active set when
   rounding is not enough), converts it to an exact behavioral strategy,
4. certifies in exact Fraction arithmetic, with its own best response, that
   lo = min over p1 strategies of p0's payoff against x  equals
   hi = max over p0 strategies of p0's payoff against y.
   lo == hi proves both strategies optimal and that the game value is exactly lo
   (weak duality), i.e. the profile has exploitability exactly 0.

Never reports exactness it did not certify: if lo < hi it says so and reports
(hi - lo) / 2, the exploitability of the best rational profile it found.

CLI:
    uv run reference/lp.py GAME.game.json [...]      # writes GAME.lp-strategy.json next to each
    uv run reference/lp.py --no-write GAME.game.json  # certify only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack

sys.setrecursionlimit(100000)

# ---------------------------------------------------------------------------
# Game loading
# ---------------------------------------------------------------------------


# CONTRACTS.md section 1: an integer, a fraction a/b or a finite decimal, nothing else (no
# exponents, spaces or underscores, which Fraction() would otherwise accept). Same grammar as
# reference/fractions_util.GAME_RATIONAL_RE, kept local so this oracle stays independent.
_RATIONAL_RE = re.compile(r"^-?(?:[0-9]+(?:\.[0-9]+)?|[0-9]+/0*[1-9][0-9]*)$")


def parse_rational(s) -> Fraction:
    """Exact rational from a contract string: integer, a/b, or finite decimal."""
    if not isinstance(s, str) or not _RATIONAL_RE.match(s):
        raise ValueError(f"not an exact rational string: {s!r}")
    return Fraction(s)


@dataclass
class Game:
    name: str
    root: int
    nodes: dict  # id -> node dict, with "prob"/"payoff_p0" parsed to Fraction
    infosets: dict = field(default_factory=dict)  # name -> {"player", "actions", "nodes"}
    known_value_p0: Fraction | None = None


def load_game(path: str | Path) -> Game:
    with open(path) as fh:
        raw = json.load(fh)
    return game_from_dict(raw)


def game_from_dict(raw: dict) -> Game:
    if raw.get("format") != "bluff-game-v1":
        raise ValueError(f"unexpected format {raw.get('format')!r}")
    nodes = {}
    for n in raw["nodes"]:
        nid = n["id"]
        if nid in nodes:
            raise ValueError(f"duplicate node id {nid}")
        n = dict(n)
        if n["kind"] == "chance":
            n["outcomes"] = [dict(o, prob=parse_rational(o["prob"])) for o in n["outcomes"]]
            if not n["outcomes"] or any(not 0 < o["prob"] <= 1 for o in n["outcomes"]):
                raise ValueError(f"chance node {nid}: needs outcomes with probabilities in (0, 1]")
            total = sum(o["prob"] for o in n["outcomes"])
            if total != 1:
                raise ValueError(f"chance node {nid} probabilities sum to {total}")
        elif n["kind"] == "terminal":
            n["payoff_p0"] = parse_rational(n["payoff_p0"])
        elif n["kind"] == "decision":
            labels = [a["label"] for a in n["actions"]]
            if n["player"] not in (0, 1) or isinstance(n["player"], bool):
                raise ValueError(f"node {nid}: player must be 0 or 1")
            if not labels or len(set(labels)) != len(labels):
                raise ValueError(f"node {nid}: actions must be non-empty with distinct labels")
        else:
            raise ValueError(f"node {nid}: unknown kind {n['kind']!r}")
        nodes[nid] = n
    kv = raw.get("known_value_p0")
    g = Game(raw.get("name", "?"), raw["root"], nodes,
             known_value_p0=parse_rational(kv) if kv is not None else None)
    for n in nodes.values():
        if n["kind"] != "decision":
            continue
        labels = [a["label"] for a in n["actions"]]
        info = g.infosets.setdefault(n["infoset"], {"player": n["player"], "actions": labels, "nodes": []})
        if info["player"] != n["player"] or info["actions"] != labels:
            raise ValueError(f"infoset {n['infoset']!r} is inconsistent across nodes")
        info["nodes"].append(n["id"])
    return g


def children(n: dict) -> list[int]:
    if n["kind"] == "chance":
        return [o["child"] for o in n["outcomes"]]
    if n["kind"] == "decision":
        return [a["child"] for a in n["actions"]]
    return []


# ---------------------------------------------------------------------------
# Sequence form
# ---------------------------------------------------------------------------

EMPTY = None  # the empty sequence


@dataclass
class SequenceForm:
    seqs: list  # per player: list of sequences, index 0 = EMPTY, others (infoset, action)
    index: list  # per player: seq -> column
    parent: dict  # infoset -> parent sequence (of the owning player)
    infoset_order: list  # per player: infosets in constraint-row order (row r+1)
    A: dict  # (i0, i1) -> Fraction, chance-weighted payoff to p0
    E: list  # per player: sparse constraint matrix (rows = 1 + #infosets)


def build_sequence_form(g: Game) -> SequenceForm:
    seqs = [[EMPTY], [EMPTY]]
    index = [{EMPTY: 0}, {EMPTY: 0}]
    parent: dict = {}
    order = [[], []]
    A: dict = {}
    # Iterative DFS carrying (node, last sequence of p0, last sequence of p1, chance prob).
    stack = [(g.root, EMPTY, EMPTY, Fraction(1))]
    seen = set()
    while stack:
        nid, s0, s1, pc = stack.pop()
        if nid in seen:
            raise ValueError(f"node {nid} reached twice: not a tree")
        seen.add(nid)
        n = g.nodes[nid]
        k = n["kind"]
        if k == "terminal":
            key = (index[0][s0], index[1][s1])
            A[key] = A.get(key, Fraction(0)) + pc * n["payoff_p0"]
        elif k == "chance":
            for o in n["outcomes"]:
                stack.append((o["child"], s0, s1, pc * o["prob"]))
        else:
            p = n["player"]
            iset = n["infoset"]
            own = s0 if p == 0 else s1
            if iset in parent:
                if parent[iset] != own:
                    raise ValueError(f"perfect recall fails at infoset {iset!r}")
            else:
                parent[iset] = own
                order[p].append(iset)
                for a in g.infosets[iset]["actions"]:
                    index[p][(iset, a)] = len(seqs[p])
                    seqs[p].append((iset, a))
            for a in n["actions"]:
                seq = (iset, a["label"])
                if p == 0:
                    stack.append((a["child"], seq, s1, pc))
                else:
                    stack.append((a["child"], s0, seq, pc))
    if len(seen) != len(g.nodes):
        raise ValueError(f"{len(g.nodes) - len(seen)} nodes unreachable from the root")
    E = []
    for p in (0, 1):
        rows, cols, vals = [0], [0], [1.0]
        for r, iset in enumerate(order[p], start=1):
            rows.append(r)
            cols.append(index[p][parent[iset]])
            vals.append(-1.0)
            for a in g.infosets[iset]["actions"]:
                rows.append(r)
                cols.append(index[p][(iset, a)])
                vals.append(1.0)
        E.append(csr_matrix(coo_matrix((vals, (rows, cols)), shape=(1 + len(order[p]), len(seqs[p])))))
    return SequenceForm(seqs, index, parent, order, A, E)


def payoff_matrix(sf: SequenceForm) -> csr_matrix:
    keys = list(sf.A)
    rows = [k[0] for k in keys]
    cols = [k[1] for k in keys]
    vals = [float(sf.A[k]) for k in keys]
    return csr_matrix(coo_matrix((vals, (rows, cols)), shape=(len(sf.seqs[0]), len(sf.seqs[1]))))


def solve_side(B: csr_matrix, Eme: csr_matrix, Eopp: csr_matrix):
    """max_x min_y x^T B y over realization plans; returns (x, value, linprog result).

    Dual of the inner min: max_q e1^T q s.t. Eopp^T q <= B^T x. Variables z = [x, q].
    """
    n, m = B.shape
    kopp = Eopp.shape[0]
    c = np.zeros(n + kopp)
    c[n] = -1.0  # maximize q[0] (the root row of Eopp has right-hand side 1)
    A_ub = hstack([-B.T, Eopp.T]).tocsr()
    b_ub = np.zeros(m)
    A_eq = hstack([Eme, csr_matrix((Eme.shape[0], kopp))]).tocsr()
    b_eq = np.zeros(Eme.shape[0])
    b_eq[0] = 1.0
    bounds = [(0, None)] * n + [(None, None)] * kopp
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs-ds",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x[:n], -res.fun, res


# ---------------------------------------------------------------------------
# Exact strategies and exact best response (independent of reference/br.py)
# ---------------------------------------------------------------------------


def realization_to_behavioral(g: Game, sf: SequenceForm, p: int, x: list[Fraction]) -> dict:
    """Exact behavioral strategy from an exact (possibly slightly inconsistent) plan.

    Each infoset's row is x[iset, a] divided by the row's exact sum; a row summing to 0
    (unreachable under the plan) becomes uniform. The result is always a valid strategy.
    """
    beh = {}
    for iset in sf.infoset_order[p]:
        acts = g.infosets[iset]["actions"]
        row = [max(x[sf.index[p][(iset, a)]], Fraction(0)) for a in acts]
        tot = sum(row)
        if tot > 0:
            beh[iset] = {a: v / tot for a, v in zip(acts, row)}
        else:
            beh[iset] = {a: Fraction(1, len(acts)) for a in acts}
    return beh


def best_response_value(g: Game, beh: dict, p: int) -> tuple[Fraction, dict]:
    """Exact value to player p of a pure best response against beh (the other player's
    infosets read from beh; p's own entries in beh are ignored). Returns (value, pure BR).

    W(h) = sum over terminals z under h, with p's choices fixed to the BR at deeper
    infosets, of (chance and opponent reach from h to z) * u_p(z), scaled by the reach
    of h. Computed top-down for reach, then bottom-up per infoset with memoization;
    perfect recall makes the infoset choices well founded.
    """
    sign = 1 if p == 0 else -1
    reach: dict[int, Fraction] = {}
    stack = [(g.root, Fraction(1))]
    while stack:
        nid, r = stack.pop()
        reach[nid] = r
        n = g.nodes[nid]
        if n["kind"] == "chance":
            for o in n["outcomes"]:
                stack.append((o["child"], r * o["prob"]))
        elif n["kind"] == "decision":
            if n["player"] == p:
                for a in n["actions"]:
                    stack.append((a["child"], r))
            else:
                probs = beh[n["infoset"]]
                for a in n["actions"]:
                    stack.append((a["child"], r * probs[a["label"]]))
    W: dict[int, Fraction] = {}
    choice: dict[str, str] = {}

    def w(nid: int) -> Fraction:
        if nid in W:
            return W[nid]
        n = g.nodes[nid]
        if n["kind"] == "terminal":
            v = reach[nid] * sign * n["payoff_p0"]
        elif n["kind"] == "decision" and n["player"] == p:
            iset = n["infoset"]
            if iset not in choice:
                best_a, best_v = None, None
                for a in g.infosets[iset]["actions"]:
                    tot = Fraction(0)
                    for h in g.infosets[iset]["nodes"]:
                        tot += w(child_by_label(g.nodes[h], a))
                    if best_v is None or tot > best_v:
                        best_a, best_v = a, tot
                choice[iset] = best_a
            v = w(child_by_label(n, choice[iset]))
        else:
            v = sum((w(c) for c in children(n)), Fraction(0))
        W[nid] = v
        return v

    return w(g.root), choice


def child_by_label(n: dict, label: str) -> int:
    for a in n["actions"]:
        if a["label"] == label:
            return a["child"]
    raise KeyError(label)


def profile_value(g: Game, beh: dict) -> Fraction:
    total = Fraction(0)
    stack = [(g.root, Fraction(1))]
    while stack:
        nid, r = stack.pop()
        n = g.nodes[nid]
        if n["kind"] == "terminal":
            total += r * n["payoff_p0"]
        elif n["kind"] == "chance":
            for o in n["outcomes"]:
                stack.append((o["child"], r * o["prob"]))
        else:
            probs = beh[n["infoset"]]
            for a in n["actions"]:
                pa = probs[a["label"]]
                if pa:
                    stack.append((a["child"], r * pa))
    return total


def guarantee(g: Game, beh_p: dict, p: int) -> Fraction:
    """p0's payoff guaranteed by player p's strategy: lo for p=0, hi for p=1."""
    v, _ = best_response_value(g, beh_p, 1 - p)
    return -v if p == 0 else v


# ---------------------------------------------------------------------------
# Exact vertex reconstruction (fallback when rounding alone does not certify)
# ---------------------------------------------------------------------------


def exact_solve(rows: list[dict], ncols: int, rhs: list[Fraction]) -> list[Fraction] | None:
    """Solve a consistent sparse linear system exactly (Fraction Gaussian elimination).

    rows: list of {col: Fraction}. Free variables are set to 0. Returns None if inconsistent.
    """
    pivots: dict[int, tuple[dict, Fraction]] = {}  # pivot col -> (row normalized, rhs)
    for row, b in zip(rows, rhs):
        row = dict(row)
        # Eliminate existing pivots.
        changed = True
        while changed:
            changed = False
            for c in [c for c in row if c in pivots]:
                coef = row.pop(c)
                if coef == 0:
                    continue
                prow, pb = pivots[c]
                for cc, vv in prow.items():
                    if cc == c:
                        continue
                    nv = row.get(cc, Fraction(0)) - coef * vv
                    if nv:
                        row[cc] = nv
                    else:
                        row.pop(cc, None)
                b -= coef * pb
                changed = True
        row = {c: v for c, v in row.items() if v}
        if not row:
            if b != 0:
                return None
            continue
        pc = min(row, key=lambda c: c)
        coef = row[pc]
        row = {c: v / coef for c, v in row.items()}
        b = b / coef
        # Back-substitute into existing pivots to keep them reduced.
        for c, (prow, pb) in list(pivots.items()):
            if pc in prow:
                k = prow[pc]
                for cc, vv in row.items():
                    nv = prow.get(cc, Fraction(0)) - k * vv
                    if nv:
                        prow[cc] = nv
                    else:
                        prow.pop(cc, None)
                pivots[c] = (prow, pb - k * b)
        pivots[pc] = (row, b)
    # Rows are fully reduced: each pivot row holds its pivot and free columns only, so with
    # free variables at 0 each pivot variable equals its row's right-hand side.
    sol = [Fraction(0)] * ncols
    for c, (_, pb) in pivots.items():
        sol[c] = pb
    return sol


def vertex_plan(sf: SequenceForm, B: dict, p: int, z: np.ndarray, slack: np.ndarray, tol: float):
    """Rebuild the LP's optimal vertex exactly from its active set.

    Variables z = [x (n), q (k)]. Constraints: Eme x = e, Eopp^T q - B^T x <= 0 (tight
    where slack ~ 0), x_j = 0 where x_j ~ 0. Solving the active equalities exactly gives
    a rational point; returns its x part or None.
    """
    me, opp = p, 1 - p
    n = len(sf.seqs[me])
    k = 1 + len(sf.infoset_order[opp])
    rows, rhs = [], []
    # Eme x = e
    erow = [{0: Fraction(1)}]
    by_infoset = [{}, {}]  # player -> infoset -> its sequence columns
    for q in (0, 1):
        for s in sf.seqs[q][1:]:
            by_infoset[q].setdefault(s[0], []).append(sf.index[q][s])
    for iset in sf.infoset_order[me]:
        r = {sf.index[me][sf.parent[iset]]: Fraction(-1)}
        for j in by_infoset[me][iset]:
            r[j] = r.get(j, Fraction(0)) + 1
        erow.append(r)
    rows += erow
    rhs += [Fraction(1)] + [Fraction(0)] * (len(erow) - 1)
    # x_j = 0 where small
    for j in range(n):
        if abs(z[j]) <= tol:
            rows.append({j: Fraction(1)})
            rhs.append(Fraction(0))
    # tight dual rows: (Eopp^T q)_s - (B^T x)_s = 0
    Bt: dict[int, dict[int, Fraction]] = {}
    for (i, j), v in B.items():
        Bt.setdefault(j, {})[i] = v
    eopp_cols: dict[int, dict[int, Fraction]] = {}  # opp seq -> {row: coef}
    eopp_cols.setdefault(0, {})[0] = Fraction(1)
    for r, iset in enumerate(sf.infoset_order[opp], start=1):
        ps = sf.index[opp][sf.parent[iset]]
        eopp_cols.setdefault(ps, {})[r] = eopp_cols.get(ps, {}).get(r, Fraction(0)) - 1
        for j in by_infoset[opp][iset]:
            eopp_cols.setdefault(j, {})[r] = Fraction(1)
    for s in range(len(sf.seqs[opp])):
        if abs(slack[s]) <= tol:
            r = {n + row: c for row, c in eopp_cols.get(s, {}).items()}
            for i, v in Bt.get(s, {}).items():
                r[i] = r.get(i, Fraction(0)) - v
            rows.append(r)
            rhs.append(Fraction(0))
    sol = exact_solve(rows, n + k, rhs)
    if sol is None:
        return None
    return sol[:n]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class Result:
    game: str
    value: Fraction | None  # exact value when certified
    lo: Fraction  # best certified lower bound on the value (from p0's rational strategy)
    hi: Fraction  # best certified upper bound (from p1's rational strategy)
    certified: bool
    beh: dict  # behavioral profile for both players, exact
    lp_value_float: float
    method: dict  # how each player's strategy was recovered
    seconds: float
    sizes: dict


DENOMS = [10, 100, 1000, 10**4, 10**5, 10**6, 10**7, 10**8]


def candidate_plans(x: np.ndarray, vertex):
    """Yield (method, exact plan) candidates for a float plan, simplest first."""
    for d in DENOMS:
        yield f"limit_denominator({d})", [Fraction(float(v)).limit_denominator(d) for v in x]
    v = vertex()
    if v is not None:
        yield "exact vertex", v


def solve_game(g: Game) -> Result:
    t0 = time.time()
    sf = build_sequence_form(g)
    A = payoff_matrix(sf)
    x, v0, res0 = solve_side(A, sf.E[0], sf.E[1])
    BT = csr_matrix((-A).T)
    y, v1, res1 = solve_side(BT, sf.E[1], sf.E[0])  # v1 = value to p1
    lp_value = float(v0)
    Bp1 = {(j, i): -v for (i, j), v in sf.A.items()}

    plans = {0: x, 1: y}
    results = {0: res0, 1: res1}
    Bs = {0: sf.A, 1: Bp1}

    def vertex_for(p):
        def vertex():
            z = results[p].x
            slack = results[p].ineqlin.residual
            for tol in (1e-9, 1e-7, 1e-11):
                sol = vertex_plan(sf, Bs[p], p, z, slack, tol)
                if sol is not None and all(s >= 0 for s in sol):
                    return sol
            return None
        return vertex

    # Walk both sides' candidates in lockstep (simplest first); stop as soon as the best
    # certified lower bound from p0's side meets the best upper bound from p1's side.
    gens = {p: candidate_plans(plans[p], vertex_for(p)) for p in (0, 1)}
    best = {0: (None, None, None), 1: (None, None, None)}
    live = {0: True, 1: True}
    while any(live.values()):
        for p in (0, 1):
            if not live[p]:
                continue
            try:
                m, plan_q = next(gens[p])
            except StopIteration:
                live[p] = False
                continue
            beh = realization_to_behavioral(g, sf, p, plan_q)
            gv = guarantee(g, beh, p)  # p0 payoff this strategy guarantees
            bv = best[p][0]
            if bv is None or (gv > bv if p == 0 else gv < bv):
                best[p] = (gv, beh, m)
        if best[0][0] is not None and best[0][0] == best[1][0]:
            break

    lo, beh0, m0 = best[0]
    hi, beh1, m1 = best[1]
    if lo > hi:  # weak duality: impossible unless the exact best response is wrong
        raise AssertionError(f"internal error: certified lower bound {lo} exceeds upper bound {hi}")
    certified = lo == hi
    beh = {**beh0, **beh1}
    return Result(g.name, lo if certified else None, lo, hi, certified, beh, lp_value,
                  {"p0": m0, "p1": m1}, time.time() - t0,
                  {"nodes": len(g.nodes), "infosets": len(g.infosets),
                   "seqs_p0": len(sf.seqs[0]), "seqs_p1": len(sf.seqs[1])})


def certify_profile(g: Game, beh: dict) -> dict:
    """Exact nash_conv of a full behavioral profile, independent of how it was produced."""
    v = profile_value(g, beh)
    br0, _ = best_response_value(g, beh, 0)
    br1, _ = best_response_value(g, beh, 1)
    nc = br0 + br1
    return {"value_p0": v, "br_value_p0": br0, "br_value_p1": br1, "nash_conv": nc,
            "exploitability": nc / 2}


def frac_str(f: Fraction) -> str:
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def strategy_json(g: Game, r: Result) -> dict:
    status = (f"certified exact equilibrium, value_p0 {frac_str(r.value)}" if r.certified
              else f"NOT certified: exploitability {frac_str((r.hi - r.lo) / 2)}")
    return {
        "format": "bluff-strategy-v1",
        "game": g.name,
        "producer": f"reference/lp.py sequence-form LP (HiGHS) + exact rational certification; {status}",
        "probs": {iset: {a: frac_str(r.beh[iset][a]) for a in g.infosets[iset]["actions"]}
                  for iset in sorted(g.infosets)},
    }


def out_path(game_path: Path) -> Path:
    name = game_path.name
    stem = name[: -len(".game.json")] if name.endswith(".game.json") else game_path.stem
    return game_path.with_name(stem + ".lp-strategy.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("games", nargs="+")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    rc = 0
    for gp in map(Path, args.games):
        g = load_game(gp)
        r = solve_game(g)
        cert = certify_profile(g, r.beh)
        ok = r.certified and cert["exploitability"] == 0 and cert["value_p0"] == r.value
        report = {
            "game": g.name,
            "file": str(gp),
            "certified_exact": ok,
            "value_p0": frac_str(r.value) if ok else None,
            "value_p0_float": float(r.value) if ok else None,
            "lower_bound": frac_str(r.lo),
            "upper_bound": frac_str(r.hi),
            "exploitability": frac_str(cert["exploitability"]),
            "lp_value_float": r.lp_value_float,
            "known_value_p0": frac_str(g.known_value_p0) if g.known_value_p0 is not None else None,
            "known_value_matches": (r.value == g.known_value_p0) if (ok and g.known_value_p0 is not None) else None,
            "recovery": r.method,
            "sizes": r.sizes,
            "seconds": round(r.seconds, 2),
        }
        if not args.no_write:
            op = out_path(gp)
            with open(op, "w") as fh:
                json.dump(strategy_json(g, r), fh, indent=1)
                fh.write("\n")
            report["wrote"] = str(op)
        print(json.dumps(report))
        if not ok or report["known_value_matches"] is False:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
