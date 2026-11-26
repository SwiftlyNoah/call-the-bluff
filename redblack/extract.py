#!/usr/bin/env python3
"""Turn one Red/Black endgame class of the SERVED pack into a CONTRACTS.md
game file and strategy file.

    uv run redblack/extract.py d0-2.0.0-1.0.0            # one class
    uv run redblack/extract.py --layer 2                  # every class at T'=2
    uv run redblack/extract.py d0-2.0.0-2.0.0 --guards    # also the guarded profile

What it writes, per class id C, under redblack/out/:

    games/C.game.json            the REAL heads-up Red/Black subgame below C:
                                 a root chance node over the class's entry
                                 type pairs (exact rational bounded-
                                 hypergeometric prior, the solver's own
                                 `rounds.entry_weights2`, recomputed in
                                 Fractions), then every move to elimination,
                                 nothing forgotten (perfect recall), flip
                                 ORDER public (see `--merge-flip-order`).
    strategies/C.served.strategy.json
                                 BOTH seats playing the served pack
                                 (`solver-pack-v2-profile-solve9b`): the
                                 pre-stage from meta.pre, in-round rows from
                                 the quantized bytes, renormalized exactly.
                                 Later rounds play the successor class's pack,
                                 exactly as the deployed bot does (belief
                                 reset at the boundary).
    strategies/C.guarded.strategy.json   (--guards) the same with the
                                 deployed bot's play-time dominance guards
                                 (`solverBot.ts` guardBidding / guardFlip).
    reports/C.json               sizes and this script's own exact
                                 (Fraction) best response, as a cross-check
                                 for the verified checker. NOT a verified
                                 number: `bin/bluff check` is.

The Red/Black game core (`core.py`, `rounds.py`, `pack.py`) is loaded
read-only from ~/coding/red_black by `rbsrc.py`. Packs are read from
redblack/out/pack/ (fetched from gs://red-black-dev-solver, read only).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rbsrc import class_id, core, pack, parse_class_id, rounds

RED, BLACK = core.RED, core.BLACK
OUT = HERE / "out"
PACK_DIR = OUT / "pack"
PACK_PREFIX = "solver-pack-v2-profile-solve9b"


# ---------------------------------------------------------------------------
# Exact entry prior (rounds.entry_weights2, in Fractions)
# ---------------------------------------------------------------------------


def entry_weights_exact(ec) -> dict:
    s0, s1 = ec.s0, ec.s1
    raw = {}
    for t0 in rounds.seat_types(ec, 0):
        r0, x0 = t0
        for t1 in rounds.seat_types(ec, 1):
            r1, x1 = t1
            ur = core.DECK_RED - (r0 + x0 + r1 + x1)
            used_b = (s0.n - r0) + (s0.d - x0) + (s1.n - r1) + (s1.d - x1)
            if not (0 <= ur <= rounds.UNDEALT) or core.DECK_BLACK - used_b != rounds.UNDEALT - ur:
                continue
            w = comb(s0.n, r0) * comb(s0.d, x0) * comb(s1.n, r1) * comb(s1.d, x1) * comb(rounds.UNDEALT, ur)
            if w:
                raw[(t0, t1)] = w
    tot = sum(raw.values())
    out = {k: Fraction(v, tot) for k, v in raw.items()}
    # Cross-check against the solver's float prior.
    flt = rounds.entry_weights2(ec)
    assert set(flt) == set(out), "support differs from rounds.entry_weights2"
    for k, v in out.items():
        assert abs(float(v) - flt[k]) < 1e-12, (k, v, flt[k])
    return out


def start_state(ec, t0, t1):
    """Same construction as simulator/src/redblack/solver/exact.py::_start_state."""
    hands = tuple(
        tuple([RED] * r + [BLACK] * (ec.stat(i).n - r)) for i, (r, _x) in enumerate((t0, t1))
    )
    common = {
        "rnd": 1,
        "starter": ec.starter,
        "hands": (hands[0], hands[1]),
        "ready": (False, False),
        "disc": ((t0[1], ec.s0.d - t0[1]), (t1[1], ec.s1.d - t1[1])),
        "bounds": ((ec.s0.r_lb, ec.s0.b_lb), (ec.s1.r_lb, ec.s1.b_lb)),
        "winner": -1,
    }
    if ec.loser < 0:
        return core.S(phase=core.REORDER, bid_color=-1, bids=(), bidder=-1, flips=(0, 0), loser=-1, **common)
    return core.S(
        phase=core.DISCARD, bid_color=RED, bids=(1,), bidder=1 - ec.loser, flips=(0, 0), loser=ec.loser, **common
    )


# ---------------------------------------------------------------------------
# The served policy
# ---------------------------------------------------------------------------

_PACKS: dict[str, object] = {}


def load_pack(cid: str):
    p = _PACKS.get(cid)
    if p is None:
        path = PACK_DIR / f"{cid}.pack.gz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing: fetch gs://red-black-dev-solver/{PACK_PREFIX}/{cid}.pack.gz"
            )
        p = _PACKS[cid] = pack.read(path)
        assert p.meta["id"] == cid
    return p


def hstr(hand) -> str:
    return core.hand_str(hand)


def node_key(s) -> str:
    """export_explorer.node_id for the in-round node the state sits at."""
    color = "-" if s.bid_color < 0 else "RB"[s.bid_color]
    counts = ",".join(map(str, s.bids))
    if s.phase == core.BIDDING:
        return f"bid|{color}|{counts}"
    fb, fc = s.flips[s.bidder], s.flips[1 - s.bidder]
    return f"proc|{color}|{counts}|{fb}|{fc}"


def exact_row(raw) -> list[Fraction]:
    tot = sum(Fraction(v) for v in raw)
    if tot == 0:
        raise ValueError("empty row")
    return [Fraction(v) / tot for v in raw]


def frac_pre(fs) -> list[Fraction]:
    """meta.pre `f` values are the corpus's 4-dp decimals; read them as the
    decimals they print as (Fraction(str)), not as binary doubles."""
    return [Fraction(repr(float(v))) for v in fs]


class Ctx:
    """Per-seat served-bot context for the current round: the class the bot
    derived at the boundary, its type index there, and the pre-stage choice
    (hence the private-state row) it committed to."""

    __slots__ = ("cid", "drop", "state", "t")

    def __init__(self, cid, t, drop=None, state=None):
        self.cid, self.t, self.drop, self.state = cid, t, drop, state

    def copy(self, **kw):
        c = Ctx(self.cid, self.t, self.drop, self.state)
        for k, v in kw.items():
            setattr(c, k, v)
        return c


def boundary_ctx(s):
    """At a boundary state (root or DISCARD), derive the class and both
    seats' type indices the way the bot does (rounds.ec2_of_state)."""
    ec, types = rounds.ec2_of_state(s)
    cid = class_id(ec)
    p = load_pack(cid)
    ctxs = []
    for seat in range(2):
        tlist = [tuple(t) for t in p.meta["types"][seat]]
        ctxs.append(Ctx(cid, tlist.index(tuple(types[seat]))))
    return ctxs


def guard_bidding(s, seat, labels, probs):
    """solverBot.ts guardBidding, on the core state. Ceiling per color: own
    matching cards + opponent cards not certainly the other color (cross-
    round bounds; no flips are visible during bidding)."""
    own = s.hands[seat]
    opp = 1 - seat
    o_r, o_b = s.bounds[opp]
    own_r = sum(1 for c in own if c == RED)
    ceil = {"R": own_r + len(s.hands[opp]) - o_b, "B": (len(own) - own_r) + len(s.hands[opp]) - o_r}
    if s.bids and "challenge" in labels and s.bids[-1] > ceil["RB"[s.bid_color]]:
        return [Fraction(1) if lbl == "challenge" else Fraction(0) for lbl in labels]

    def impossible(lbl):
        if lbl.startswith("bid_"):
            _, c, n = lbl.split("_")
            return int(n) > ceil[c]
        if lbl.startswith("raise_"):
            return int(lbl[6:]) > ceil["RB"[s.bid_color]]
        return False

    pruned = [Fraction(0) if impossible(lbl) else p for lbl, p in zip(labels, probs)]
    if sum(pruned) > 0:
        return pruned
    feas = [Fraction(0) if impossible(lbl) else Fraction(1) for lbl in labels]
    return feas if sum(feas) > 0 else probs


def guard_flip(s, seat, labels, probs):
    """solverBot.ts guardFlip: if the bidder's own next card is known
    off-color and the challenger's unflipped cards could still cover the
    need, never flip own deck."""
    if "flip_bid" not in labels or "flip_ch" not in labels:
        return probs
    fb, fc = s.flips[seat], s.flips[1 - seat]
    own = s.hands[seat]
    if own[fb] == s.bid_color:
        return probs
    opp_unflipped = len(s.hands[1 - seat]) - fc
    need = s.bids[-1] - fb - fc
    if need <= opp_unflipped:  # knownOff is always 0: unflipped cards are never visible
        out = [Fraction(0) if lbl == "flip_bid" else p for lbl, p in zip(labels, probs)]
        if sum(out) > 0:
            return out
        return [Fraction(0) if lbl == "flip_bid" else Fraction(1) for lbl in labels]
    return probs


class Stats:
    def __init__(self):
        self.missing_rows = 0
        self.missing_nodes = 0
        self.offpath_drop = 0
        self.guard_changed = 0


def served_probs(s, seat, acts, ctx: Ctx, guards: bool, st: Stats):
    """Distribution over core action codes `acts` the served bot plays at s.
    The context is only read here; the caller advances it."""
    p = load_pack(ctx.cid)
    pre = p.meta["pre"][seat]
    row = pre["rows"][ctx.t]
    assert row["t"] == ctx.t
    if s.phase == core.DISCARD:
        assert pre["d"], "discard by a seat the pack does not mark as discarder"
        f = frac_pre(row["f"])
        out = []
        for a in acts:
            c = a[5:]  # drop_R / drop_B
            out.append(sum((fi for fi, pi in zip(f, row["p"]) if pi == c), Fraction(0)))
        if sum(out) == 0:
            raise AssertionError(f"{ctx.cid}: no drop mass for type {ctx.t}")
        tot = sum(out)
        return [x / tot for x in out]
    if s.phase == core.REORDER:
        f = frac_pre(row["f"])
        if pre["d"]:
            cand = [k for k, pi in enumerate(row["p"]) if pi == ctx.drop]
            assert cand, f"{ctx.cid}: drop {ctx.drop} not among type {ctx.t}'s choices"
        else:
            cand = list(range(len(row["o"])))
        mass = {k: f[k] for k in cand}
        tot = sum(mass.values())
        if tot == 0:  # a drop the served bot never takes; value-irrelevant
            st.offpath_drop += 1
            mass = {k: Fraction(1) for k in cand}
            tot = Fraction(len(cand))
        out = []
        for a in acts:
            seq = a[4:]
            out.append(sum((m for k, m in mass.items() if row["o"][k] == seq), Fraction(0)) / tot)
        assert sum(out) == 1, (ctx.cid, acts, row)
        return out
    # In-round: the row for the committed private state.
    key = node_key(s)
    if key not in p:
        st.missing_nodes += 1
        if len(acts) == 1:
            return [Fraction(1)]
        raise AssertionError(f"{ctx.cid}: node {key} missing with {len(acts)} actions")
    e = p.entry(key)
    assert e.actor == seat, (ctx.cid, key, e.actor, seat)
    labels = p.labels(key)
    mask_ok = (e.state_mask >> ctx.state) & 1
    if not mask_ok:
        st.missing_rows += 1
        raise AssertionError(f"{ctx.cid}: node {key} has no row for state {ctx.state}")
    rank = (e.state_mask & ((1 << ctx.state) - 1)).bit_count()
    start = e.freq_offset + rank * e.action_count
    raw = list(p._freqs[start : start + e.action_count])
    probs = exact_row(raw)
    if guards:
        g = guard_bidding(s, seat, labels, probs) if s.phase == core.BIDDING else guard_flip(s, seat, labels, probs)
        tot = sum(g)
        g = [x / tot for x in g]
        if g != probs:
            st.guard_changed += 1
        probs = g
    if s.phase == core.PROCUREMENT:
        lab_of = {"flip_self": "flip_bid", "flip_opp": "flip_ch"}
        want = [lab_of[a] for a in acts]
    else:
        want = list(acts)
    if sorted(want) != sorted(labels):
        raise AssertionError(f"{ctx.cid} {key}: core actions {want} vs pack labels {labels}")
    by = dict(zip(labels, probs))
    return [by[w] for w in want]


def commit_state(ctx: Ctx, seat: int, hand) -> Ctx:
    """After the reorder: the pre-stage choice -> private-state row index.
    A non-representative ordering is mapped to its order class (members of a
    class are payoff-equivalent, `rounds.order_class_of`)."""
    p = load_pack(ctx.cid)
    pre = p.meta["pre"][seat]
    row = pre["rows"][ctx.t]
    rep = hstr(rounds.order_class_of(hand))
    for k, o in enumerate(row["o"]):
        if o == rep and (not pre["d"] or row["p"][k] == ctx.drop):
            return ctx.copy(state=row["s"][k])
    raise AssertionError(f"{ctx.cid}: no pre-stage choice for {rep} drop={ctx.drop}")


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------


def obs_token(prev, new, seq, merge: bool) -> str:
    """Public procurement-resolution observation, appended to BOTH seats'
    histories. With merge=False the flip ORDER (the target sequence) is
    included, as the real game shows it; merge=True reproduces exact.py's
    `_obs_token`, which drops it."""
    pref = "/".join(hstr(prev.hands[i][: new.flips[i]]) for i in range(2))
    order = "" if merge else "~" + "".join(seq)
    color = "RB"[prev.bid_color]
    return f"<{color}{','.join(map(str, prev.bids))}:{pref}{order}:L{new.loser}>"


class Builder:
    def __init__(self, ec, guards: bool, merge: bool, max_nodes: int):
        self.ec = ec
        self.guards = guards
        self.merge = merge
        self.max_nodes = max_nodes
        self.nodes: list[dict] = []
        self.strat: dict[str, dict[str, Fraction]] = {}
        self.strat_g: dict[str, dict[str, Fraction]] = {}
        self.stats = Stats()
        self.stats_g = Stats()
        self.classes_used: set[str] = set()
        self.boundaries: list[int] = []
        self.boundary_info: dict[int, tuple] = {}

    def infoset_name(self, seat, hist, key) -> str:
        return f"p{seat}|{key}|{'^'.join(hist)}"

    def new_id(self) -> int:
        self.nodes.append(None)
        if len(self.nodes) > self.max_nodes:
            raise MemoryError(f"tree exceeds {self.max_nodes} nodes")
        return len(self.nodes) - 1

    def record(self, table, name, acts, probs):
        row = dict(zip(acts, probs))
        old = table.get(name)
        if old is None:
            table[name] = row
        elif old != row:
            raise AssertionError(f"served strategy not constant on infoset {name}: {old} vs {row}")

    def build(self, s, h0, h1, ctx, ctx_g, seq):
        nid = self.new_id()
        if s.phase == core.FINISHED:
            self.nodes[nid] = {"id": nid, "kind": "terminal", "payoff_p0": "1" if s.winner == 0 else "-1"}
            return nid
        acts = core.legal_actions(s)
        seat = core.active_seat(s)
        key = core.build_key(s, seat)
        hist = h0 if seat == 0 else h1
        name = self.infoset_name(seat, hist, key)
        probs = served_probs(s, seat, acts, ctx[seat], False, self.stats)
        self.record(self.strat, name, acts, probs)
        if self.guards:
            probs_g = served_probs(s, seat, acts, ctx_g[seat], True, self.stats_g)
            self.record(self.strat_g, name, acts, probs_g)
        children = []
        for i, code in enumerate(acts):
            s2 = core.step(s, code)
            tok = f"{key}>{code}"
            nh0 = h0 + (tok,) if seat == 0 else h0
            nh1 = h1 + (tok,) if seat == 1 else h1
            nseq = seq
            if s.phase == core.PROCUREMENT:
                nseq = seq + (("S" if code == "flip_self" else "O"),)
            c2 = self._advance_ctx(s, s2, seat, code, ctx)
            c2g = self._advance_ctx(s, s2, seat, code, ctx_g) if self.guards else None
            if s.phase == core.PROCUREMENT and s2.phase in (core.DISCARD, core.FINISHED):
                ot = obs_token(s, s2, nseq, self.merge)
                nh0, nh1 = nh0 + (ot,), nh1 + (ot,)
                nseq = ()
                if s2.phase == core.DISCARD:
                    # Cross-check the bot-style derivation (ec2_of_state on the
                    # state) against the solver's validated round transition.
                    nxt = rounds.apply_round_outcome2(
                        parse_class_id(ctx[0].cid), s.hands, (s.disc[0][0], s.disc[1][0]),
                        s.bidder, s2.flips, s2.loser)
                    ec_b, ty_b = rounds.ec2_of_state(s2)
                    assert (nxt.ec, tuple(nxt.types)) == (ec_b, tuple(ty_b)), (nxt, ec_b, ty_b)
                    c2 = boundary_ctx(s2)
                    c2g = boundary_ctx(s2) if self.guards else None
                    self.classes_used.add(c2[0].cid)
            children.append({"label": code, "child": self.build(s2, nh0, nh1, c2, c2g, nseq)})
        self.nodes[nid] = {"id": nid, "kind": "decision", "player": seat, "infoset": name, "actions": children}
        if s.phase == core.DISCARD and s.rnd > 1:
            # An inner V2 boundary: the bot re-derives its class here.
            self.boundaries.append(nid)
            self.boundary_info[nid] = (ctx[0].cid, ctx[0].t, ctx[1].t)
        return nid

    def _advance_ctx(self, s, s2, seat, code, ctx):
        if ctx is None:
            return None
        if s.phase == core.DISCARD:
            new = list(ctx)
            new[seat] = ctx[seat].copy(drop=code[5:])
            return new
        if s.phase == core.REORDER:
            new = list(ctx)
            new[seat] = commit_state(ctx[seat], seat, s2.hands[seat])
            return new
        return ctx

    def run(self):
        w = entry_weights_exact(self.ec)
        pairs = sorted(w)
        root = self.new_id()
        outcomes = []
        for t0, t1 in pairs:
            s = start_state(self.ec, t0, t1)
            ctx = boundary_ctx(s)
            assert ctx[0].cid == class_id(self.ec), (ctx[0].cid, class_id(self.ec))
            self.classes_used.add(ctx[0].cid)
            ctxg = boundary_ctx(s) if self.guards else None
            child = self.build(s, (), (), ctx, ctxg, ())
            outcomes.append({"label": f"t0={t0[0]}.{t0[1]};t1={t1[0]}.{t1[1]}", "prob": str(w[(t0, t1)]), "child": child})
        self.nodes[root] = {"id": root, "kind": "chance", "outcomes": outcomes}
        return w


# ---------------------------------------------------------------------------
# Independent exact best response (Fractions), a cross-check only
# ---------------------------------------------------------------------------


def exact_eval(nodes, root, strat):
    """Returns (value_p0, br_p0, br_p1) in Fractions. BR by the sequence-tree
    recursion: bucket terminal mass by (last own infoset, action), then
    V(seq) = bucket(seq) + sum over next own infosets of max_a V((I, a))."""
    byid = {n["id"]: n for n in nodes}

    def value(nid):
        stack = [(nid, Fraction(1))]
        tot = Fraction(0)
        while stack:
            i, pr = stack.pop()
            n = byid[i]
            if n["kind"] == "terminal":
                tot += pr * Fraction(n["payoff_p0"])
            elif n["kind"] == "chance":
                for o in n["outcomes"]:
                    stack.append((o["child"], pr * Fraction(o["prob"])))
            else:
                row = strat[n["infoset"]]
                for a in n["actions"]:
                    q = row[a["label"]]
                    if q:
                        stack.append((a["child"], pr * q))
        return tot

    def br(p):
        bucket: dict = {}
        kids: dict = {}
        sign = 1 if p == 0 else -1
        stack = [(root, Fraction(1), None)]
        while stack:
            i, pr, seq = stack.pop()
            n = byid[i]
            if n["kind"] == "terminal":
                bucket[seq] = bucket.get(seq, 0) + pr * sign * Fraction(n["payoff_p0"])
            elif n["kind"] == "chance":
                for o in n["outcomes"]:
                    stack.append((o["child"], pr * Fraction(o["prob"]), seq))
            elif n["player"] == p:
                I = n["infoset"]
                kids.setdefault(seq, set()).add(I)
                for a in n["actions"]:
                    stack.append((a["child"], pr, (I, a["label"])))
            else:
                row = strat[n["infoset"]]
                for a in n["actions"]:
                    q = row[a["label"]]
                    if q:
                        stack.append((a["child"], pr * q, seq))
        acts_of: dict = {}
        for n in nodes:
            if n["kind"] == "decision" and n["player"] == p:
                acts_of[n["infoset"]] = [a["label"] for a in n["actions"]]
        memo: dict = {}
        choice: dict = {}

        def V(seq):
            if seq in memo:
                return memo[seq]
            v = bucket.get(seq, Fraction(0))
            for I in kids.get(seq, ()):
                best = None
                for a in acts_of[I]:
                    va = V((I, a))
                    if best is None or va > best:
                        best, choice[I] = va, a
                v += best
            memo[seq] = v
            return v

        sys.setrecursionlimit(100000)
        return V(None), choice

    v = value(root)
    b0, c0 = br(0)
    b1, c1 = br(1)
    return v, b0, b1, c0, c1


def node_values(nodes, strat) -> list:
    """Exact seat-0 value of every node under the profile (Fractions).
    Children always have larger ids than parents here (preorder ids), so one
    reverse sweep suffices."""
    val = [None] * len(nodes)
    for n in reversed(nodes):
        i = n["id"]
        if n["kind"] == "terminal":
            val[i] = Fraction(n["payoff_p0"])
        elif n["kind"] == "chance":
            val[i] = sum((Fraction(o["prob"]) * val[o["child"]] for o in n["outcomes"]), Fraction(0))
        else:
            row = strat[n["infoset"]]
            val[i] = sum((row[a["label"]] * val[a["child"]] for a in n["actions"] if row[a["label"]]), Fraction(0))
    return val


CORPUS_DIR = Path.home() / "coding" / "red_black" / "tools" / "explorer" / "public" / "data" / "class"


def corpus_vtypes(cid: str):
    """The successor class's stored type-pair values (`vTypes`, 4 dp) from
    the explorer corpus of profile-solve9b on disk (read only): the QRE
    prices the parent's policy pass was solved against."""
    import gzip

    with gzip.open(CORPUS_DIR / f"{cid}.json.gz", "rb") as fh:
        d = json.loads(fh.read())
    return d["vTypes"]


_VT: dict = {}


def corpus_vtypes_cached(cid):
    if cid not in _VT:
        _VT[cid] = corpus_vtypes(cid)
    return _VT[cid]


def gprime_nodes(nodes, boundaries, strat, info=None):
    """The solver's own model G' for this class: round 1 exactly as in G, and
    every inner boundary replaced by a terminal worth the successor subtree's
    exact value under the served profile, which is the successor class's
    type-pair value v(t0, t1) (a function of the two types alone, because
    each seat's successor strategy depends only on its own private state).
    The opponent can then no longer exploit the belief reset."""
    val = node_values(nodes, strat)
    cut = set(boundaries)
    out: list[dict] = []

    def copy(i):
        n = nodes[i]
        nid = len(out)
        out.append(None)
        if i in cut:
            if info is None:
                v = val[i]
            else:
                cid, t0, t1 = info[i]
                v = Fraction(repr(corpus_vtypes_cached(cid)[t0][t1]))
            out[nid] = {"id": nid, "kind": "terminal", "payoff_p0": fstr(v)}
        elif n["kind"] == "terminal":
            out[nid] = {"id": nid, "kind": "terminal", "payoff_p0": n["payoff_p0"]}
        elif n["kind"] == "chance":
            out[nid] = {"id": nid, "kind": "chance",
                        "outcomes": [dict(o, child=copy(o["child"])) for o in n["outcomes"]]}
        else:
            out[nid] = {"id": nid, "kind": "decision", "player": n["player"], "infoset": n["infoset"],
                        "actions": [dict(a, child=copy(a["child"])) for a in n["actions"]]}
        return nid

    sys.setrecursionlimit(100000)
    copy(0)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def fstr(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def values_meta(cid):
    import gzip
    import struct

    path = OUT / "values" / f"{cid}.values.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rb") as fh:
        blob = fh.read()
    _magic, _ver, mlen = struct.unpack_from("<III", blob, 0)
    m = json.loads(blob[32 : 32 + mlen])
    return {"eps": m.get("eps"), "epsRound": m.get("epsRound")}


def extract(cid: str, guards: bool, merge: bool, max_nodes: int, do_br: bool, gprime: str | None = None,
            write: bool = True) -> dict:
    t0 = time.time()
    ec = parse_class_id(cid)
    assert class_id(ec) == cid
    if rounds.is_terminal(ec):
        raise ValueError(f"{cid} is terminal")
    b = Builder(ec, guards, merge, max_nodes)
    b.run()
    tag = "-merged" if merge else ""
    if gprime:
        if not b.boundaries:
            raise ValueError(f"{cid} has no inner boundary; G' = G")
        b.nodes = gprime_nodes(b.nodes, b.boundaries, b.strat,
                               b.boundary_info if gprime == "stored" else None)
        live = {n["infoset"] for n in b.nodes if n["kind"] == "decision"}
        b.strat = {k: v for k, v in b.strat.items() if k in live}
        b.strat_g = {k: v for k, v in b.strat_g.items() if k in live}
        tag += f"-gprime-{gprime}"
    name = f"rb-{cid}{tag}"
    nprior = len(b.nodes[0]["outcomes"])
    game = {
        "format": "bluff-game-v1",
        "name": name,
        "description": (
            f"Heads-up Red/Black endgame below entry class {cid} (post-discard total "
            f"T'={rounds.post_total(ec)}), played to elimination with perfect recall"
            f"{'; flip order merged as in exact.py' if merge else '; flip order public'}. "
            f"Root chance: the solver's bounded-hypergeometric class prior over {nprior} "
            "entry type pairs (rounds.entry_weights2), exact. Payoff +1/-1 to the survivor."
        ),
        "root": 0,
        "nodes": b.nodes,
    }
    if gprime:
        game["description"] = (
            f"The solver's model G' of Red/Black endgame class {cid}: round 1 played exactly, every inner "
            + ("boundary replaced by a terminal worth the served successor class's exact type-pair value "
               "under the served profile" if gprime == "served" else
               "boundary replaced by a terminal worth the successor class's STORED type-pair value "
               "(vTypes of the profile-solve9b explorer corpus, 4 dp)")
            + " (belief reset). Diagnostic only: payoffs come from the solver, not the rules."
        )
    for d in ("games", "strategies", "reports"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    gpath = OUT / "games" / f"{cid}{tag}.game.json"
    if write:
        gpath.write_text(json.dumps(game, separators=(",", ":")))
    out = {"class": cid, "layer": rounds.post_total(ec), "nodes": len(b.nodes), "type_pairs": nprior,
           "inner_boundaries": len(b.boundaries),
           "infosets": len(b.strat), "classes_played": sorted(b.classes_used),
           "missing_single_action_nodes": b.stats.missing_nodes, "offpath_drop_rows": b.stats.offpath_drop,
           "game_file": str(gpath.relative_to(HERE.parent)), "solver_meters": values_meta(cid)}
    profiles = [("served", b.strat)] + ([("guarded", b.strat_g)] if guards else [])
    for label, table in profiles:
        sp = OUT / "strategies" / f"{cid}{tag}.{label}.strategy.json"
        strategy = {
            "format": "bluff-strategy-v1",
            "game": name,
            "producer": f"redblack/extract.py from gs://red-black-dev-solver/{PACK_PREFIX}"
                        f"{' + solverBot.ts dominance guards' if label == 'guarded' else ''}",
            "probs": {I: {a: fstr(p) for a, p in row.items()} for I, row in sorted(table.items())},
        }
        if write:
            sp.write_text(json.dumps(strategy, separators=(",", ":")))
        rec = {"strategy_file": str(sp.relative_to(HERE.parent))}
        if label == "guarded":
            rec["rows_changed_by_guards"] = b.stats_g.guard_changed
        if do_br:
            v, b0, b1, _c0, _c1 = exact_eval(b.nodes, 0, table)
            nc = b0 + b1
            rec.update({"value_p0": fstr(v), "value_p0_float": float(v), "br_value_p0": fstr(b0),
                        "br_value_p1": fstr(b1), "nash_conv": fstr(nc), "exploitability": fstr(nc / 2),
                        "exploitability_float": float(nc / 2),
                        "gain_p0_float": float(b0 - v), "gain_p1_float": float(b1 + v)})
        out[label] = rec
    out["seconds"] = round(time.time() - t0, 2)
    if write:
        (OUT / "reports" / f"{cid}{tag}.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("classes", nargs="*")
    ap.add_argument("--layer", type=int, help="every non-root class with this post-discard total")
    ap.add_argument("--guards", action="store_true")
    ap.add_argument("--merge-flip-order", action="store_true",
                    help="drop flip order from the public observation (exact.py's tree)")
    ap.add_argument("--max-nodes", type=int, default=3_000_000)
    ap.add_argument("--no-br", action="store_true")
    ap.add_argument("--gprime", choices=("served", "stored"),
                    help="the solver's model G': cut every inner boundary to a terminal worth the successor's "
                         "type-pair value, either exactly under the served profile or the stored corpus vTypes")
    args = ap.parse_args()
    cids = list(args.classes)
    if args.layer is not None:
        ids = (OUT / "classes_le3.txt").read_text().split()
        cids += [c for c in ids if not c.startswith("dr") and rounds.post_total(parse_class_id(c)) == args.layer]
    rows = []
    for cid in cids:
        try:
            r = extract(cid, args.guards, args.merge_flip_order, args.max_nodes, not args.no_br, args.gprime)
        except MemoryError as e:
            r = {"class": cid, "error": str(e)}
        rows.append(r)
        brief = {k: r.get(k) for k in ("class", "nodes", "infosets", "error")}
        for lab in ("served", "guarded"):
            if lab in r and "exploitability_float" in r[lab]:
                brief[lab] = (round(r[lab]["value_p0_float"], 6), round(r[lab]["exploitability_float"], 6))
        print(json.dumps(brief), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
