"""Differential test: a written Red/Black game file against the real engine.

    uv run games/redblack/difftest.py                    # every default variant, exhaustive
    uv run games/redblack/difftest.py rb-pennies --reps all
    uv run games/redblack/difftest.py --deal-samples 200000   # also test the deal distribution

The engine is the Python mirror of the production TypeScript engine, in
~/coding/red_black/simulator/src/redblack/engine. It is imported read-only
(no bytecode is written into that repo) and never modified.

The walk reads the GAME FILE (games/out/redblack/<name>.game.json), not
model.py, so what is tested is the artifact a checker consumes. Every node of
the file is visited, in lockstep with an engine state reached by real engine
actions (`apply_action`), and at every step it asserts:

1. Legal actions: the engine's legal actions for the acting seat
   (`enumerate_legal_actions`), mapped to labels, are exactly the node's
   action labels. A label may stand for several engine actions (reorders by
   color sequence, discards by color); a step where the engine's actions
   collapse to one label is a forced move the file skips, and the walk applies
   it to the engine without consuming a file node.
2. The acting player matches the engine's active seat (the bidder in
   procurement, the loser in the discard).
3. Outcomes: at a terminal node the engine is `finished` and its winner
   gives the node's payoff (+1 seat 0 wins, -1 seat 1 wins).
4. Information: for each player, the map infoset -> engine observation
   history is a bijection over every decision node of that player. The
   observation history is the sequence of that player's projected views
   (`project_for_recipient`, the exact payload a client receives), with
   consecutive duplicates removed and two canonicalizations: the player's own
   card ids are replaced by "own" (a seat knows its own cards; which of two
   same-colored cards is which is not information), and the own hand is shown
   sorted while the player has not yet confirmed its reorder (the order
   before a reorder is the deal's or the last round's, and is overwritten).
   `version` and `reorderReady` are dropped (they say when someone clicked,
   not what anyone holds). Two nodes in one infoset with different engine
   histories = the file merges what a player can tell apart. Two infosets
   with the same engine history = the file lets a player see something the
   engine never shows: a LEAK (lessons.md #1).

`--reps first|random|all` picks which engine action stands for a label that
covers several: `all` explores every one (exponential in duplicate colors, so
for the small files), which tests the claim that they are interchangeable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True  # never write into the read-only engine repo
ENGINE_SRC = Path.home() / "coding" / "red_black" / "simulator" / "src"
sys.path.insert(0, str(ENGINE_SRC))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from redblack.engine import (  # noqa: E402
    Bid,
    Card,
    Challenge,
    ConfirmReorder,
    Discard,
    FlipAction,
    GameState,
    PlaceBid,
    PlayerSeat,
    RoundOutcome,
    apply_action,
    deal_initial_hands,
    enumerate_legal_actions,
    project_for_recipient,
)

import generate as G  # noqa: E402  (only for the variant table: start positions)
import model as M  # noqa: E402  (only deal_distribution, for the deal test)

OUT = HERE.parents[0] / "out" / "redblack"
COLOR = {"R": "red", "B": "black"}
SHORT = {"red": "R", "black": "B"}
NOW = "2026-11-20T00:00:00Z"


# ---------------------------------------------------------------------------
# Engine side


def engine_state(cfg: M.Config, hands: tuple[str, str]) -> GameState:
    """An engine state holding these hands (color strings, top first)."""
    k = 0
    seats = []
    for i, h in enumerate(hands):
        cards = []
        for c in h:
            cards.append(Card(id=f"c{k}", color=COLOR[c]))
            k += 1
        seats.append(PlayerSeat(userId=f"u{i}", displayName=f"seat{i}", elo=1500, seat=i, hand=tuple(cards)))
    st = cfg.start
    if st is not None and st.phase == "discard":
        L = st.loser
        return GameState(
            version=1, phase="discard", seats=tuple(seats), activeSeat=L, bidHistory=(), procurement=None,
            round=1,
            lastRoundOutcome=RoundOutcome(loserSeat=L, winnerSeat=1 - L, bid=Bid(by=L, color="red", count=1), flips=()),
            winnerSeat=None, dealSeed="difftest",
        )
    starter = st.starter if st is not None else 0
    return GameState(
        version=1, phase="reorder", seats=tuple(seats), activeSeat=starter, bidHistory=(), procurement=None,
        round=1, lastRoundOutcome=None, winnerSeat=None, dealSeed="difftest",
    )


def label_of(state: GameState, a, slot_discards: bool = False) -> str:
    if isinstance(a, ConfirmReorder):
        by_id = {c.id: c.color for c in state.seats[a.seat].hand}
        return "".join(SHORT[by_id[i]] for i in a.orderedCardIds)
    if isinstance(a, PlaceBid):
        return f"bid {a.count}{SHORT[a.color]}"
    if isinstance(a, Challenge):
        return "challenge"
    if isinstance(a, FlipAction):
        return f"flip{a.targetSeat}"
    if isinstance(a, Discard) and slot_discards:
        return f"discard slot{[c.id for c in state.seats[a.seat].hand].index(a.cardId)}"
    if isinstance(a, Discard):
        c = next(c.color for c in state.seats[a.seat].hand if c.id == a.cardId)
        return f"discard {SHORT[c]}"
    raise TypeError(a)


def acting_seat(state: GameState) -> int:
    """Whose move it is. In the reorder both seats may move; the file (and
    this walk) orders them seat 0 then seat 1, which no view reveals."""
    if state.phase == "reorder":
        for s in state.seats:
            if not s.eliminated and not s.reorderReady:
                return s.seat
        raise AssertionError("reorder with nobody left to confirm")
    if state.phase == "procurement":
        return state.procurement.bidderSeat
    return state.activeSeat


def grouped_actions(state: GameState, slot_discards: bool = False) -> tuple[int, dict[str, list]]:
    p = acting_seat(state)
    if state.phase != "reorder":
        assert state.activeSeat == p, (state.phase, state.activeSeat, p)
        assert not enumerate_legal_actions(state, 1 - p), "the other seat has legal actions"
    groups: dict[str, list] = defaultdict(list)
    for a in enumerate_legal_actions(state, p):
        groups[label_of(state, a, slot_discards)].append(a)
    return p, dict(groups)


def canon_view(state: GameState, p: int):
    v = project_for_recipient(state, f"u{p}")

    def cid(x: str | None, owner: int):
        # Only the player's OWN card ids are canonicalized, by owner, never by
        # id shape: an opponent id is kept verbatim, so if the engine ever
        # shipped a stable opponent id (the lesson-1 leak) it would show up
        # here as an engine distinction the file lacks (a merge).
        if x is None:
            return None
        return "own" if owner == p else x

    seats = []
    for s in v.seats:
        cards = tuple((cid(c.id, s.seat), c.color) for c in s.hand)
        if s.seat == p and v.phase == "reorder" and not s.reorderReady:
            cards = tuple(sorted(cards, key=lambda t: t[1] or ""))
        seats.append((s.seat, s.cardCount, s.eliminated, cards))

    def flips(fs):
        return tuple((f.fromSeat, f.handIndex, cid(f.cardId, f.fromSeat), f.color) for f in fs)

    proc = None
    if v.procurement is not None:
        pr = v.procurement
        proc = (pr.bidderSeat, pr.challengerSeat, pr.bid, flips(pr.flips), pr.revealedTargetCount,
                tuple(sorted(pr.nextFlipIndexBySeat)), pr.misses)
    out = None
    if v.lastRoundOutcome is not None:
        o = v.lastRoundOutcome
        out = (o.loserSeat, o.winnerSeat, o.bid, flips(o.flips), cid(o.discardedCardId, o.loserSeat), o.discardedColor)
    return (v.phase, v.activeSeat, v.round, v.bidHistory, proc, out, tuple(seats), v.winnerSeat)


def push_views(hist: tuple, state: GameState) -> tuple:
    h0, h1 = hist
    for p in (0, 1):
        cv = canon_view(state, p)
        if p == 0 and (not h0 or h0[-1] != cv):
            h0 = h0 + (cv,)
        if p == 1 and (not h1 or h1[-1] != cv):
            h1 = h1 + (cv,)
    return (h0, h1)


# ---------------------------------------------------------------------------
# The walk


class Report:
    def __init__(self) -> None:
        self.nodes = 0
        self.engine_steps = 0
        self.forced = 0
        self.errors: list[str] = []
        self.key_to_obs: dict[str, int] = {}
        self.obs_to_key: list[dict] = [{}, {}]
        self.obs_ids: dict = {}
        # Examples are capped at 20 for printing; n_split / n_merged count every
        # distinct leaked infoset pair and every distinct merged infoset.
        self.split: list[tuple[str, str]] = []  # same engine history, two infosets: a leak
        self.merged: list[str] = []  # one infoset, two engine histories
        self._split_all: set[tuple[str, str]] = set()
        self._merged_all: set[str] = set()
        self.n_errors = 0
        self.terminals = Counter()

    @property
    def n_split(self) -> int:
        return len(self._split_all)

    @property
    def n_merged(self) -> int:
        return len(self._merged_all)

    def err(self, msg: str) -> None:
        self.n_errors += 1
        if len(self.errors) < 50:
            self.errors.append(msg)

    def record(self, p: int, key: str, obs) -> None:
        oid = self.obs_ids.setdefault(obs, len(self.obs_ids))
        prev = self.key_to_obs.setdefault(key, oid)
        if prev != oid and key not in self._merged_all:
            self._merged_all.add(key)
            if len(self.merged) < 20:
                self.merged.append(key)
        other = self.obs_to_key[p].setdefault(oid, key)
        if other != key and (other, key) not in self._split_all:
            self._split_all.add((other, key))
            if len(self.split) < 20:
                self.split.append((other, key))


def walk(doc: dict, cfg: M.Config, reps: str, seed: int) -> Report:
    nodes = doc["nodes"]
    rep = Report()
    rng = random.Random(seed)

    def choose(actions: list) -> list:
        if reps == "all":
            return actions
        if reps == "random":
            return [rng.choice(actions)]
        return [actions[0]]

    def apply(state, a, hist):
        rep.engine_steps += 1
        nxt, _ = apply_action(state, a, NOW)
        return nxt, push_views(hist, nxt)

    # Iterative DFS: (file node id, engine state, per-player view histories, path for messages)
    stack = []
    root = nodes[doc["root"]]
    if root["kind"] == "chance":
        # The deal: the file's chance probabilities must be the exact
        # hypergeometric (which --deal-samples tests against the engine's shuffle).
        exact = {f"{''.join(M.sorted_hand(r0, n0))}|{''.join(M.sorted_hand(r1, n1))}": q
                 for (n0, n1) in [M.hand_sizes(cfg)] for r0, r1, q in M.deal_distribution(cfg)}
        got = {o["label"]: Fraction(o["prob"]) for o in root["outcomes"]}
        if got != exact:
            rep.err(f"deal: file chance probabilities {got} differ from the exact hypergeometric {exact}")
        for o in root["outcomes"]:
            hands = tuple(o["label"].split("|"))
            st = engine_state(cfg, hands)
            stack.append((o["child"], st, push_views(((), ()), st), o["label"]))
        rep.nodes += 1
    else:
        hands = (cfg.start.hands[0], cfg.start.hands[1])
        hands = tuple("".join(sorted(h, reverse=True)) for h in hands)
        st = engine_state(cfg, hands)
        stack.append((doc["root"], st, push_views(((), ()), st), "start"))

    while stack:
        nid, st, hist, path = stack.pop()
        node = nodes[nid]
        # Apply forced engine moves (a single label) that the file skips.
        while st.phase != "finished":
            p, groups = grouped_actions(st, cfg.leak_discard)
            if len(groups) != 1:
                break
            rep.forced += 1
            ((label, acts),) = groups.items()
            for a in choose(acts)[1:]:
                st2, hist2 = apply(st, a, hist)
                stack.append((nid, st2, hist2, path + f" >{label}*"))
            st, hist = apply(st, choose(acts)[0], hist)
            path += f" >{label}"
        rep.nodes += 1
        if node["kind"] == "terminal":
            if st.phase != "finished":
                rep.err(f"{path}: file terminal, engine in phase {st.phase}")
                continue
            want = "1" if st.winnerSeat == 0 else "-1"
            rep.terminals[node["payoff_p0"]] += 1
            if node["payoff_p0"] != want:
                rep.err(f"{path}: file payoff {node['payoff_p0']}, engine winner seat {st.winnerSeat}")
            continue
        if st.phase == "finished":
            rep.err(f"{path}: engine finished, file has a {node['kind']} node")
            continue
        if node["kind"] != "decision":
            rep.err(f"{path}: unexpected {node['kind']} node mid-game")
            continue
        p, groups = grouped_actions(st, cfg.leak_discard)
        if node["player"] != p:
            rep.err(f"{path}: file player {node['player']}, engine seat {p} to act")
            continue
        labels = [a["label"] for a in node["actions"]]
        if set(labels) != set(groups):
            rep.err(f"{path}: labels file {sorted(labels)} vs engine {sorted(groups)}")
            continue
        rep.record(p, node["infoset"], hist[p])
        for a in node["actions"]:
            for ea in choose(groups[a["label"]]):
                st2, hist2 = apply(st, ea, hist)
                stack.append((a["child"], st2, hist2, path + f" {a['label']}"))
    return rep


# ---------------------------------------------------------------------------
# The deal


def deal_test(samples: int, hand_size: int, seed: int = 0) -> dict:
    """Chi-square of the engine's own deal (deal_initial_hands, the shuffle
    every real game is dealt by) against the file's exact hypergeometric."""
    cfg = M.Config(name="deal", description="", hand_size=hand_size)
    exact = {(r0, r1): p for r0, r1, p in M.deal_distribution(cfg)}
    seen: Counter = Counter()
    for i in range(samples):
        hands = deal_initial_hands(2, f"difftest-{seed}-{i}", cards_per_player=hand_size)
        seen[tuple(sum(c.color == "red" for c in h) for h in hands)] += 1
    chi2 = sum((seen[k] - samples * float(p)) ** 2 / (samples * float(p)) for k, p in exact.items())
    unexpected = set(seen) - set(exact)
    from scipy.stats import chi2 as chi2dist

    dof = len(exact) - 1
    return {
        "hand_size": hand_size,
        "samples": samples,
        "cells": len(exact),
        "chi2": round(chi2, 3),
        "dof": dof,
        "p_value": round(float(chi2dist.sf(chi2, dof)), 4),
        "unexpected_cells": sorted(unexpected),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--reps", choices=["first", "random", "all"], default="random")
    ap.add_argument("--seed", type=int, default=2520)
    ap.add_argument("--deal-samples", type=int, default=0)
    ap.add_argument("--in-memory", action="store_true",
                    help="build the tree with generate.build instead of reading the file (for rb-31, rb-h2: too big to keep)")
    ap.add_argument("--mutants", action="store_true",
                    help="seed each model.MUTANTS bug into the named variants (built in memory) and require a FAIL")
    args = ap.parse_args(argv)

    failed = False
    if args.mutants:
        from dataclasses import replace

        for name in args.names or ["rb-21", "rb-12"]:
            for mutant, what in M.MUTANTS.items():
                cfg = replace(G.VARIANTS[name], mutant=mutant)
                doc = {"root": 0, "nodes": G.build(cfg)}
                rep = walk(doc, cfg, args.reps, args.seed)
                caught = bool(rep.errors or rep.split or rep.merged)
                first = rep.errors[0] if rep.errors else (
                    f"leak: {rep.split[0]}" if rep.split else (f"merge: {rep.merged[0]}" if rep.merged else ""))
                print(f"{name} mutant {mutant} ({what}): {'CAUGHT' if caught else 'MISSED'} "
                      f"errors={rep.n_errors} leaks={rep.n_split} merges={rep.n_merged}")
                if first:
                    print("   first:", first[:300])
                failed |= not caught
        return 1 if failed else 0
    for name in args.names or G.DEFAULT:
        cfg = G.VARIANTS[name]
        t0 = time.time()
        if args.in_memory:
            doc = {"root": 0, "nodes": G.build(cfg)}
        else:
            doc = json.loads((OUT / f"{name}.game.json").read_text())
        rep = walk(doc, cfg, args.reps, args.seed)
        ok = not rep.errors and not rep.split and not rep.merged
        summary = {
            "file_nodes": len(doc["nodes"]),
            "visits": rep.nodes,
            "engine_steps": rep.engine_steps,
            "forced_steps": rep.forced,
            "terminals": dict(rep.terminals),
            "infosets_checked": len(rep.key_to_obs),
            "errors": rep.n_errors,
            "leaks(same engine view, two infosets)": rep.n_split,
            "merges(one infoset, two engine views)": rep.n_merged,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{name} reps={args.reps}: {'PASS' if ok else 'FAIL'} {json.dumps(summary)}")
        for e in rep.errors[:10]:
            print("  error:", e)
        for a, b in rep.split[:3]:
            print(f"  leak: engine shows the same history for\n    {a}\n    {b}")
        for k in rep.merged[:3]:
            print(f"  merge: {k}")
        failed |= not ok
    if args.deal_samples:
        for h in (1, 2, 3):
            print("deal", json.dumps(deal_test(args.deal_samples, h, args.seed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
