"""Independent exact best response and exploitability, in Python fractions.Fraction.

Usage:
    uv run reference/br.py GAME.game.json STRATEGY.strategy.json [--brute | --no-brute]
                           [--brute-limit N]

Prints a bluff-verdict-v1 JSON (CONTRACTS.md section 3) with "verified": false, since this is the
unverified reference, not the Dafny core. Exit 2 on a malformed game or strategy, with the
reasons on stderr. Exit 3 if the internal cross-checks disagree (that would be a bug here).

Written from the definitions, sharing no code with the Dafny core or the solvers:

* value_p0: expected payoff to player 0 when both play the (normalized) profile.
* Best response of player p against the profile's opponent strategy: for every infoset I of p,
  processed bottom-up (longest own sequence first, so every infoset of p below I is already
  decided), the counterfactual value of action a is
      sum over nodes h in I of reach_{-p}(h) * v_p(child(h, a)),
  where reach_{-p}(h) is the product of chance and opponent probabilities on the path to h and
  v_p is p's expected utility from a node on, with p playing the choices already made. The
  choice at I is the first action (in game order) with the largest counterfactual value. The
  best-response value is v_p(root) with all choices made.
* Cross-checks on every run: the returned pure strategy, evaluated from scratch, attains the
  best-response value; nash_conv >= 0. With brute force (automatic when player p has at most
  --brute-limit pure strategies, forced by --brute), every pure strategy of p is enumerated and
  the maximum must equal the best-response value.

Importable: `compute_verdict(game, sigma)` and `brute_force_br(game, sigma, p)`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys

sys.set_int_max_str_digits(0)  # exact verdict fractions can run to many thousands of digits
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fractions_util import StrategyError, format_fraction, normalize_strategy
from validate_game import (
    Game,
    GameError,
    load_game,
    load_json,
    player_sequences,
    schema_errors,
)

Profile = Mapping[str, Mapping[str, Fraction]]
Pure = Mapping[str, str]

DEFAULT_BRUTE_LIMIT = 4096


class CrossCheckError(AssertionError):
    """Two independent computations here disagree: a bug in this file, not in the input."""


def utility(g: Game, nid: int, p: int) -> Fraction:
    payoff = g.nodes[nid].payoff_p0
    assert payoff is not None
    return payoff if p == 0 else -payoff


def expected_value(g: Game, behavior: Mapping[int, Profile | Pure], p: int) -> Fraction:
    """Player p's expected utility when each player q plays behavior[q].

    behavior[q] maps infoset -> {action: prob} (behavioral) or infoset -> action (pure).
    Iterative over the tree, so deep games do not hit the recursion limit.
    """
    total = Fraction(0)
    stack: list[tuple[int, Fraction]] = [(g.root, Fraction(1))]
    while stack:
        nid, w = stack.pop()
        node = g.nodes[nid]
        if node.kind == "terminal":
            total += w * utility(g, nid, p)
        elif node.kind == "chance":
            for _, prob, child in node.edges:
                assert prob is not None
                stack.append((child, w * prob))
        else:
            assert node.player is not None and node.infoset is not None
            plan = behavior[node.player][node.infoset]
            if isinstance(plan, str):
                stack.extend((c, w) for lab, _, c in node.edges if lab == plan)
            else:
                for lab, _, child in node.edges:
                    q = plan[lab]
                    if q:
                        stack.append((child, w * q))
    return total


def opponent_reach(g: Game, sigma: Profile, p: int) -> dict[int, Fraction]:
    """reach_{-p}(h): product of chance and opponent probabilities on the path to each node."""
    reach = {g.root: Fraction(1)}
    stack = [g.root]
    while stack:
        nid = stack.pop()
        node = g.nodes[nid]
        r = reach[nid]
        for lab, prob, child in node.edges:
            if node.kind == "chance":
                assert prob is not None
                reach[child] = r * prob
            elif node.player == p:
                reach[child] = r
            else:
                assert node.infoset is not None
                reach[child] = r * sigma[node.infoset][lab]
            stack.append(child)
    return reach


def best_response(g: Game, sigma: Profile, p: int) -> tuple[Fraction, dict[str, str]]:
    """Exact best-response value of player p against sigma's opponent, and a pure BR attaining it."""
    reach = opponent_reach(g, sigma, p)
    seqs = player_sequences(g)
    members: dict[str, list[int]] = {}
    depth: dict[str, int] = {}
    for nid, node in g.nodes.items():
        if node.kind == "decision" and node.player == p:
            assert node.infoset is not None
            members.setdefault(node.infoset, []).append(nid)
            depth[node.infoset] = len(seqs[nid][p])
    choice: dict[str, str] = {}
    memo: dict[int, Fraction] = {}

    def value(nid: int) -> Fraction:
        # v_p(nid) with p playing `choice`; only ever asked for nodes whose p-infosets are decided.
        if nid in memo:
            return memo[nid]
        node = g.nodes[nid]
        if node.kind == "terminal":
            v = utility(g, nid, p)
        elif node.kind == "chance":
            v = sum((prob * value(c) for _, prob, c in node.edges), Fraction(0))  # type: ignore[operator]
        elif node.player == p:
            assert node.infoset is not None
            a = choice[node.infoset]  # KeyError here would mean the bottom-up order is wrong
            v = next(value(c) for lab, _, c in node.edges if lab == a)
        else:
            assert node.infoset is not None
            row = sigma[node.infoset]
            v = sum((row[lab] * value(c) for lab, _, c in node.edges if row[lab]), Fraction(0))
        memo[nid] = v
        return v

    def _decide(infoset: str) -> None:
        # Counterfactual value of each action, summed over the infoset's nodes weighted by
        # opponent and chance reach; ties go to the first action in game order.
        actions = g.infosets[infoset][1]
        cf = {a: Fraction(0) for a in actions}
        for nid in members[infoset]:
            if reach[nid] == 0:
                continue
            for lab, _, child in g.nodes[nid].edges:
                cf[lab] += reach[nid] * value(child)
        best = actions[0]
        for a in actions[1:]:
            if cf[a] > cf[best]:
                best = a
        choice[infoset] = best

    order = sorted(members, key=lambda i: (-depth[i], i))
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10 * len(g.nodes) + 1000))
    try:
        for infoset in order:
            _decide(infoset)
        br_value = value(g.root)
    finally:
        sys.setrecursionlimit(old_limit)
    return br_value, choice


def count_pure_strategies(g: Game, p: int) -> int:
    n = 1
    for q, acts in g.infosets.values():
        if q == p:
            n *= len(acts)
    return n


def brute_force_br(g: Game, sigma: Profile, p: int) -> tuple[Fraction, dict[str, str]]:
    """Maximum over every pure strategy of player p (first maximizer in enumeration order)."""
    infosets = sorted(i for i, (q, _) in g.infosets.items() if q == p)
    best_v: Fraction | None = None
    best_s: dict[str, str] = {}
    for combo in itertools.product(*(g.infosets[i][1] for i in infosets)):
        pure = dict(zip(infosets, combo))
        behavior: dict[int, Profile | Pure] = {p: pure, 1 - p: sigma}
        v = expected_value(g, behavior, p)
        if best_v is None or v > best_v:
            best_v, best_s = v, pure
    assert best_v is not None
    return best_v, best_s


def compute_verdict(
    g: Game, sigma: Profile, brute: bool | None = None, brute_limit: int = DEFAULT_BRUTE_LIMIT,
    log=None,
) -> dict:
    """The verdict dict. brute: True forces brute force, False skips it, None means automatic."""
    value_p0 = expected_value(g, {0: sigma, 1: sigma}, 0)
    br_vals: list[Fraction] = []
    br_strats: list[dict[str, str]] = []
    for p in (0, 1):
        v, s = best_response(g, sigma, p)
        attained = expected_value(g, {p: s, 1 - p: sigma}, p)
        if attained != v:
            raise CrossCheckError(f"p{p}: returned pure BR attains {attained}, BR value says {v}")
        n_pure = count_pure_strategies(g, p)
        run_brute = brute if brute is not None else n_pure <= brute_limit
        if run_brute:
            if n_pure > brute_limit:
                raise CrossCheckError(
                    f"p{p} has {n_pure} pure strategies, over --brute-limit {brute_limit}"
                )
            bv, _ = brute_force_br(g, sigma, p)
            if bv != v:
                raise CrossCheckError(f"p{p}: brute force max {bv} != best response {v}")
            if log:
                log(f"p{p}: brute force over {n_pure} pure strategies agrees: {format_fraction(v)}")
        elif log:
            size = str(n_pure) if n_pure < 10**12 else f"about 10^{len(str(n_pure)) - 1}"
            log(f"p{p}: brute force skipped ({size} pure strategies)")
        own_value = value_p0 if p == 0 else -value_p0
        if v < own_value:
            raise CrossCheckError(f"p{p}: BR value {v} below profile value {own_value}")
        br_vals.append(v)
        br_strats.append(s)
    nash_conv = br_vals[0] + br_vals[1]
    if nash_conv < 0:
        raise CrossCheckError(f"nash_conv {nash_conv} < 0")
    expl = nash_conv / 2
    return {
        "format": "bluff-verdict-v1",
        "game": g.name,
        "value_p0": format_fraction(value_p0),
        "br_value_p0": format_fraction(br_vals[0]),
        "br_value_p1": format_fraction(br_vals[1]),
        "nash_conv": format_fraction(nash_conv),
        "exploitability": format_fraction(expl),
        "exploitability_float": float(expl),
        "best_response_p0": br_strats[0],
        "best_response_p1": br_strats[1],
        "verified": False,
    }


def load_strategy(g: Game, path: str | Path) -> dict[str, dict[str, Fraction]]:
    """Load, schema-check and normalize a strategy file against g; raise StrategyError."""
    try:
        doc = load_json(path)
    except (OSError, ValueError) as e:
        raise StrategyError(f"cannot read {path}: {e}") from e
    errs = schema_errors(doc, "strategy.schema.json")
    if errs:
        raise StrategyError("; ".join(errs))
    if doc["game"] != g.name:
        raise StrategyError(f"strategy is for game {doc['game']!r}, game file is {g.name!r}")
    return normalize_strategy(g.infoset_actions(), doc["probs"])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Reference exact best response (unverified).")
    ap.add_argument("game")
    ap.add_argument("strategy")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--brute", dest="brute", action="store_true", default=None,
                      help="force brute-force enumeration (fails over --brute-limit)")
    mode.add_argument("--no-brute", dest="brute", action="store_false")
    ap.add_argument("--brute-limit", type=int, default=DEFAULT_BRUTE_LIMIT)
    ap.add_argument("--quiet", action="store_true", help="no cross-check log on stderr")
    args = ap.parse_args(argv)
    try:
        g = load_game(args.game)
        sigma = load_strategy(g, args.strategy)
    except GameError as e:
        print(f"malformed game {args.game}:", file=sys.stderr)
        for r in e.reasons:
            print(f"  {r}", file=sys.stderr)
        return 2
    except StrategyError as e:
        print(f"malformed strategy {args.strategy}: {e}", file=sys.stderr)
        return 2
    log = None if args.quiet else (lambda m: print(m, file=sys.stderr))
    try:
        verdict = compute_verdict(g, sigma, brute=args.brute, brute_limit=args.brute_limit, log=log)
    except CrossCheckError as e:
        print(f"internal cross-check failed: {e}", file=sys.stderr)
        return 3
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
