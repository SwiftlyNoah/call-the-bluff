"""The trusted adapter between the JSON files and the verified core in this folder.

Usage: bin/bluff check GAME.game.json STRATEGY.strategy.json
   or: uv run dafny/core/adapter.py GAME.game.json STRATEGY.strategy.json

Reads a bluff-game-v1 game and a bluff-strategy-v1 profile, builds the Dafny
values (exact rationals throughout), calls the compiled Checker.Check, and prints
a bluff-verdict-v1 JSON verdict. Exit 2 on a malformed game or strategy, with
the reason on stderr.

This file is in the trusted base: JSON parsing, the infoset and action
numbering, the number grammar, the schema's key sets, the perfect recall check
and the strategy normalization (each row must sum to 1 within 1e-6 and is then
divided by its exact sum) are not verified. Everything after the call to Check
is the compiled Dafny core, whose guarantees are CheckSound in Check.dfy.

Build stamps: build.sh writes SOURCE_SHA256 (the .dfy files) only after
`dafny verify` succeeds, and BUILD_SHA256 (every file of the compiled core it
just produced). This file refuses to run unless both match, and imports the
compiled core from source, ignoring any cached bytecode, so a verdict never
comes from a build of unverified sources or from a compiled core edited after
the build. (A file swapped between the hash check and the import, within one
run, is not covered.)

Exit codes: 0 with a verdict; 2 when the input is malformed, breaks the
contract (including a game without perfect recall) or exceeds a stated limit
(numbers over MAX_NUMBER_CHARS characters, trees deeper than MAX_DEPTH); 1 on
an internal error, never with a verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from fractions import Fraction
from pathlib import Path

# Exact values can have more digits than Python's default 4300-digit int/str
# limit (a 15-level chain of 1e-300 probabilities has a 4501-digit denominator),
# so fmt lifts it, here and for every importer; input numbers are bounded by
# MAX_NUMBER_CHARS instead.
sys.set_int_max_str_digits(0)

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build" / "checker-py"

TOL = Fraction(1, 10**6)
SOURCES = ["Math.dfy", "Game.dfy", "BestResponse.dfy", "Check.dfy"]
STAMPS = {"SOURCE_SHA256", "BUILD_SHA256"}

# Input limits (exit 2 beyond them). A number string or JSON integer literal
# longer than this is refused before Fraction or int sees it. Trees deeper than
# MAX_DEPTH are refused because the compiled core recurses once per tree level
# (Val, WFN, Sane, BRG) and would run out of stack; 300,000 levels are tested
# (tests/regressions.py --slow grades 300,000-deep chains and refuses 300,001).
MAX_NUMBER_CHARS = 10_000
MAX_DEPTH = 300_000

# The keys schema/game.schema.json and schema/strategy.schema.json allow; any
# other key is refused, so a payoff written on an inner node is never dropped.
GAME_KEYS = {"format", "name", "description", "known_value_p0", "known_value_source", "root", "nodes"}
NODE_KEYS = {
    "chance": {"id", "kind", "outcomes"},
    "decision": {"id", "kind", "player", "infoset", "actions"},
    "terminal": {"id", "kind", "payoff_p0"},
}
EDGE_KEYS = {"outcomes": {"label", "prob", "child"}, "actions": {"label", "child"}}
STRAT_KEYS = {"format", "game", "producer", "probs"}

# CONTRACTS.md section 1: game numbers are integers, fractions or plain decimals.
GAME_RAT = re.compile(r"-?(?:[0-9]+(?:\.[0-9]+)?|[0-9]+/0*[1-9][0-9]*)")
# Section 2 and board/issues.md (spec 1): strategy probabilities may also use the
# exponent form Python's repr prints ("1e-05"); the exponent is capped at 3 digits
# so "1e-999999999" cannot make Fraction build a huge integer.
STRAT_RAT = re.compile(r"\+?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]{1,3})?|[0-9]+/0*[1-9][0-9]*)")


def source_hash() -> str:
    h = hashlib.sha256()
    for name in SOURCES:
        h.update(name.encode() + b"\0" + (HERE / name).read_bytes() + b"\0")
    return h.hexdigest()


def build_hash(build: Path = BUILD) -> str:
    """SHA-256 over every file of a compiled core (relative path and bytes), stamps and caches excluded."""
    h = hashlib.sha256()
    files = sorted(
        f for f in build.rglob("*")
        if f.is_file() and "__pycache__" not in f.relative_to(build).parts and f.name not in STAMPS
    )
    for f in files:
        rel = f.relative_to(build).as_posix()
        h.update(rel.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def stamp_ok(name: str, want: str) -> bool:
    stamp = BUILD / name
    return stamp.exists() and stamp.read_text().strip() == want


class Malformed(Exception):
    pass


def load_core():
    if not stamp_ok("SOURCE_SHA256", source_hash()):
        raise SystemExit(f"compiled core missing or stale: run bin/audit (or {HERE / 'build.sh'})")
    if not stamp_ok("BUILD_SHA256", build_hash()):
        raise SystemExit(f"compiled core differs from what build.sh produced: run bin/audit (or {HERE / 'build.sh'})")
    # Compile the hashed sources afresh: never load bytecode cached in __pycache__.
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(BUILD / ".no-bytecode-cache")
    sys.path.insert(0, str(BUILD))
    import _dafny
    import Checker
    import Game

    return _dafny, Checker, Game


def no_dup_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise Malformed(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


def bounded_int(text: str) -> int:
    """json parse_int hook: refuse huge integer literals instead of crashing on them."""
    if len(text) > MAX_NUMBER_CHARS:
        raise Malformed(f"JSON integer literal of {len(text)} digits (limit {MAX_NUMBER_CHARS})")
    return int(text)


def load_json(path: str):
    """Parse a file as strict UTF-8 JSON, whatever the locale; a byte order mark is refused."""
    try:
        text = Path(path).read_bytes().decode("utf-8")
        return json.loads(text, object_pairs_hook=no_dup_keys, parse_int=bounded_int)
    except (OSError, ValueError) as e:  # UnicodeDecodeError and JSONDecodeError are ValueErrors
        raise Malformed(f"cannot read {path}: {e}") from e


def only_keys(d: dict, allowed: set, what: str) -> None:
    extra = sorted(set(d) - allowed)
    if extra:
        raise Malformed(f"{what} has keys the schema does not allow: {extra[:10]}")


def nonempty_str(x) -> bool:
    return isinstance(x, str) and x != ""


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def edges(n: dict, nid: int, key: str) -> list[dict]:
    """The outcome or action list of a node, each entry a dict with a unique string label."""
    es = n.get(key)
    if not isinstance(es, list) or not all(isinstance(e, dict) for e in es):
        raise Malformed(f"game: node {nid} {key} is not a list of objects")
    for e in es:
        only_keys(e, EDGE_KEYS[key], f"game: an entry of node {nid} {key}")
    labels = [e.get("label") for e in es]
    if not all(isinstance(x, str) for x in labels) or len(set(labels)) != len(labels):
        raise Malformed(f"game: node {nid} {key} labels are missing or not unique: {labels!r}")
    return es


def rat(s, what: str, grammar: re.Pattern = GAME_RAT) -> Fraction:
    if not isinstance(s, str):
        raise Malformed(f"{what}: expected an exact rational string, got {s!r}"[:300])
    if len(s) > MAX_NUMBER_CHARS:
        raise Malformed(f"{what}: number string of {len(s)} characters (limit {MAX_NUMBER_CHARS})")
    if not grammar.fullmatch(s):
        raise Malformed(f"{what}: not an exact rational in the contract's format: {s!r}")
    return Fraction(s)


def fmt(q: Fraction) -> str:
    q = Fraction(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def build_game(game: dict, D, G):
    """Return (root Node, arity list, infoset names, action labels per infoset)."""
    if game.get("format") != "bluff-game-v1":
        raise Malformed("game: format is not bluff-game-v1")
    only_keys(game, GAME_KEYS, "game")
    if not nonempty_str(game.get("name")):
        raise Malformed("game: name is missing or not a non-empty string")
    for k in ("description", "known_value_source"):
        if k in game and not isinstance(game[k], str):
            raise Malformed(f"game: {k} is not a string")
    if "known_value_p0" in game:
        rat(game["known_value_p0"], "game known_value_p0")
        if not nonempty_str(game.get("known_value_source")):
            raise Malformed("game: known_value_p0 is set without a known_value_source")
    if not isinstance(game.get("nodes"), list) or not game["nodes"]:
        raise Malformed("game: nodes is not a non-empty list")
    nodes = {}
    for n in game["nodes"]:
        if not isinstance(n, dict):
            raise Malformed(f"game: a node is not an object: {n!r}"[:300])
        nid = n.get("id")
        if not is_int(nid) or nid < 0 or nid in nodes:
            raise Malformed(f"game: bad or duplicate node id {nid!r}")
        nodes[nid] = n
    root_id = game.get("root")
    if not is_int(root_id) or root_id not in nodes:
        raise Malformed("game: root is not a node id")

    infoset_ids: dict[str, int] = {}
    labels: list[list[str]] = []
    owner: list[int] = []
    seen: set[int] = set()

    # Perfect recall (CONTRACTS.md section 1, as reference/validate_game.py
    # defines it): every node of an infoset has the same sequence of its
    # player's own (infoset, action) pairs. Sequences are interned as integers
    # (0 is the empty one), so each node costs O(1).
    seq_id: dict[tuple[int, str, int], int] = {}
    recall: dict[str, int] = {}

    def extend(h: int, name: str, a: int) -> int:
        return seq_id.setdefault((h, name, a), len(seq_id) + 1)

    # Iterative post-order build (no Python recursion on deep trees). A stack
    # entry is (node id, children done?, p0's sequence, p1's sequence, depth).
    built: dict[int, object] = {}
    kids_of: dict[int, list] = {}
    stack = [(root_id, False, 0, 0, 0)]
    while stack:
        nid, done, h0, h1, depth = stack.pop()
        n = nodes.get(nid) if is_int(nid) else None
        if n is None:
            raise Malformed(f"game: child {nid!r} is not a node"[:300])
        if depth > MAX_DEPTH:
            raise Malformed(f"game: tree deeper than {MAX_DEPTH} levels, the checker's limit")
        kind = n.get("kind")
        if not done and kind in NODE_KEYS:
            only_keys(n, NODE_KEYS[kind], f"game: {kind} node {nid}")
        if kind == "terminal":
            if nid in seen:
                raise Malformed(f"game: node {nid} appears twice (not a tree)")
            seen.add(nid)
            built[nid] = G.Node_Terminal(D.BigRational(rat(n.get("payoff_p0"), f"node {nid} payoff")))
            continue
        if done:
            kids = kids_of.pop(nid)
        elif kind == "chance":
            kids = [o.get("child") for o in edges(n, nid, "outcomes")]
        elif kind == "decision":
            kids = [a.get("child") for a in edges(n, nid, "actions")]
        else:
            raise Malformed(f"game: node {nid} has unknown kind {kind!r}")
        if not done:
            kids_of[nid] = kids
            if nid in seen:
                raise Malformed(f"game: node {nid} appears twice (not a tree)")
            seen.add(nid)
            stack.append((nid, True, h0, h1, depth))
            if kind == "chance":
                hs = [(h0, h1)] * len(kids)
            else:
                player, name = n.get("player"), n.get("infoset")
                if not is_int(player) or player not in (0, 1):
                    raise Malformed(f"game: node {nid} player {player!r} is not 0 or 1")
                if not nonempty_str(name):
                    raise Malformed(f"game: node {nid} has no infoset string")
                own = h0 if player == 0 else h1
                if recall.setdefault(name, own) != own:
                    raise Malformed(
                        f"game: perfect recall fails at infoset {name!r} (node {nid}): its nodes differ in "
                        f"player {player}'s own past infosets and actions")
                hs = [(extend(h0, name, a), h1) if player == 0 else (h0, extend(h1, name, a))
                      for a in range(len(kids))]
            for c, (k0, k1) in zip(reversed(kids), reversed(hs)):
                stack.append((c, False, k0, k1, depth + 1))
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
            acts = [a.get("label") for a in n["actions"]]
            if not all(nonempty_str(x) for x in acts):
                raise Malformed(f"game: node {nid} has an empty action label")
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


def build_profile(strat: dict, game_name, names, labels, D):
    if strat.get("format") != "bluff-strategy-v1":
        raise Malformed("strategy: format is not bluff-strategy-v1")
    only_keys(strat, STRAT_KEYS, "strategy")
    if "producer" in strat and not isinstance(strat["producer"], str):
        raise Malformed("strategy: producer is not a string")
    if not nonempty_str(strat.get("game")):
        raise Malformed("strategy: game is missing or not a non-empty string")
    if strat["game"] != game_name:
        raise Malformed(f"strategy: is for game {strat['game']!r}, the game file is {game_name!r}"[:300])
    probs = strat.get("probs")
    if not isinstance(probs, dict):
        raise Malformed("strategy: probs is not an object")
    unknown = sorted(set(probs) - set(names))
    if unknown:
        raise Malformed(f"strategy: names infosets not in the game: {unknown[:10]}")
    rows = []
    for name, acts in zip(names, labels):
        if name not in probs:
            raise Malformed(f"strategy: infoset {name!r} is missing")
        row = probs[name]
        if not isinstance(row, dict):
            raise Malformed(f"strategy: infoset {name!r} is not an object of action probabilities")
        extra = set(row) - set(acts)
        if extra:
            raise Malformed(f"strategy: infoset {name!r} has unknown actions {sorted(extra)}")
        vals = []
        for a in acts:
            if a not in row:
                raise Malformed(f"strategy: infoset {name!r} is missing action {a!r}")
            q = rat(row[a], f"strategy {name!r}/{a!r}", STRAT_RAT)
            if q < 0:
                raise Malformed(f"strategy: infoset {name!r} action {a!r} is negative")
            vals.append(q)
        total = sum(vals, Fraction(0))
        if abs(total - 1) > TOL:
            shown = fmt(total)
            if len(shown) > 60:
                shown = shown[:60] + "..."
            raise Malformed(f"strategy: infoset {name!r} sums to {shown}, not 1 within 1e-6")
        rows.append(D.SeqWithoutIsStrInference([D.BigRational(v / total) for v in vals]))
    return D.SeqWithoutIsStrInference(rows)


def to_float(q) -> float:
    """The convenience float of an exact nonnegative value: nearest double, clamped
    to the largest finite double (JSON has no infinity). Underflows to 0.0 below
    about 5e-324; read the exact field for anything that matters."""
    try:
        return float(Fraction(q))
    except OverflowError:
        return sys.float_info.max


def run(game_path: str, strat_path: str) -> dict:
    D, C, G = load_core()
    game = load_json(game_path)
    strat = load_json(strat_path)
    if not isinstance(game, dict) or not isinstance(strat, dict):
        raise Malformed("game and strategy must be JSON objects")
    try:
        root, arity, names, labels, owner = build_game(game, D, G)
        prof = build_profile(strat, game.get("name"), names, labels, D)
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
        "exploitability_float": to_float(v.exploitability),
        "best_response_p0": br_map(v.br0, 0),
        "best_response_p1": br_map(v.br1, 1),
        "verified": True,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: bin/bluff check GAME.game.json STRATEGY.strategy.json", file=sys.stderr)
        return 2
    out: dict = {}

    def work():
        try:
            out["verdict"] = run(argv[1], argv[2])
        except Malformed as e:
            out["error"] = str(e)
        except SystemExit as e:
            out["internal"] = str(e)
        except RecursionError as e:
            out["internal"] = f"RecursionError ({e}): the compiled core ran out of stack, a resource limit"
        except BaseException as e:  # noqa: BLE001 (reported, never turned into a verdict)
            out["internal"] = f"{type(e).__name__}: {e}"

    # The compiled core recurses once per tree level, several frames per level
    # where the responder moves (BRG, BranchesA, the comprehension); a
    # MAX_DEPTH-deep chain of one player's nodes needs more than 1,000,000
    # frames. Give it the frames and a deep thread stack.
    sys.setrecursionlimit(4_000_000)
    threading.stack_size(1 << 29)
    t = threading.Thread(target=work)
    t.start()
    t.join()
    if "error" in out:
        print(out["error"], file=sys.stderr)
        return 2
    if "verdict" not in out:
        print("internal error, no verdict: " + out.get("internal", "unknown"), file=sys.stderr)
        return 1
    print(json.dumps(out["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
