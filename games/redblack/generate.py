"""Write CONTRACTS.md game files for small heads-up Red/Black variants.

    uv run games/redblack/generate.py            # every variant in VARIANTS
    uv run games/redblack/generate.py rb-pennies # just one
    uv run games/redblack/generate.py --sizes    # print sizes, write nothing

Output: games/out/redblack/<name>.game.json. Utility: win = +1, loss = -1 to
player 0 (the research's convention: value v means P(seat 0 wins) = (1+v)/2,
so the 25% gate class has value -1/2).
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import model as M  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out" / "redblack"

PENNIES_DESC = (
    "Red/Black gate endgame (Red/Black Paper II, lessons.md #3; solver test GATE_CLASS = "
    "EntryClass2(-1, SeatStat(1,1,0), SeatStat(2,1,1))). Seat 0 holds one card, a known red, and "
    "opens; seat 1 holds a known red and a known black and chooses their order in secret. Standard "
    "rules from there. Two-stage matching pennies: seat 0 wins with probability exactly 1/4, value -1/2."
)

VARIANTS: dict[str, M.Config] = {
    "rb-pennies": M.Config(
        name="rb-pennies",
        description=PENNIES_DESC,
        start=M.Start(hands=("R", "RB"), phase="reorder", starter=0),
    ),
    "rb-pennies-inner": M.Config(
        name="rb-pennies-inner",
        description=(
            "Second stage of the gate endgame (solver test EntryClass2(1, SeatStat(1,1,0), SeatStat(2,1,1))): "
            "seat 1 just lost a procurement holding a known red and a known black and discards one in "
            "secret; seat 0 (one known red) then opens. Pure matching pennies: value 0."
        ),
        start=M.Start(hands=("R", "RB"), phase="discard", loser=1),
    ),
    "rb-pennies-leaky": M.Config(
        name="rb-pennies-leaky",
        description=(
            "DELIBERATELY WRONG SPEC, kept as a mutant: rb-pennies, but the opponent learns which slot a "
            "discard came from (what stable card ids leaked in the v0 solver, lessons.md #1). "
            "It violates RULES 4d/6a; its value is not Red/Black's."
        ),
        start=M.Start(hands=("R", "RB"), phase="reorder", starter=0),
        leak_discard=True,
    ),
    "rb-h1": M.Config(
        name="rb-h1",
        description=(
            "Heads-up Red/Black from the real 26 red / 26 black deck, 1 card per seat (real game: 5). "
            "One round. Every other rule standard."
        ),
        hand_size=1,
    ),
    "rb-21": M.Config(
        name="rb-21",
        description=(
            "Heads-up Red/Black from the real 26 red / 26 black deck, seat 0 dealt 2 cards and seat 1 dealt 1 "
            "(real game: 5 each); seat 0 opens. Every other rule standard."
        ),
        hand_size=(2, 1),
    ),
    "rb-12": M.Config(
        name="rb-12",
        description=(
            "Heads-up Red/Black from the real 26 red / 26 black deck, seat 0 dealt 1 card and seat 1 dealt 2 "
            "(real game: 5 each); seat 0 opens. Every other rule standard."
        ),
        hand_size=(1, 2),
    ),
    "rb-31": M.Config(
        name="rb-31",
        description=(
            "Heads-up Red/Black from the real 26 red / 26 black deck, seat 0 dealt 3 cards and seat 1 dealt 1 "
            "(real game: 5 each); seat 0 opens. Up to 3 rounds. Every other rule standard."
        ),
        hand_size=(3, 1),
    ),
    "rb-pos-2v2": M.Config(
        name="rb-pos-2v2",
        description=(
            "A fixed public-color position: each seat holds a red and a black; seat 0 opens "
            "with hand orders private. Exists so that later rounds still have reorder choices "
            "(difftest's mutant bed)."
        ),
        start=M.Start(hands=("RB", "RB"), phase="reorder", starter=0),
    ),
    "rb-h2": M.Config(
        name="rb-h2",
        description=(
            "Heads-up Red/Black from the real 26 red / 26 black deck, 2 cards per seat (real game: 5). "
            "Up to 3 rounds. Every other rule standard."
        ),
        hand_size=2,
    ),
    "rb-h2-d4": M.Config(
        name="rb-h2-d4",
        description=(
            "Heads-up Red/Black, 2 cards per seat dealt from a 4-card deck of 2 red and 2 black "
            "(real game: 5 each from 26/26), so each seat can deduce the other's colors but not their order. "
            "Every other rule standard."
        ),
        hand_size=2,
        deck_red=2,
        deck_black=2,
    ),
    "rb-h3": M.Config(
        name="rb-h3",
        description=(
            "Heads-up Red/Black from the real 26 red / 26 black deck, 3 cards per seat (real game: 5). "
            "Up to 5 rounds. Every other rule standard."
        ),
        hand_size=3,
    ),
}

# rb-31 (985k nodes, 137 MB), rb-h2-d4 (1.3M) and rb-h2 (2.3M) are written only when named:
# too big to commit (GitHub refuses files over 100 MB).
DEFAULT = ["rb-h1", "rb-pennies-inner", "rb-pennies", "rb-pennies-leaky", "rb-21", "rb-12"]

KNOWN = {
    # name: (value to p0, source)
    "rb-pennies": (
        "-1/2",
        "Red/Black Paper II (research/redblack-cfrplus.tex, 25% to the first seat) and "
        "lessons.md #3; win=+1/loss=-1 so 25% is -1/2",
    ),
    "rb-pennies-inner": ("0", "Red/Black solver test test_inner_pennies_stage_is_even (matching pennies)"),
}


def q(x: Fraction | int) -> str:
    f = Fraction(x)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def build(cfg: M.Config) -> list[dict]:
    """Pre-order node list. Forced moves (one legal successor) are skipped:
    they carry no choice and no information."""
    nodes: list[dict] = []

    def new(node: dict) -> int:
        node = {"id": len(nodes), **node}
        nodes.append(node)
        return node["id"]

    def rec(s: M.State) -> int:
        while True:
            if s.phase == "over":
                return new({"kind": "terminal", "payoff_p0": q(M.payoff_p0(s))})
            if s.phase == "deal":
                nid = new({"kind": "chance", "outcomes": []})
                for label, p, child in M.deal_children(cfg):
                    nodes[nid]["outcomes"].append({"label": label, "prob": q(p), "child": rec(child)})
                return nid
            succ = M.legal(s, cfg)
            if len(succ) == 1:
                s = succ[0][1]
                continue
            p = M.player(s)
            nid = new({"kind": "decision", "player": p, "infoset": M.infoset_key(s, p), "actions": []})
            for label, child in succ:
                nodes[nid]["actions"].append({"label": label, "child": rec(child)})
            return nid

    rec(M.initial_state(cfg))
    return nodes


def game_doc(cfg: M.Config, nodes: list[dict]) -> dict:
    doc = {"format": "bluff-game-v1", "name": cfg.name, "description": cfg.description}
    if cfg.name in KNOWN:
        doc["known_value_p0"], doc["known_value_source"] = KNOWN[cfg.name]
    doc["root"] = 0
    doc["nodes"] = nodes
    return doc


def sizes(nodes: list[dict]) -> dict:
    kinds = {"chance": 0, "decision": 0, "terminal": 0}
    isets = {0: set(), 1: set()}
    for n in nodes:
        kinds[n["kind"]] += 1
        if n["kind"] == "decision":
            isets[n["player"]].add(n["infoset"])
    return {"nodes": len(nodes), **kinds, "infosets_p0": len(isets[0]), "infosets_p1": len(isets[1])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--sizes", action="store_true", help="print sizes only, write nothing")
    args = ap.parse_args(argv)
    for name in args.names or DEFAULT:
        cfg = VARIANTS[name]
        nodes = build(cfg)
        info = sizes(nodes)
        if not args.sizes:
            path = OUT / f"{name}.game.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(game_doc(cfg, nodes), separators=(",", ":")) + "\n")
            info["file"] = str(path.relative_to(OUT.parents[2]))
            info["bytes"] = path.stat().st_size
        print(name, json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
