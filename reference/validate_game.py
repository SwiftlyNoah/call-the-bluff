"""Validate a bluff-game-v1 file (CONTRACTS.md section 1).

Usage:
    uv run reference/validate_game.py GAME.game.json [GAME2.game.json ...]

Exit 0 when every file is valid (prints a one-line summary each), exit 2 otherwise with the
reasons on stderr.

Checks, in order: JSON with no duplicate keys and no floats; the JSON schema; tree shape (unique
ids, the root exists and has no parent, every other node has exactly one parent, every child
exists, every node is reachable from the root); exact chance sums (each prob in (0, 1], sum
exactly 1); unique action labels per node; infoset consistency (one player, the same action
labels in the same order, no label shared by both players); perfect recall (for each player,
every node of an infoset has the same sequence of that player's own (infoset, action) pairs).

Importable: `load_game(path)` returns a validated `Game` or raises `GameError`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fractions_util import parse_game_rational

REPO = Path(__file__).resolve().parent.parent
SCHEMA_DIR = REPO / "schema"


class GameError(ValueError):
    """A game file that must be rejected. `reasons` lists every problem found."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass
class Node:
    id: int
    kind: str  # "chance" | "decision" | "terminal"
    player: int | None = None
    infoset: str | None = None
    # (label, prob or None, child id); prob is set for chance outcomes only
    edges: list[tuple[str, Fraction | None, int]] = field(default_factory=list)
    payoff_p0: Fraction | None = None


@dataclass
class Game:
    name: str
    root: int
    nodes: dict[int, Node]
    # infoset label -> (player, action labels in order)
    infosets: dict[str, tuple[int, list[str]]]
    known_value_p0: Fraction | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def infoset_actions(self, player: int | None = None) -> dict[str, list[str]]:
        return {
            i: list(acts)
            for i, (p, acts) in self.infosets.items()
            if player is None or p == player
        }


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON key {k!r}")
        out[k] = v
    return out


class _Float:
    """Marker for a JSON float, so the schema (strings only) rejects it by type."""

    def __init__(self, text: str):
        self.text = text


def load_json(path: str | Path, allow_floats: bool = False) -> Any:
    """Load JSON rejecting duplicate keys. Floats become an invalid marker unless allowed."""
    text = Path(path).read_text()
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicates,
        parse_float=(float if allow_floats else _Float),
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-finite number {c}")),
    )


def _schema_errors(doc: Any, schema_name: str) -> list[str]:
    import jsonschema

    schema = json.loads((SCHEMA_DIR / schema_name).read_text())
    validator = jsonschema.Draft202012Validator(schema)
    out = []
    for err in sorted(validator.iter_errors(_json_safe(doc)), key=lambda e: list(e.path)):
        loc = "/".join(str(p) for p in err.absolute_path) or "(root)"
        out.append(f"schema: at {loc}: {err.message}")
    return out


def _json_safe(doc: Any) -> Any:
    # A float marker becomes an actual float so jsonschema reports "is not of type 'string'".
    if isinstance(doc, _Float):
        return float(doc.text)
    if isinstance(doc, dict):
        return {k: _json_safe(v) for k, v in doc.items()}
    if isinstance(doc, list):
        return [_json_safe(v) for v in doc]
    return doc


def schema_errors(doc: Any, schema_name: str) -> list[str]:
    """Schema errors for `doc` against `schema/<schema_name>` (empty list when valid)."""
    return _schema_errors(doc, schema_name)


def _build(doc: dict[str, Any]) -> tuple[Game | None, list[str]]:
    errors: list[str] = []
    nodes: dict[int, Node] = {}
    for raw in doc["nodes"]:
        nid = raw["id"]
        if nid in nodes:
            errors.append(f"duplicate node id {nid}")
            continue
        kind = raw["kind"]
        node = Node(id=nid, kind=kind)
        if kind == "chance":
            for o in raw["outcomes"]:
                try:
                    p = parse_game_rational(o["prob"])
                except ValueError as e:
                    errors.append(f"node {nid}: {e}")
                    p = Fraction(0)
                node.edges.append((o["label"], p, o["child"]))
        elif kind == "decision":
            node.player = raw["player"]
            node.infoset = raw["infoset"]
            node.edges = [(a["label"], None, a["child"]) for a in raw["actions"]]
        else:
            try:
                node.payoff_p0 = parse_game_rational(raw["payoff_p0"])
            except ValueError as e:
                errors.append(f"node {nid}: {e}")
        nodes[nid] = node
    known = None
    if "known_value_p0" in doc:
        try:
            known = parse_game_rational(doc["known_value_p0"])
        except ValueError as e:
            errors.append(f"known_value_p0: {e}")
    if errors:
        return None, errors
    return Game(name=doc["name"], root=doc["root"], nodes=nodes, infosets={},
                known_value_p0=known, raw=doc), []


def _check_tree(g: Game) -> list[str]:
    errors: list[str] = []
    if g.root not in g.nodes:
        return [f"root {g.root} is not a node"]
    parents: dict[int, list[int]] = {nid: [] for nid in g.nodes}
    for node in g.nodes.values():
        for label, _, child in node.edges:
            if child not in g.nodes:
                errors.append(f"node {node.id} edge {label!r} points to missing node {child}")
            else:
                parents[child].append(node.id)
    if parents[g.root]:
        errors.append(f"root {g.root} has parent(s) {parents[g.root]}")
    for nid, ps in parents.items():
        if nid != g.root and len(ps) != 1:
            errors.append(f"node {nid} has {len(ps)} parents {ps[:5]}, expected exactly 1")
    if errors:
        return errors
    seen = {g.root}
    stack = [g.root]
    while stack:
        for _, _, c in g.nodes[stack.pop()].edges:
            if c not in seen:
                seen.add(c)
                stack.append(c)
    unreached = sorted(set(g.nodes) - seen)
    if unreached:
        errors.append(f"{len(unreached)} node(s) unreachable from root, e.g. {unreached[:5]}")
    return errors


def _check_nodes(g: Game) -> list[str]:
    errors: list[str] = []
    for node in g.nodes.values():
        labels = [lab for lab, _, _ in node.edges]
        if len(set(labels)) != len(labels):
            errors.append(f"node {node.id}: duplicate edge labels {labels}")
        if node.kind == "chance":
            total = Fraction(0)
            for lab, p, _ in node.edges:
                assert p is not None
                if not (0 < p <= 1):
                    errors.append(f"node {node.id}: outcome {lab!r} prob {p} not in (0, 1]")
                total += p
            if total != 1:
                errors.append(f"node {node.id}: chance probs sum to {total}, not exactly 1")
    return errors


def _check_infosets(g: Game) -> list[str]:
    errors: list[str] = []
    infosets: dict[str, tuple[int, list[str]]] = {}
    for node in g.nodes.values():
        if node.kind != "decision":
            continue
        assert node.infoset is not None and node.player is not None
        labels = [lab for lab, _, _ in node.edges]
        if node.infoset not in infosets:
            infosets[node.infoset] = (node.player, labels)
            continue
        p, acts = infosets[node.infoset]
        if p != node.player:
            errors.append(
                f"infoset {node.infoset!r}: node {node.id} is player {node.player}, "
                f"others are player {p}"
            )
        if acts != labels:
            errors.append(
                f"infoset {node.infoset!r}: node {node.id} has actions {labels}, others {acts}"
            )
    g.infosets = infosets
    return errors


def player_sequences(g: Game) -> dict[int, tuple[tuple[tuple[str, str], ...], ...]]:
    """node id -> (sequence of player 0, sequence of player 1), each a tuple of (infoset, action)."""
    out: dict[int, tuple[tuple[tuple[str, str], ...], ...]] = {g.root: ((), ())}
    stack = [g.root]
    while stack:
        nid = stack.pop()
        node = g.nodes[nid]
        seqs = out[nid]
        for label, _, child in node.edges:
            if node.kind == "decision":
                assert node.player is not None and node.infoset is not None
                new = list(seqs)
                new[node.player] = seqs[node.player] + ((node.infoset, label),)
                out[child] = tuple(new)
            else:
                out[child] = seqs
            stack.append(child)
    return out


def _check_perfect_recall(g: Game) -> list[str]:
    errors: list[str] = []
    seqs = player_sequences(g)
    first: dict[str, tuple[int, tuple[tuple[str, str], ...]]] = {}
    for node in g.nodes.values():
        if node.kind != "decision":
            continue
        assert node.player is not None and node.infoset is not None
        seq = seqs[node.id][node.player]
        if node.infoset not in first:
            first[node.infoset] = (node.id, seq)
        elif first[node.infoset][1] != seq:
            other, oseq = first[node.infoset]
            errors.append(
                f"perfect recall violated at infoset {node.infoset!r}: node {other} has "
                f"player {node.player}'s history {list(oseq)}, node {node.id} has {list(seq)}"
            )
    return errors


def _float_literals(doc: Any, path: str = "(root)") -> list[str]:
    # JSON Schema counts 1.0 as an integer, so a float in an id, child or player field would
    # pass the schema and then break the tree code. Game files never contain a JSON float.
    if isinstance(doc, _Float | float):
        text = doc.text if isinstance(doc, _Float) else repr(doc)
        return [f"at {path}: float literal {text} (game files use integers and exact strings)"]
    if isinstance(doc, dict):
        return [e for k, v in doc.items() for e in _float_literals(v, f"{path}/{k}")]
    if isinstance(doc, list):
        return [e for i, v in enumerate(doc) for e in _float_literals(v, f"{path}/{i}")]
    return []


def validate_game_doc(doc: Any) -> Game:
    """Validate a parsed game document; return the Game or raise GameError with all reasons."""
    errors = _float_literals(doc)[:10] + _schema_errors(doc, "game.schema.json")
    if errors:
        raise GameError(errors)
    game, errors = _build(doc)
    if game is None:
        raise GameError(errors)
    errors = _check_tree(game)
    if errors:
        raise GameError(errors)
    errors = _check_nodes(game) + _check_infosets(game)
    if errors:
        raise GameError(errors)
    errors = _check_perfect_recall(game)
    if errors:
        raise GameError(errors)
    return game


def load_game(path: str | Path) -> Game:
    """Load and validate a game file; raise GameError with every reason on failure."""
    try:
        doc = load_json(path)
    except (OSError, ValueError) as e:
        raise GameError([f"cannot read {path}: {e}"]) from e
    return validate_game_doc(doc)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    status = 0
    for path in argv:
        try:
            g = load_game(path)
        except GameError as e:
            status = 2
            print(f"INVALID {path}", file=sys.stderr)
            for r in e.reasons:
                print(f"  {r}", file=sys.stderr)
            continue
        n_terminal = sum(1 for n in g.nodes.values() if n.kind == "terminal")
        per_player = [sum(1 for p, _ in g.infosets.values() if p == q) for q in (0, 1)]
        print(
            f"OK {path}: game {g.name!r}, {len(g.nodes)} nodes, {n_terminal} terminals, "
            f"infosets p0={per_player[0]} p1={per_player[1]}, perfect recall holds"
        )
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
