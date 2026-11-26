"""Read-only loader for the Red/Black solver's pure-Python game core.

Imports `core.py` and `rounds.py` from ~/coding/red_black/simulator by file
path, without importing the package (whose other modules pull numba), and
without writing bytecode into that tree. Nothing here writes to red_black.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True

RB_ROOT = Path(os.environ.get("RED_BLACK_ROOT", Path.home() / "coding" / "red_black"))
SOLVER_DIR = RB_ROOT / "simulator" / "src" / "redblack" / "solver"

_PKG = "rb_solver_ro"


def _load():
    if f"{_PKG}.rounds" in sys.modules:
        return sys.modules[f"{_PKG}.core"], sys.modules[f"{_PKG}.rounds"]
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(SOLVER_DIR)]
    sys.modules[_PKG] = pkg
    mods = {}
    for name in ("core", "rounds"):
        spec = importlib.util.spec_from_file_location(f"{_PKG}.{name}", SOLVER_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{_PKG}.{name}"] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["core"], mods["rounds"]


core, rounds = _load()


def class_id(ec) -> str:
    """Same id scheme as simulator/scripts/export_explorer.py::class_id."""
    who = "r" if ec.loser < 0 else str(ec.loser)
    s0, s1 = ec.s0, ec.s1
    return f"d{who}-{s0.n}.{s0.r_lb}.{s0.b_lb}-{s1.n}.{s1.r_lb}.{s1.b_lb}"


def parse_class_id(cid: str):
    m = cid.split("-")
    who = m[0][1:]
    loser = -1 if who == "r" else int(who)
    trips = [tuple(int(v) for v in part.split(".")) for part in m[1:]]
    return rounds.EntryClass2(loser, rounds.SeatStat(*trips[0]), rounds.SeatStat(*trips[1]))


def _load_pack():
    name = f"{_PKG}.pack"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOLVER_DIR / "pack.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pack = _load_pack()
