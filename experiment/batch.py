"""Run a batch of experiment trials, a few at a time, resumably.

Usage:
    uv run experiment/batch.py calibrate [--jobs 4]
    uv run experiment/batch.py matrix [--jobs 4] [--reps 4] [--cells kuhn-file,kuhn-rules,...]
        [--models claude-haiku-4-5,...]
    uv run experiment/batch.py plan ...           # print the trial list and what is already done

Each trial is `run.py trial` under a fixed run id (cal-<game>-<variant>-<model> or
m-<cell>-<model>-r<k>), so the batch can be killed and restarted at any point: a run whose
result.json has a summary with agent_finished true, or that hit its wall-clock limit (an agent
outcome, graded on what it left), is skipped. A run that did not finish (the
process was killed, the pool refused the account, the agent crashed before its result event) is
an infrastructure failure, not an agent failure: its directory is moved to runs/_infra_failed/
with a timestamp and the trial is run again, at most --max-attempts times in all.

Trials are submitted in the given order (all Kuhn cells, then Leduc, then Red/Black), so a cut
leaves complete cells. A progress line per trial goes to stdout and experiment/runs/_batch.log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXP = Path(__file__).resolve().parent
REPO = EXP.parent
RUNS = EXP / "runs"
FAILED = RUNS / "_infra_failed"
LOG = RUNS / "_batch.log"

MODELS = ["claude-haiku-4-5", "claude-haiku-5-5", "claude-sonnet-4-5", "claude-sonnet-5-5",
          "claude-opus-5-5", "claude-fable-5-2"]
# cell name -> (game, variant, extra run.py args)
CELLS = {
    "kuhn-file": ("kuhn", "file", []),
    "kuhn-rules": ("kuhn", "rules", []),
    "leduc-file": ("leduc", "file", []),
    "leduc-rules": ("leduc", "rules", []),
    # rb-21: payoffs in [-1, 1] (win probability scaled), so the natural pot is 1.
    "rb21-file": ("redblack", "file",
                  ["--game-file", "games/out/redblack/rb-21.game.json", "--pot", "1"]),
}
CELL_ORDER = list(CELLS)
_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with _lock:
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")


def status(run_id: str) -> str:
    """done, absent, ungraded (agent finished, grade missing) or broken (agent did not finish)."""
    d = RUNS / run_id
    if not d.exists():
        return "absent"
    try:
        meta = json.loads((d / "result.json").read_text())
    except (OSError, json.JSONDecodeError):
        return "broken"
    s = meta.get("summary") or {}
    if "ended_at" not in meta and "agent" not in meta:
        # Possibly still running in another process (a batch restarted under live trials).
        try:
            started = dt.datetime.fromisoformat(meta["started_at"])
            age = (dt.datetime.now().astimezone() - started).total_seconds()
            if age < meta.get("timeout_s", 1800) + 600:
                return "running"
        except (KeyError, ValueError):
            pass
    if s.get("agent_finished"):
        return "done"
    # A trial killed at its wall-clock limit is an agent outcome (it was too slow), graded on
    # whatever strategy.json it left; rerunning it would select for fast trials.
    if (meta.get("agent") or {}).get("timed_out") and "summary" in meta:
        return "done"
    # The agent finished (a result event was parsed) but grading crashed or never ran: that is
    # our bug, not the agent's, so the trial is regraded, never rerun.
    if (meta.get("agent") or {}).get("subtype") is not None:
        return "ungraded"
    return "broken"


POOL_WAIT_S = 300
MAX_POOL_WAITS = 60


def pool_refused(run_id: str) -> bool:
    try:
        err = (RUNS / run_id / "stderr.txt").read_text()
    except OSError:
        return False
    return "claude-pool:" in err and ("leave point" in err or "limit" in err or "room" in err)


def move_aside(run_id: str) -> None:
    FAILED.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.move(str(RUNS / run_id), str(FAILED / f"{run_id}.{stamp}"))


def run_trial(spec: dict, max_attempts: int, timeout_min: float, budget: float) -> str:
    rid = spec["run_id"]
    attempt, pool_waits = 0, 0
    while attempt < max_attempts:
        attempt += 1
        st = status(rid)
        if st == "done":
            return "done"
        if st == "running":
            log(f"{rid}: already running in another process; left to it")
            return "running"
        if st == "ungraded":
            log(f"{rid}: agent finished but grading did not; regrading")
            subprocess.run(["uv", "run", "experiment/run.py", "grade", str(RUNS / rid)], cwd=REPO,
                           capture_output=True, check=False)
            if status(rid) == "done":
                return "done"
            log(f"{rid}: regrade failed; left in place for a look")
            return "failed"
        if st == "broken":
            log(f"{rid}: unfinished run moved to _infra_failed")
            move_aside(rid)
        cmd = ["uv", "run", "experiment/run.py", "trial", "--game", spec["game"],
               "--variant", spec["variant"], "--model", spec["model"], "--run-id", rid,
               "--timeout-min", str(timeout_min), "--max-budget-usd", str(budget)] + spec["extra"]
        log(f"{rid}: start (attempt {attempt})")
        t0 = time.time()
        (RUNS / "_logs").mkdir(parents=True, exist_ok=True)
        with open(RUNS / "_logs" / f"{rid}.log", "a") as out:
            p = subprocess.run(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, check=False)
        st = status(rid)
        if st == "ungraded":
            continue
        refused = st == "broken" and pool_refused(rid)
        summary = {}
        if (RUNS / rid / "result.json").exists():
            try:
                summary = json.loads((RUNS / rid / "result.json").read_text()).get("summary") or {}
            except json.JSONDecodeError:
                pass
        log(f"{rid}: {st} rc={p.returncode} {time.time() - t0:.0f}s "
            f"claim={summary.get('claim_converged')} expl={summary.get('exploitability_float')} "
            f"strict={summary.get('strict_accepted')}")
        if st == "done":
            return "done"
        if refused and pool_waits < MAX_POOL_WAITS:
            # The pool refused the account (it is at its leave point): not the trial's fault, and
            # retrying at once only burns attempts. Wait for room without counting the attempt.
            pool_waits += 1
            attempt -= 1
            log(f"{rid}: pool refused the account; waiting {POOL_WAIT_S}s ({pool_waits})")
            time.sleep(POOL_WAIT_S)
            continue
        time.sleep(30)
    if status(rid) == "ungraded":
        subprocess.run(["uv", "run", "experiment/run.py", "grade", str(RUNS / rid)], cwd=REPO,
                       capture_output=True, check=False)
        return "done" if status(rid) == "done" else "failed"
    return "failed"


def plan(args: argparse.Namespace) -> list[dict]:
    specs = []
    if args.cmd == "calibrate":
        for m in args.models:
            game, variant, extra = CELLS["leduc-file"]
            specs.append({"run_id": f"cal-leduc-file-{m}", "game": game, "variant": variant,
                          "model": m, "extra": extra})
        return specs
    for cell in args.cells:
        game, variant, extra = CELLS[cell]
        # Replicate-major within a cell, so a cut mid-cell still leaves every model balanced.
        for r in range(1, args.reps + 1):
            for m in args.models:
                specs.append({"run_id": f"m-{cell}-{m}-r{r}", "game": game, "variant": variant,
                              "model": m, "extra": extra})
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["calibrate", "matrix", "plan"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--cells", default=",".join(CELL_ORDER))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--timeout-min", type=float, default=30)
    ap.add_argument("--max-budget-usd", type=float, default=5.0)
    args = ap.parse_args()
    args.cells = [c for c in args.cells.split(",") if c]
    args.models = [m for m in args.models.split(",") if m]
    if args.cmd == "plan":
        args.cmd = "matrix"
        for s in plan(args):
            print(s["run_id"], status(s["run_id"]))
        return
    specs = plan(args)
    RUNS.mkdir(parents=True, exist_ok=True)
    busy = os.environ.get("CLAUDE_POOL_BUSY")
    stop = threading.Event()

    def keep_busy() -> None:
        # Tell claude-pool that long work is in flight, so a planned switch waits for it.
        while busy and not stop.is_set():
            Path(busy).touch()
            stop.wait(60)

    threading.Thread(target=keep_busy, daemon=True).start()
    todo = [s for s in specs if status(s["run_id"]) != "done"]
    log(f"batch {args.cmd}: {len(specs)} trials, {len(specs) - len(todo)} already done, "
        f"jobs {args.jobs}")
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda s: run_trial(s, args.max_attempts, args.timeout_min,
                                                  args.max_budget_usd), todo))
    stop.set()
    failed = [s["run_id"] for s, r in zip(todo, results) if r not in ("done", "running")]
    log(f"batch {args.cmd} finished: {len(todo) - len(failed)} run, {len(failed)} failed "
        f"{failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
