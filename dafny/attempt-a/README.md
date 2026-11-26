# Verified core, attempt A: inductive tree, infoset aggregation

A Dafny model of finite two-player zero-sum extensive-form games with chance, exact `real`
(rational) arithmetic everywhere, and a best response computed by infoset aggregation, proved
by induction with no `assume`, no `{:axiom}`, no `{:verify false}` and no `{:extern}`.

## Files

| File | What |
| --- | --- |
| `Math.dfy` | weighted sums (`Dot`, `Sum`) over `seq<real>` and their facts: linearity, monotonicity, convex bound, one-hot selection |
| `Game.dfy` | `datatype Node = Terminal | Chance | Decision`, behavioral semantics `Val`, pure semantics `PVal`, well-formedness `WFN`, `ValidBeh`, `ValidPure`, pure = one-hot behavioral, zero sum |
| `BestResponse.dfy` | the algorithm `BRV` and the theorems `UpperBound` and `Attain` |
| `Check.dfy` | `Check(game, profile)` producing the verdict, `BRTheorem`, `CheckSound` |
| `check.py` | the thin trusted wrapper: JSON in, compiled `Check`, verdict JSON out |
| `build.sh` | audit grep, `dafny verify`, `dafny build --target:py` into `build/checker-py` |
| `tests/games/` | hand-made games: matching pennies, a two-card one-bet game, an imperfect-recall game |
| `tests/differential.py` | exact comparison with the independent `reference/br.py` on random profiles |
| `tests/malformed.py` | malformed games and strategies `check.py` must reject (exit 2), and a 20000-deep chain it must finish |

## The algorithm

`BRV(L, P, p, sigma, A)` keeps a weighted list `L` of nodes still to expand and a list `P` of
pending decision nodes of the responder `p`. Terminals add `w * U(p, payoff)`; chance and opponent
nodes are replaced by their children with weights multiplied by the chance or `sigma`
probabilities (zero weights are dropped); responder nodes wait in `P`. When `L` is empty, the
pending nodes are grouped by infoset: for the first infoset `I`, every action `a` gets the value
of the whole group advanced along `a` (`Branches`), the best one is kept (`Argmax`, first on
ties), the choice `(I, a)` is recorded, and the rest of `P` is handled the same way. The top call
is `BR(root, p, sigma, A) = BRV([(1, root)], [], p, sigma, A)`. Termination is by the total node
count of both lists.

## What is proved (copied from the code)

The model. `Val(h, p, b, sigma)` is the expected utility to player `p` when `p` plays behavioral
strategy `b` at its own decision nodes and every other decision node follows `sigma`;
`Val(root, 0, prof, prof)` is the profile's value to p0. `PVal(h, p, s, sigma)` is the same with
`p` playing the pure strategy `s: nat -> nat` (infoset to action index). `U(p, x)` is `x` for
p0 and `-x` for p1.

`Check.dfy`, the best response theorem for any player `p` and any opponent strategy `sigma`
(no condition on `sigma` at all):

```dafny
lemma BRTheorem(root: Node, p: nat, sigma: Strat, A: seq<nat>)
  requires WFN(root, A)
  // (1a) No pure strategy of p does better.
  ensures forall s: nat -> nat :: ValidPure(s, A) ==> PVal(root, p, s, sigma) <= BR(root, p, sigma, A).v
  // (1b) No behavioral strategy of p does better.
  ensures forall b: Strat :: ValidBeh(b, A) ==> Val(root, p, b, sigma) <= BR(root, p, sigma, A).v
  // (2) The returned pure strategy attains the value, whenever the runtime
  // consistency check passes.
  ensures Consistent(BR(root, p, sigma, A).ch, BuildBR(BR(root, p, sigma, A).ch, A), A) ==>
    var br := BuildBR(BR(root, p, sigma, A).ch, A);
    ValidPure(ToFun(br), A) && PVal(root, p, ToFun(br), sigma) == BR(root, p, sigma, A).v
```

So the computed value is the maximum over all pure strategies of `p` (an upper bound on every
pure and every behavioral strategy, attained by the returned pure one).

`Check.dfy`, everything an `Ok` verdict guarantees:

```dafny
lemma CheckSound(g: GameT, prof: Strat)
  requires Check(g, prof).Ok?
  ensures WFN(g.root, g.A) && ValidBeh(prof, g.A)
  ensures var v := Check(g, prof).verdict;
    // the profile's value
    && v.value_p0 == Val(g.root, 0, prof, prof)
    // best response values: upper bounds over every pure and behavioral strategy
    && (forall s: nat -> nat :: ValidPure(s, g.A) ==> PVal(g.root, 0, s, prof) <= v.br_value_p0)
    && (forall s: nat -> nat :: ValidPure(s, g.A) ==> PVal(g.root, 1, s, prof) <= v.br_value_p1)
    && (forall b: Strat :: ValidBeh(b, g.A) ==> Val(g.root, 0, b, prof) <= v.br_value_p0)
    && (forall b: Strat :: ValidBeh(b, g.A) ==> Val(g.root, 1, b, prof) <= v.br_value_p1)
    // ... attained by the returned pure strategies
    && ValidPure(ToFun(v.br0), g.A) && PVal(g.root, 0, ToFun(v.br0), prof) == v.br_value_p0
    && ValidPure(ToFun(v.br1), g.A) && PVal(g.root, 1, ToFun(v.br1), prof) == v.br_value_p1
    // NashConv and exploitability
    && v.nash_conv == v.br_value_p0 + v.br_value_p1
    && v.nash_conv >= 0.0
    && v.exploitability == v.nash_conv / 2.0
    && v.exploitability >= 0.0
```

The supporting theorems in `BestResponse.dfy` (general lists, used by induction):

```dafny
lemma UpperBound(L: seq<Item>, P: seq<Item>, p: nat, b: Strat, sigma: Strat, A: seq<nat>)
  requires AllWF(L, A) && AllWF(P, A) && AllP(P, p) && ValidBeh(b, A)
  ensures LVal(L, p, b, sigma) + LVal(P, p, b, sigma) <= BRV(L, P, p, sigma, A).v

lemma Attain(L: seq<Item>, P: seq<Item>, p: nat, s: nat -> nat, sigma: Strat, A: seq<nat>)
  requires AllWF(L, A) && AllWF(P, A) && AllP(P, p) && ValidPure(s, A)
  requires Agrees(s, BRV(L, P, p, sigma, A).ch)
  ensures LVal(L, p, OneHot(s, A), sigma) + LVal(P, p, OneHot(s, A), sigma)
    == BRV(L, P, p, sigma, A).v
```

and in `Game.dfy`: `PureIsOneHot` (`PVal(h, p, s, sigma) == Val(h, p, OneHot(s, A), sigma)`, so
the pure semantics is the behavioral one at a one-hot strategy) and `ZeroSum`
(`Val(h, 1, sigma, sigma) == -Val(h, 0, sigma, sigma)`), which gives `nash_conv >= 0`.

### Perfect recall: where it enters

The upper bound needs no perfect recall at all: grouping can only overestimate. Attainment needs
the recorded `(infoset, action)` choices to be consistent: no infoset chosen twice with different
actions. Perfect recall implies that (every infoset is met in exactly one group, the standard
sequence-form argument), but that implication is NOT mechanized. Instead `Check` tests
`Consistent` at runtime with verified code and returns `Fail` when it does not hold. So the
theorem is stronger than "under perfect recall": every `Ok` verdict is correct for the game as
given, whatever its recall; on a game whose choices conflict, the checker refuses rather than
answer. `tests/games/forgetful.game.json` is such a game and is rejected. On every game in
`games/out` the check passes.

### Fine print on the statements

- `ValidBeh(b, A)` asks `b` to be an exact distribution at every infoset, including the
  opponent's rows, which `Val` never reads at the responder's nodes; any strategy of `p` extends
  to one. Likewise `ValidPure(s, A)` asks for a legal action at every infoset.
- `BRTheorem` puts no condition on `sigma`. `Check` additionally requires the profile to be
  exact distributions (`ValidBeh(prof, g.A)`), which the `nash_conv >= 0` proof uses.
- `Check` also checks `Sane` at runtime (players are 0 or 1, chance rows are exact distributions);
  the theorems do not need it.
- Infosets the best response never reaches get action 0.
- `BuildBR` is specified by `Lookup` (first recorded action) and computed by a verified
  `by method` (one backward pass with an array), so the compiled checker is linear in the
  number of infosets plus choices. Its implementation cannot weaken the theorems: attainment
  is stated for whatever `br` passes the runtime `Consistent` check.

## Trusted base

- `check.py`: JSON parsing (duplicate keys rejected), the infoset numbering (order of first
  appearance), the action numbering (the game file's order), the tree checks it does itself
  (integer ids, unique ids, every child used once, every node reachable from the root, labels
  unique at each node, chance probabilities in (0, 1], players the integers 0 or 1, an infoset
  belongs to one player and lists the same labels everywhere; structurally wrong documents exit 2,
  never with a traceback), and the strategy
  normalization from CONTRACTS.md (each row must sum to 1 within 1e-6, then is divided by its
  exact sum; negative entries are rejected).
- That the Dafny model (`Node`, `Val`, `PVal`) means what CONTRACTS.md says a game file means.
- Dafny 4.11.0, Boogie, Z3, the Dafny-to-Python compiler and its runtime (`_dafny.BigRational`
  is Python's `fractions.Fraction`), and CPython.
- The game files themselves.

There is no extern anywhere; the wrapper calls the compiled module directly.

## Reproduce

All commands from the repo root.

```bash
dafny/attempt-a/build.sh                       # audit grep, verify (93 obligations), compile to Python
uv run dafny/attempt-a/check.py games/out/kuhn.game.json games/out/kuhn.ref-strategy.json
uv run dafny/attempt-a/tests/differential.py --seeds 3
uv run dafny/attempt-a/tests/malformed.py
```

Measured on 2026-11-20 (Apple silicon laptop):

- Verification: `dafny verify --verify-included-files dafny/attempt-a/Check.dfy` finishes with
  93 verified, 0 errors, in about 4.8 s wall (5.9 s for the whole `build.sh`).
- Size: 1065 lines of Dafny (Math 178, Game 205, BestResponse 517, Check 165), 888 of them
  neither blank nor comment; `check.py` is 262 lines.
- Kuhn with the reference profile: value_p0 -1/18, br_value_p0 -1/18, br_value_p1 1/18,
  nash_conv 0, exploitability 0.
- Leduc (9451 nodes, 936 infosets, the largest game tried): LP profile gives exploitability 0 and
  the LP's exact value; the uniform profile gives value_p0 -5/64, br_value_p0 167/80,
  br_value_p1 383/144, the same as `reference/br.py`. Wall time of `uv run check.py`: 0.33 s on
  the LP profile, 0.37 to 0.39 s on the uniform one.
- Deep trees: a 20000-deep chain (40001 nodes, 20000 infosets) runs in 1.9 s and a 200000-deep
  one (400001 nodes) in 25 s (`tests/malformed.py` builds the first).
- Differential test: 57 random exact profiles over 19 games (every game in `games/out` plus
  `tests/games`), 0 mismatches in all five values against `reference/br.py`, and every returned
  best response attains its value when played back through the reference.
- `dafny build --target:js` also compiles (the generated `checker.js` needs `bignumber.js` to run);
  only the Python build is exercised here.
