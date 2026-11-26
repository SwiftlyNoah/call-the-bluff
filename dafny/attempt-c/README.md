# Attempt C: sequence-form verified core

A Dafny model of finite two-player zero-sum extensive-form games with chance, over Dafny's
exact `real` (rationals), with a best response computed in sequence form and proved optimal.
Everything below was run on 2026-11-20 with Dafny 4.11.0.

## Build, verify, run

```sh
dafny/attempt-c/build.sh          # audit (no assume/axiom/verify false/extern), verify all, compile to Python
uv run dafny/attempt-c/check.py GAME.game.json STRATEGY.strategy.json   # prints a bluff-verdict-v1 JSON
uv run dafny/attempt-c/tests/make_tiny.py      # writes this attempt's own tiny test games
uv run dafny/attempt-c/tests/differential.py --n 3   # exact agreement with reference/br.py
```

`check.py` exits 2 on a malformed game or strategy, or when the core returns an error.

## The idea (angle C)

For player p and the opponent's behavioral strategy, `Terms(n, s, p)` lists every terminal z
as `Term(w, path)`: `w` = chance reach times opponent reach times p's payoff, and `path` = p's own
sequence to z, the (infoset, action) pairs p plays on the way. For any behavioral strategy t of
p, p's expected utility is the bilinear sum `BSum(Z, t) = sum_z w(z) * Reach(t, path(z))`, where
`Reach` is t's realization weight of that sequence. `BSum` is linear in p's realization plan, so
`V` maximizes it by grouping terminals by p's first infoset, splitting each group by action and
taking the best action's value (recursively, per infoset). `BRValue(g, s, p) = V(Terms(g.root, s, p))`.

## What is proved (statements copied from the code)

Definitions the statements rest on (the specification; read these): `Model.dfy` (`Node`,
`Game`, `EU`, `WF`, `ValidRow`, `ValidProfile`, `AgreeOpp`, `PureOk`, `PureRow`, `Override`) and
`PerfectRecall` / `PRZ` (below). `EU(n, s, p)` is the plain recursive expected utility to p of
the tree under profile s; nothing in it mentions sequences.

**Theorem 1, upper bound over every behavioral strategy** (`Check.dfy`):

```dafny
lemma BRUpperBound(g: Game, s: Profile, t: Profile, p: nat)
  requires WF(g) && ValidProfile(g, t) && AgreeOpp(g, s, t, p)
  ensures EU(g.root, t, p) <= BRValue(g, s, p)
```

t is any valid profile that plays like s at every opponent infoset, so its rows at p's
infosets are an arbitrary behavioral strategy of p. This needs no perfect recall.

**Theorem 1', upper bound over every pure strategy** (corollary):

```dafny
lemma PureUpperBound(g: Game, s: Profile, pi: seq<nat>, p: nat)
  requires WF(g) && ValidProfile(g, s) && PureOk(g, pi)
  ensures EU(g.root, Override(g, s, pi, p), p) <= BRValue(g, s, p)
```

**Theorem 3, attainment under perfect recall**:

```dafny
lemma BRAttained(g: Game, s: Profile, p: nat)
  requires WF(g) && ValidProfile(g, s) && PerfectRecall(g, p)
  ensures PureOk(g, BRPure(g, s, p))
  ensures EU(g.root, Override(g, s, BRPure(g, s, p), p), p) == BRValue(g, s, p)
```

Together: under perfect recall, `BRValue` is exactly the maximum of p's expected utility over
all pure strategies, and equals the supremum over all behavioral strategies; `BRPure` attains it.

Perfect recall is stated in sequence form, over the p-sequences of the terminals (which do not
depend on the profile, lemma `TermsPaths`): if infoset J occurs at position k1 of one terminal's
p-sequence and at k2 of another's, the prefixes before J are equal. That is "every infoset of
p is reached by exactly one sequence of p's own moves"; since every decision node has a child,
every node of p lies on some terminal's sequence.

```dafny
ghost predicate PerfectRecall(g: Game, p: nat) { PRZ(Terms(g.root, [], p)) }
opaque ghost predicate PRZ(Z: seq<Term>) {
  forall t1, t2, k1, k2 ::
    (t1 in Z && t2 in Z && 0 <= k1 < |t1.path| && 0 <= k2 < |t2.path| &&
     t1.path[k1].inf == t2.path[k2].inf) ==> t1.path[..k1] == t2.path[..k2]
}
```

**Theorem 2, nash_conv >= 0**:

```dafny
lemma NashConvNonneg(g: Game, s: Profile)
  requires WF(g) && ValidProfile(g, s)
  ensures BRValue(g, s, 0) + BRValue(g, s, 1) >= 0.0
```

**The checker** (`Check.dfy`, compiled):

```dafny
method Check(g: Game, s: Profile) returns (r: Result)
  ensures r.Ok? ==> WF(g) && ValidProfile(g, s)
  ensures r.Ok? ==> r.value_p0 == EU(g.root, s, 0)
  ensures r.Ok? ==> r.br0 == BRValue(g, s, 0) && r.br1 == BRValue(g, s, 1)
  ensures r.Ok? ==> forall t :: ValidProfile(g, t) && AgreeOpp(g, s, t, 0) ==> EU(g.root, t, 0) <= r.br0
  ensures r.Ok? ==> forall t :: ValidProfile(g, t) && AgreeOpp(g, s, t, 1) ==> EU(g.root, t, 1) <= r.br1
  ensures r.Ok? ==> forall pi :: PureOk(g, pi) ==> EU(g.root, Override(g, s, pi, 0), 0) <= r.br0
  ensures r.Ok? ==> forall pi :: PureOk(g, pi) ==> EU(g.root, Override(g, s, pi, 1), 1) <= r.br1
  ensures r.Ok? ==> PureOk(g, r.pi0) && EU(g.root, Override(g, s, r.pi0, 0), 0) == r.br0
  ensures r.Ok? ==> PureOk(g, r.pi1) && EU(g.root, Override(g, s, r.pi1, 1), 1) == r.br1
  ensures r.Ok? ==> r.nash_conv == r.br0 + r.br1 && r.nash_conv >= 0.0
  ensures r.Ok? ==> r.expl == r.nash_conv / 2.0 && r.expl >= 0.0
  ensures WF(g) && ValidProfile(g, s) && PerfectRecall(g, 0) && PerfectRecall(g, 1) ==> r.Ok?
```

`Check` tests `WF` and `ValidProfile` at runtime (verified boolean functions) and also
recomputes `EU` of the returned pure best responses and compares it exactly with `br_p`. The last
postcondition proves that this runtime comparison never fails on a perfect-recall game, so a
successful result is guaranteed there, and on a game without perfect recall the checker either
still returns a correct verdict (the comparison holds) or refuses (`tests/data/forgetful`
is refused, as it should be). Perfect recall itself is not checked at runtime: it is not needed
for soundness, only for completeness.

## Trusted base

- The specification above: `Model.dfy` and the `PerfectRecall` / `PRZ` definitions. `BRValue`,
  `V`, `Terms` and the pickers need not be trusted: the theorems pin `BRValue` down as the
  maximum over pure strategies.
- `check.py` (unverified, kept small): JSON reading, rational parsing with `fractions`,
  infoset and action numbering, tree construction from node ids, the strategy row normalization of
  CONTRACTS.md section 2 (sum within 1e-6, then divide by the exact sum), and printing.
- Dafny 4.11.0, Z3, the Dafny-to-Python compiler and its runtime (`_dafny.BigRational`, a
  `fractions.Fraction` subclass), and CPython.
- There are no `assume`, `{:axiom}`,
  `{:verify false}` or `{:extern}` anywhere in the five `.dfy` files (`build.sh` greps for them).

## Files

| File | Lines | Content |
| --- | --- | --- |
| `Model.dfy` | 121 | games, profiles, EU, well-formedness, pure strategies |
| `SeqForm.dfy` | 244 | terminal list, realization weights, filters, V, the pure best response pickers |
| `Proofs.dfy` | 558 | bridge (tree EU = bilinear sum), linearity, partition lemmas, the bound V >= BSum |
| `Recall.dfy` | 499 | perfect recall in sequence form, disjointness, frame lemmas, attainment |
| `Check.dfy` | 307 | BRValue, the theorems, `Check` |
| `check.py` | 216 | trusted Python wrapper |

Total 1729 lines of Dafny, 352 verification tasks (PickHAttains is split per assertion), about
12 s wall to verify all five files with `--cores 2` (`build.sh`, which also compiles). Proof
stability: `dafny measure-complexity Model.dfy SeqForm.dfy Proofs.dfy Recall.dfy Check.dfy
--mutations 5 --random-seed 11 --cores 2 --verification-time-limit 60` reported no errors.

## Results (see board/claims.md, "core-c")

- Kuhn, reference profile and LP profile: `value_p0` = -1/18, `nash_conv` = 0, exploitability 0.
- Leduc LP profile (9451 nodes, 5520 terminals, 936 infosets): exploitability exactly 0, value
  -1454920850547486749701863871533/16995463438839962469905132501930 (about -0.0856), the same
  exact rational as reference/br.py; about 4.1 s per run.
- Differential test against `reference/br.py`: 82/82 profiles agree exactly on every field
  checked, over 19 games (all of games/out plus this attempt's tiny games). It also checks the
  wrapper's label mapping: p playing the printed `best_response_p` against the original opponent
  strategy earns exactly `br_value_p` by reference/br.py's own evaluation.
- `check.py` rejects (exit 2) the malformed inputs reference/br.py rejects that would otherwise
  slip past it: duplicate action labels at a node, boolean ids or players, strategy infosets
  not in the game, non-object nodes or rows, and unreadable files.

## Known limits

- Performance: the filters are recursive over sequences (O(n^2) slicing in the Python
  runtime), and the pure best response recomputes V once per level, so this is about 4x slower
  than reference/br.py on Leduc. Fine for Leduc; not tuned for larger games.
- The JS target compiles (`dafny build --target:js Check.dfy`) but was not run here.
