# Verified core, attempt B: flat node table

A Dafny model of finite two-player zero-sum extensive-form games with chance, stored as a flat
node table that mirrors the JSON game file, plus a best response, a checker, and proofs. All
arithmetic is Dafny `real`, which compiles to exact rationals (`fractions.Fraction` in Python).
No `assume`, `{:axiom}`, `{:verify false}` or `{:extern}` anywhere in the `.dfy` files.

| File | What |
| --- | --- |
| `Model.dfy` | `Node`, `Game` (node table, root, heights, infoset table), `WF`, behavioral and pure strategies, expected utility `Val`, `ZeroSum` |
| `BestResponse.dfy` | the best-response algorithm `BR` and the bound `BRBound` |
| `Check.dfy` | the theorems, `BestResponseOf`, `Check`, `NashConvNonNegative` |
| `check.py` | the thin trusted adapter: JSON in, Dafny values, `Check`, verdict JSON out |
| `build.sh` | verify everything, then `dafny build --target:py` into `build/bluffcore-py` |
| `tests/` | my own tiny games, a pytest suite against an independent brute force, a differential script |

## The representation (angle B)

`Game(nodes, root, height, infosetPlayer, infosetActions)`: `nodes[i]` is `Terminal(payoff)`,
`Chance(probs, kids)` or `Decision(player, infoset, kids)`, with children as node indices, just
like the JSON. Recursion is justified by the explicit `height` array: `WF` requires every child to
have a strictly smaller height than its parent, so `decreases g.height[n]` is the well-founded
measure. The adapter computes heights (longest path to a leaf); the verified `WF` check rejects
anything inconsistent, so heights are not trusted. Sharing (a DAG) is allowed by the model. A
strategy is `seq<seq<real>>`, one row per infoset id, so one value holds a whole profile.

## What is proved

Definitions (in `Model.dfy`): `Val(g, p, beta, sigma, n)` is player p's expected utility from node
n when p plays `beta` at p's infosets and the opponent plays `sigma` at theirs, defined by direct
recursion (terminal: payoff, negated for player 1; chance and decision nodes: probability-weighted
sum over children). `ValidBeh(g, p, b)` says b is a behavioral strategy for p (each of p's rows is
nonnegative and sums to exactly 1). `PureValid(g, p, pi)` says pi picks an in-range action at each
of p's infosets; `OneHot(g, pi)` is the behavioral strategy that plays pi with probability one.

**Theorem 1a, behavioral upper bound** (`Check.dfy`):

```dafny
lemma BehavioralUpperBound(g: Game, p: nat, sigma: Strat, beta: Strat)
  requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma) && ValidBeh(g, p, beta)
  ensures Val(g, p, beta, sigma, g.root) <= BRValue(g, p, sigma)
```

**Theorem 1b, pure upper bound** (`Check.dfy`):

```dafny
lemma PureUpperBound(g: Game, p: nat, sigma: Strat, pi: seq<nat>)
  requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma) && PureValid(g, p, pi)
  ensures Val(g, p, OneHot(g, pi), sigma, g.root) <= BRValue(g, p, sigma)
```

Both hold for **every** well-formed game, with or without perfect recall. They rest on
`BRBound` (`BestResponse.dfy`), proved by induction on the total subtree size of the work lists:

```dafny
lemma BRBound(g: Game, p: nat, beta: Strat, sigma: Strat, P: seq<WN>, R: seq<WN>)
  requires Inv(g, p, beta, sigma) && ValidBeh(g, p, beta) && NodesIn(g, P) && AllPDec(g, p, R)
  ensures VS(g, p, beta, sigma, P) + VS(g, p, beta, sigma, R) <= BR(g, p, sigma, P, R).0
```

**Theorem 2, the checker** (`Check.dfy`). `Check` is a compiled method; whenever it returns `Ok`,
the verdict satisfies `Sound`:

```dafny
method Check(g: Game, sigma: Strat) returns (r: Result)
  ensures r.Ok? ==> WF(g) && ValidBeh(g, 0, sigma) && ValidBeh(g, 1, sigma) && Sound(g, sigma, r.v)

ghost predicate IsBestResponse(g: Game, p: nat, sigma: Strat, value: real, pi: seq<nat>)
  requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma)
{
  PureValid(g, p, pi)
  && Val(g, p, OneHot(g, pi), sigma, g.root) == value
  && (forall beta | ValidBeh(g, p, beta) :: Val(g, p, beta, sigma, g.root) <= value)
  && (forall pi' | PureValid(g, p, pi') :: Val(g, p, OneHot(g, pi'), sigma, g.root) <= value)
}

ghost predicate Sound(g: Game, sigma: Strat, v: Verdict)
  requires WF(g) && ValidBeh(g, 0, sigma) && ValidBeh(g, 1, sigma)
{
  v.valueP0 == Val(g, 0, sigma, sigma, g.root)
  && Val(g, 1, sigma, sigma, g.root) == -v.valueP0
  && IsBestResponse(g, 0, sigma, v.brValue0, v.bestResponse0)
  && IsBestResponse(g, 1, sigma, v.brValue1, v.bestResponse1)
  && v.nashConv == v.brValue0 + v.brValue1
  && v.nashConv >= 0.0
  && v.exploitability == v.nashConv / 2.0
}
```

So an `Ok` verdict means: `value_p0` is the profile's exact value; each `br_value_p` equals the
**maximum over all pure strategies and over all behavioral strategies** of p against the
opponent's half of the profile, and the returned pure strategy attains it; `nash_conv = br0 + br1
>= 0`; `exploitability = nash_conv / 2`. `NashConvNonNegative` restates the `nash_conv >= 0` part.
`WF` and `ValidBeh` are compiled predicates that `Check` runs on its input, so they are not
assumptions about the caller.

### How attainment is established (and what is not proved)

The algorithm groups p's decision nodes by infoset and picks one action per group. That makes its
value an upper bound in any game (Theorem 1). Attainment needs perfect recall, and is established
**per run, by a verified check rather than a general proof**: `BestResponseOf` evaluates the
returned pure strategy with `Val` and `Check` returns `Err` unless that value equals the bound.
Hence perfect recall is not a precondition anywhere and is not trusted: on a game without it,
`Check` either still returns a correct verdict (if the bound happens to be attained) or refuses.

Not proved:

1. **Completeness**: that `Check` returns `Ok` on every perfect-recall game. It is tested, not
   proved: it returned `Ok` on all 14 perfect-recall games tried (the 13 in `tests/test_attempt_b.py` plus Leduc), and it refuses
   `tests/games/forgetful_b.game.json`, an imperfect-recall game built so that the grouping
   overestimates (`test_imperfect_recall_is_refused`).
2. That the JSON-to-Dafny translation in `check.py` is faithful (it is in the trusted base).
3. No perfect-recall checker is included: none is needed for soundness (see above).

## Trusted base

The Dafny toolchain (4.11.0) and Z3; the Python compiler backend and `_dafny` runtime
(`BigRational` is a `fractions.Fraction`); `check.py` (JSON parsing, rational parsing,
numbering nodes and infosets, checking that all nodes of an infoset list the same action labels,
computing heights, normalizing strategy rows within 1e-6 then dividing by the exact sum, printing
fractions); and the game files themselves.

## Running it

```sh
dafny/attempt-b/build.sh                       # verify (135 obligations) and compile to Python
uv run dafny/attempt-b/check.py games/out/kuhn.game.json games/out/kuhn.ref-strategy.json
uv run pytest dafny/attempt-b/tests -q         # 18 tests vs an independent brute force
uv run python dafny/attempt-b/tests/differential.py   # vs reference/br.py on every game and shipped profile
```

`check.py` rebuilds `build/bluffcore-py` WITH verification (no `--no-verify`) whenever it is
missing or older than any `.dfy` file, so a verdict marked `"verified": true` never comes from a
stale or unverified build. JSON of the wrong shape (a node that is not an object, a missing key)
exits 2 as malformed input. The same sources also
compile to JavaScript (`dafny build --target:js Check.dfy --no-verify -o build/js/bluffcore`
succeeds; the JS build was not exercised further).

## Results (2026-11-20, Apple Silicon, all from the repo root)

- Verification: `dafny verify Model.dfy BestResponse.dfy Check.dfy --cores 2` reports
  135 verified, 0 errors, in about 7.7 s wall (`/usr/bin/time -p dafny/attempt-b/build.sh`).
  Stable under 10 solver seeds (`--boogie /randomSeed:0` through `:9`, all 135 verified).
- Size: 1113 lines of Dafny (Model 235, BestResponse 743, Check 135), 943 non-blank,
  non-comment lines.
- Kuhn reference equilibrium: `value_p0 = -1/18`, `exploitability = 0`.
- Leduc LP profile (9451 nodes, the largest game tried): `exploitability = 0`,
  `value_p0 = -1454920850547486749701863871533/16995463438839962469905132501930` (about
  -0.0856), 0.46 s wall for the whole `check.py` run.
- Differential vs `reference/br.py`: identical `value_p0`, both br values, `nash_conv` and
  `exploitability` on 49 of 49 runs: 17 game files (games/out, games/out/other and
  games/out/redblack), each with every shipped profile (lp, ref, eq, cfr) plus one seeded random
  profile. Every shipped profile has exploitability exactly 0.
