# Mini Red/Black

Small heads-up variants of Red/Black (Noah's hidden-information card game, rules in
`~/coding/red_black/RULES.md`) as CONTRACTS.md game files, plus a differential test of every file
against the real engine. The point is spec fidelity: lessons.md #1 says "the implementation
defines the game, not the intent", and a game file is an implementation.

| File | What |
| --- | --- |
| `model.py` | the rules, written from RULES.md without importing the engine |
| `generate.py` | writes `games/out/redblack/<name>.game.json` |
| `difftest.py` | walks a written file in lockstep with the engine mirror and checks it |
| `solve.py` | CFR+ plus an exact (Fraction) best response, as an independent equilibrium certificate |
| `test_redblack.py` | pins all of the below (`uv run pytest games/redblack -q`, 22 tests) |

Utility: win = +1, loss = -1 to seat 0 (the research's convention). A value `v` means seat 0
wins with probability `(1 + v) / 2`, so the 25% gate class is `v = -1/2`.

## Variants

All sizes from `uv run games/redblack/generate.py` (or `--sizes NAME`). Values: "LP" is
`reference/lp.py` (sequence-form LP, exactly certified); "CFR" is `solve.py` (CFR+, default 4000
iterations; each `*.cfr.strategy.json` names its own count in `producer`, 2000 for most, rounded to
small rationals, exact NashConv 0). Both were then re-checked with
`reference/br.py` (exploitability exactly 0).

| Name | Knobs changed from the real game | Nodes | Infosets p0/p1 | Value to seat 0 | P(seat 0 wins) | Certified by |
| --- | --- | --- | --- | --- | --- | --- |
| `rb-h1` | 1 card each (real: 5) | 85 | 10 / 8 | 1/51 | 26/51 | LP, CFR, closed form |
| `rb-pennies-inner` | fixed public-color position, seat 1 to discard | 43 | 5 / 9 | 0 | 1/2 | LP, CFR, research |
| `rb-pennies` | fixed public-color position (the gate class) | 1,551 | 167 / 333 | **-1/2** | **1/4** | LP, CFR, research |
| `rb-pennies-leaky` | rb-pennies with a DELIBERATELY WRONG discard rule | 1,551 | 317 / 333 | 0 | 1/2 | LP, CFR |
| `rb-21` | seat 0 dealt 2, seat 1 dealt 1 | 4,707 | 775 / 432 | 9/17 | 13/17 | LP, CFR |
| `rb-12` | seat 0 dealt 1, seat 1 dealt 2 | 4,707 | 434 / 771 | -173/442 | 269/884 | LP |
| `rb-31` | seat 0 dealt 3, seat 1 dealt 1 | 984,661 | 190,862 / 48,324 | unknown (LP timed out at 40 min) | | none yet |
| `rb-pos-2v2` | fixed position, both hold red+black | 1,241,543 | 144,273 / 144,271 | | | |
| `rb-h2-d4` | 2 each from a 4-card deck (2R 2B) | 1,312,266 | 160,187 / 160,183 | | | |
| `rb-h2` | 2 cards each | 2,344,568 | 249,861 / 249,857 | | | |

rb-31 is the medium rung and fully engine-checked, but its value is open: `reference/lp.py` on it
ran 40 minutes (2.4 GB resident) without finishing and was stopped by a timeout, and Python CFR+
over a million nodes is too slow for this deadline. The last four are not written by default: `rb-31` alone is 137 MB of JSON, over GitHub's 100 MB
file limit. Name them to generate them (`generate.py rb-31`); `difftest.py NAME --in-memory`
tests them without a file.

Unchanged from the real game, in every variant: the deck is 26 red / 26 black (except
`rb-h2-d4`, and the fixed positions, which deal nothing), with undealt cards never seen; a reorder
opens every round; the bid ladder is locked to the opening color with strictly rising counts, capped
at the cards in play; challenge is legal on any turn after the first bid and never before it; the
bidder flips any non-empty deck top-down, the first off-color flip loses and reaching the count
wins; the loser discards one card in secret (forced and eliminating at one card); the
procurement winner opens the next round; the last player holding cards wins.

The fixed positions (`rb-pennies*`, `rb-pos-2v2`) start with every card color public, as the
research's gate classes do (`EntryClass2(-1, SeatStat(1,1,0), SeatStat(2,1,1))`: seat 0 has one
card, certainly red; seat 1 two cards, certainly one red and one black). Hand order is private.

### What is abstracted away, and why it changes nothing

- **The clock, forfeits, resignations, ratings, variants (Creative Duel), hidden hands.** None is in
  the game-theoretic game. Hidden hands is a display rule (RULES 6b): a seat is still sent its own
  colors, and a player is modeled with perfect recall.
- **Reorders are labeled by color sequence, not by card permutation.** Permutations giving the same
  colors lead to identical subtrees and identical opponent views (RULES 6a aliasing). `difftest.py
  --reps all` explores every permutation and finds no infoset that tells them apart.
- **Discards are labeled by color, not by card.** The next round always opens with a reorder, and
  the opponent sees only that the hand shrank, whichever card left (RULES 4d, 6a). Same check.
- **The reorder is simultaneous; the file orders it seat 0 then seat 1**, with neither seeing the
  other's choice. Standard, and the engine's only trace of the order (`reorderReady`) is timing.
- **Forced moves are skipped** (a reorder of one card or of all one color, a one-color discard):
  one action, no choice, no information.
- **The deal is by color counts.** Round 1 opens with a reorder of the whole hand, so the dealt
  order never matters. The probabilities are the exact hypergeometric; see the deal test below.

## The 25% gate endgame (lessons.md #3)

`rb-pennies`: seat 0 holds a known red and opens; seat 1 holds a known red and a known black.
Its value is exactly -1/2, i.e. seat 0 wins 25%, the research's figure (`research/redblack-cfrplus.tex`
line 249, `docs/solver/solver.md` "Why the boundary sits here"), certified three ways:

1. `uv run reference/lp.py games/out/redblack/rb-pennies.game.json`: certified_exact, -1/2.
2. `uv run games/redblack/solve.py games/out/redblack/rb-pennies.game.json`: CFR+ rounded to
   denominators up to 4, exact NashConv 0, value -1/2.
3. By hand, the two-stage matching pennies. Stage 1: seat 1 secretly puts red or black on top.
   Seat 0's opening `2R` wins the round exactly when the top is red (seat 1 must challenge, and seat 0
   flips its own red and seat 1's top). `1R` or `1B` wins exactly when the top is black (seat 1 cannot
   produce `2R` through a black top, and `1B` is good only off a black top). Everything else loses.
   Losing the round eliminates seat 0. Stage 2 (`rb-pennies-inner`, value 0): seat 1, down to one
   card, secretly keeps red or black. Seat 0 opens `2R` (wins against a kept red) or `1R`/`1B` (wins
   against a kept black). Each stage is an even matching pennies, so seat 0 wins 1/2 x 1/2 = 1/4.

## Spec traps, reproduced

**Lesson 1, discard identity must not leak.** `rb-pennies-leaky` is rb-pennies with one rule
changed: the opponent learns which slot a discard came from, which is what stable card ids gave
the v0 solver. Its value is exactly 0: seat 0 now wins **50%**, not 25%, because after the leak it
always knows the card seat 1 kept. Both the LP and our CFR certify it with exploitability exactly
0. **The checker is right about the wrong game.** Only the differential test sees the problem: it
reports 150 distinct pairs of seat-0 infosets that the engine's projection cannot tell apart
(exactly the 317 - 167 extra seat-0 infosets the leak creates), e.g.

    p0|start:R|RB r1:s0 ro:R 0:2B 1:3B 0:ch f1R L1 d1@1 r2:s0 ro:R
    p0|start:R|RB r1:s0 ro:R 0:2B 1:3B 0:ch f1R L1 d1@0 r2:s0 ro:R

(`uv run games/redblack/difftest.py rb-pennies-leaky --reps all`, exit 1).

**Lesson 3, round-boundary placement.** That was a solver decomposition error (cutting the round
after the discard priced this position at 15.5%), not a game-file error, so a full game tree cannot
reproduce it: the tree has no boundaries. What the tree gives is the exact truth (25%) that the
research's oracle had to bracket, from a file the engine checks node by node.

## Fidelity test results

`difftest.py` walks the game FILE (not model.py) in lockstep with the engine mirror
(`~/coding/red_black/simulator/src/redblack/engine`, imported read-only, real `apply_action` on
every step) and checks, at every node: the engine's legal actions for the acting seat map exactly
onto the node's labels; the acting seat matches; terminals match the engine's winner; and, per
player, infoset to engine-observation-history is a bijection. The observation history is the
sequence of `project_for_recipient` views (the exact client payload), canonicalized only by
writing the player's own card ids as "own", sorting the own hand before its reorder is confirmed,
and dropping `version` and `reorderReady`. One infoset with two engine histories would be a merge
(the file makes a player forget); two infosets with one engine history would be a leak.

| Run | Visits | Engine steps | Infosets checked | Result |
| --- | --- | --- | --- | --- |
| rb-h1, `--reps all` | 85 | 120 | 18 | pass |
| rb-pennies-inner, `--reps all` | 43 | 62 | 14 | pass |
| rb-pennies, `--reps all` | 1,551 | 2,267 | 500 | pass |
| rb-21, `--reps all` | 9,167 | 13,640 | 1,207 | pass |
| rb-12, `--reps all` | 9,167 | 13,634 | 1,205 | pass |
| rb-31, `--in-memory --reps random` | 984,661 | 1,456,708 | 239,186 | pass (59 s) |
| rb-pos-2v2, `--in-memory --reps random` | 1,241,543 | 1,815,606 | 288,544 | pass (71 s) |
| rb-pennies-leaky, `--reps all` | 1,551 | 2,267 | 650 | **fail: 20 leaks** (intended) |

Every file is walked exhaustively (all nodes). Visits exceed nodes under `--reps all` because a
label covering several engine actions is walked once per action.

**Does the test have teeth?** It passed every real file on the first run, which is a reason to
distrust it, so `model.MUTANTS` seeds seven plausible spec bugs and `difftest.py --mutants` requires
each to fail:

Counts are distinct leaked infoset pairs, distinct merged infosets, or error nodes; rb-21 under
`--mutants --reps all`, rb-pos-2v2 under `--mutants rb-pos-2v2 --reps first` (5 min 15 s).

| Mutant | rb-21 | rb-pos-2v2 |
| --- | --- | --- |
| leak-reorder: opponent sees the chosen order | caught (914 leaks) | caught (272,799 leaks) |
| forget-flips: challenger does not record flips | caught (112 merges) | caught (52,872 merges) |
| bid-cap-off-by-one: no bid may name every card | caught (12 errors) | caught (4 errors) |
| starter-is-loser: loser opens the next round | caught (408 errors) | caught (1,600 errors) |
| color-switch: a raise may change color | caught (240 errors) | caught (2,712 errors) |
| reorder-round-1-only: Locked Order's rule | **missed** | caught (800 errors) |
| open-discard: opponent learns a discard's color | caught (300 leaks) | caught (154,700 leaks) |

The miss is honest and instructive: in rb-21 no hand holds two cards after round 1, so a game
without later reorders is the same game. **A differential test only tests the game you
generated**; the mutant needed a position (rb-pos-2v2) where it matters.

**The deal.** `uv run games/redblack/difftest.py rb-h1 --deal-samples 200000` deals 200,000 games
through the engine's own `deal_initial_hands` (the shuffle real games use) per hand size and runs a
chi-square against the file's exact hypergeometric: hand size 1 chi2 6.97 on 3 dof (p 0.073), 2
gives 7.55 on 8 (p 0.48), 3 gives 13.21 on 15 (p 0.59), and no impossible cell appeared.

## Discrepancies found, and how each was resolved

1. **Generator vs engine: none.** No legal-action, turn, payoff or information discrepancy on any
   real file. The mutant table is the evidence that this is not a blind test.
2. **RULES.md vs the engine (not fixed: the engine repo is read-only).** RULES 6 says flips "are
   hidden again at the start of the next reorder ... the engine simply stops surfacing it". The
   projection does keep surfacing them: in round 2's reorder, `lastRoundOutcome.flips` still carries
   round 1's flip colors (under synthetic ids `o{seat}.{k}`). Reproduced by playing rb-pennies to
   round 2 through `apply_action` and printing `project_for_recipient(state, "u0").lastRoundOutcome`.
   It changes no value for a perfect-recall player (they saw those flips), so the game files are
   unaffected; it is a doc/engine mismatch for Noah to settle.
3. **A bug in our own CFR+ (solve.py), caught by the exact best response.** The first version pruned
   opponent branches whose current probability was 0. The average strategy is weighted by the
   player's OWN reach, so it must still be accumulated under such a branch. On rb-pennies-leaky,
   seat 0 learned the exploit in iteration 1 (its current strategy was right), but its average kept
   the iteration-1 uniform rows, and the average profile was exploitable by 0.747 (seat 1's best
   response) after 2000 iterations, unchanged at 20000. Rounding could never certify it. Fixed by not
   pruning and by applying regret updates after each full traversal; all five small files then
   certify. This is the project's thesis in miniature: the solver "converged" and looked fine, and
   only the exact best response said otherwise.
4. **Strategy file naming.** `reference/lp.py` writes `*.lp-strategy.json`, which a
   `*.strategy.json` glob misses; the files here are renamed `*.lp.strategy.json` and
   `*.cfr.strategy.json`.

5. **A blind spot in difftest itself, found in review.** The view canonicalization first mapped
   every card id starting with "c" to "own", and every raw engine id starts with "c", opponents'
   included. So an engine that shipped stable opponent card ids (lesson 1 from the engine side)
   passed the walk with 0 merges. Ids are now canonicalized by owner (flip `fromSeat`, discard
   `loserSeat`, hand seat), and a monkeypatched id-leaking engine yields 161 merged infosets on
   rb-pennies (`test_an_engine_that_ships_stable_card_ids_is_caught`). The review also made the walk
   check the file's deal probabilities against the exact hypergeometric (before, only model.py's
   distribution was compared with the engine's shuffle, never the file's numbers), made the leak and
   merge counts count every distinct pair instead of stopping at the 20-example print cap (so the
   leaky gate reports 150, not 20), and added a seventh mutant, open-discard (the opponent learns a
   discard's color), caught as a leak on rb-21, rb-12 and rb-pennies.
6. **RULES.md contradicts itself on #2.** Section 6's "Always public" list includes the most recent
   round's full flip sequence; its "Reset between rounds" list says those flips are hidden at the next
   reorder. The engine follows the first, so #2 is a RULES wording conflict rather than an engine bug.

## Reproduce

    uv run games/redblack/generate.py                          # the default files
    uv run games/redblack/difftest.py --reps all               # exhaustive, small files
    uv run games/redblack/difftest.py rb-31 --in-memory        # the medium rung
    uv run games/redblack/difftest.py --mutants --reps all     # seeded bugs on rb-21, rb-12
    uv run games/redblack/difftest.py --mutants rb-pos-2v2 --reps first   # about 5 minutes
    uv run games/redblack/solve.py games/out/redblack/rb-pennies.game.json
    uv run reference/lp.py games/out/redblack/rb-pennies.game.json
    uv run pytest games/redblack -q
