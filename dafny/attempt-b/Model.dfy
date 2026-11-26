// Model.dfy: finite two-player zero-sum extensive-form games with chance, stored as a flat
// node table that mirrors the JSON game file (nodes indexed by id, each with a child list).
// Recursion is justified by an explicit height array: every child has a smaller height than its
// parent, which the runtime-checked predicate WF requires. Sharing (a DAG) is allowed.
// All arithmetic is over Dafny's real, i.e. exact rationals at run time.

module GameModel {
  datatype Node =
    | Terminal(payoff: real)                               // payoff to player 0
    | Chance(probs: seq<real>, kids: seq<nat>)
    | Decision(player: nat, infoset: nat, kids: seq<nat>)  // kids[a] is the child after action a

  datatype Game = Game(
    nodes: seq<Node>,
    root: nat,
    height: seq<nat>,            // height[child] < height[parent]: the well-founded measure
    infosetPlayer: seq<nat>,     // owner of each infoset id
    infosetActions: seq<nat>)    // number of actions at each infoset id

  // A behavioral strategy (or a profile, since infoset ids are global): one row per infoset id,
  // one probability per action. Rows of the other player's infosets are ignored.
  type Strat = seq<seq<real>>

  function Sum(s: seq<real>): real
  {
    if |s| == 0 then 0.0 else s[0] + Sum(s[1..])
  }

  function Opp(p: nat): nat { if p == 0 then 1 else 0 }

  // Utility of a terminal payoff (to player 0) for player p.
  function U(p: nat, u: real): real { if p == 0 then u else -u }

  predicate Shape(g: Game)
  {
    |g.height| == |g.nodes| && g.root < |g.nodes| && |g.infosetPlayer| == |g.infosetActions|
  }

  opaque predicate KidsOK(g: Game, n: nat, kids: seq<nat>)
    requires Shape(g) && n < |g.nodes|
  {
    forall i | 0 <= i < |kids| :: kids[i] < |g.nodes| && g.height[kids[i]] < g.height[n]
  }

  predicate NodeOK(g: Game, n: nat)
    requires Shape(g) && n < |g.nodes|
  {
    match g.nodes[n]
    case Terminal(_) => true
    case Chance(pr, kids) =>
      0 < |kids| && |pr| == |kids| && KidsOK(g, n, kids)
      && (forall i | 0 <= i < |pr| :: pr[i] >= 0.0) && Sum(pr) == 1.0
    case Decision(q, I, kids) =>
      q <= 1 && I < |g.infosetPlayer| && g.infosetPlayer[I] == q
      && |kids| == g.infosetActions[I] && KidsOK(g, n, kids)
  }

  // Well-formedness of a game. Checked at run time by Check (it is compiled).
  predicate WF(g: Game)
  {
    Shape(g) && InfosetsOK(g) && NodesOK(g)
  }

  opaque predicate InfosetsOK(g: Game)
    requires Shape(g)
  {
    forall I | 0 <= I < |g.infosetPlayer| :: g.infosetPlayer[I] <= 1 && g.infosetActions[I] > 0
  }

  opaque predicate NodesOK(g: Game)
    requires Shape(g)
  {
    forall n | 0 <= n < |g.nodes| :: NodeOK(g, n)
  }

  lemma WFNode(g: Game, n: nat)
    requires WF(g) && n < |g.nodes|
    ensures NodeOK(g, n)
  {
    reveal NodesOK();
  }

  lemma WFInfoset(g: Game, I: nat)
    requires WF(g) && I < |g.infosetPlayer|
    ensures g.infosetPlayer[I] <= 1 && g.infosetActions[I] > 0
  {
    reveal InfosetsOK();
  }

  // b has a row of the right length at every infoset of player p.
  predicate Fits(g: Game, p: nat, b: Strat)
    requires Shape(g)
  {
    |b| == |g.infosetPlayer|
    && forall I | 0 <= I < |b| && g.infosetPlayer[I] == p :: |b[I]| == g.infosetActions[I]
  }

  predicate IsDist(r: seq<real>)
  {
    (forall a | 0 <= a < |r| :: r[a] >= 0.0) && Sum(r) == 1.0
  }

  // b is a behavioral strategy for player p: a probability distribution at each of p's infosets.
  predicate ValidBeh(g: Game, p: nat, b: Strat)
    requires Shape(g)
  {
    Fits(g, p, b) && forall I | 0 <= I < |b| && g.infosetPlayer[I] == p :: IsDist(b[I])
  }

  predicate Inv(g: Game, p: nat, beta: Strat, sigma: Strat)
  {
    WF(g) && p <= 1 && Fits(g, p, beta) && Fits(g, Opp(p), sigma)
  }

  lemma KidFacts(g: Game, n: nat, i: nat)
    requires WF(g) && n < |g.nodes| && !g.nodes[n].Terminal? && i < |g.nodes[n].kids|
    ensures g.nodes[n].kids[i] < |g.nodes| && g.height[g.nodes[n].kids[i]] < g.height[n]
  {
    WFNode(g, n);
    reveal KidsOK();
  }

  // The weights used at a non-terminal node: chance probabilities, or the acting player's row
  // (beta for player p, sigma for the opponent).
  function Row(g: Game, p: nat, beta: Strat, sigma: Strat, n: nat): seq<real>
    requires Inv(g, p, beta, sigma) && n < |g.nodes| && !g.nodes[n].Terminal?
    ensures |Row(g, p, beta, sigma, n)| == |g.nodes[n].kids|
  {
    WFNode(g, n);
    match g.nodes[n]
    case Chance(pr, _) => pr
    case Decision(q, I, _) => if q == p then beta[I] else sigma[I]
  }

  // Val(g, p, beta, sigma, n): expected utility to player p at node n when p plays beta and the
  // opponent plays sigma.
  function Val(g: Game, p: nat, beta: Strat, sigma: Strat, n: nat): real
    requires Inv(g, p, beta, sigma) && n < |g.nodes|
    decreases g.height[n], 1, 0
  {
    if g.nodes[n].Terminal? then U(p, g.nodes[n].payoff)
    else SumKids(g, p, beta, sigma, n, Row(g, p, beta, sigma, n), 0)
  }

  // sum over i <= j < |ws| of ws[j] * Val(kids[j])
  function SumKids(g: Game, p: nat, beta: Strat, sigma: Strat, n: nat, ws: seq<real>, i: nat): real
    requires Inv(g, p, beta, sigma) && n < |g.nodes| && !g.nodes[n].Terminal?
    requires |ws| == |g.nodes[n].kids| && i <= |ws|
    decreases g.height[n], 0, |ws| - i
  {
    if i == |ws| then 0.0
    else
      KidFacts(g, n, i);
      ws[i] * Val(g, p, beta, sigma, g.nodes[n].kids[i]) + SumKids(g, p, beta, sigma, n, ws, i + 1)
  }

  // Zero sum: under one profile s, player 1's expected utility is minus player 0's.
  lemma ZeroSum(g: Game, s: Strat, n: nat)
    requires Inv(g, 0, s, s) && Inv(g, 1, s, s) && n < |g.nodes|
    ensures Val(g, 1, s, s, n) == -Val(g, 0, s, s, n)
    decreases g.height[n], 1, 0
  {
    if !g.nodes[n].Terminal? {
      assert Row(g, 1, s, s, n) == Row(g, 0, s, s, n);
      ZeroSumKids(g, s, n, Row(g, 0, s, s, n), 0);
    }
  }

  lemma ZeroSumKids(g: Game, s: Strat, n: nat, ws: seq<real>, i: nat)
    requires Inv(g, 0, s, s) && Inv(g, 1, s, s) && n < |g.nodes| && !g.nodes[n].Terminal?
    requires |ws| == |g.nodes[n].kids| && i <= |ws|
    ensures SumKids(g, 1, s, s, n, ws, i) == -SumKids(g, 0, s, s, n, ws, i)
    decreases g.height[n], 0, |ws| - i
  {
    if i < |ws| {
      KidFacts(g, n, i);
      ZeroSum(g, s, g.nodes[n].kids[i]);
      ZeroSumKids(g, s, n, ws, i + 1);
    }
  }

  // Pure strategies: one action index per infoset of p.
  predicate PureValid(g: Game, p: nat, pi: seq<nat>)
    requires Shape(g)
  {
    |pi| == |g.infosetPlayer|
    && forall I | 0 <= I < |pi| && g.infosetPlayer[I] == p :: pi[I] < g.infosetActions[I]
  }

  function OneHotRow(k: nat, c: nat): seq<real>
  {
    seq(k, a => if a == c then 1.0 else 0.0)
  }

  // The behavioral strategy that plays pi with probability one.
  function OneHot(g: Game, pi: seq<nat>): Strat
    requires Shape(g) && |pi| == |g.infosetPlayer|
    ensures |OneHot(g, pi)| == |pi|
    ensures forall I | 0 <= I < |pi| :: |OneHot(g, pi)[I]| == g.infosetActions[I]
  {
    seq(|pi|, I requires 0 <= I < |pi| => OneHotRow(g.infosetActions[I], pi[I]))
  }

  lemma SumZeros(s: seq<real>)
    requires forall i | 0 <= i < |s| :: s[i] == 0.0
    ensures Sum(s) == 0.0
  {
    if |s| > 0 { SumZeros(s[1..]); }
  }

  lemma OneHotRowDist(k: nat, c: nat)
    requires c < k
    ensures IsDist(OneHotRow(k, c))
    decreases c
  {
    var r := OneHotRow(k, c);
    if c == 0 {
      SumZeros(r[1..]);
    } else {
      assert r[1..] == OneHotRow(k - 1, c - 1);
      OneHotRowDist(k - 1, c - 1);
    }
  }

  lemma OneHotValid(g: Game, p: nat, pi: seq<nat>)
    requires Shape(g) && PureValid(g, p, pi)
    ensures ValidBeh(g, p, OneHot(g, pi))
  {
    forall I | 0 <= I < |pi| && g.infosetPlayer[I] == p
      ensures IsDist(OneHot(g, pi)[I])
    {
      OneHotRowDist(g.infosetActions[I], pi[I]);
    }
  }
}
