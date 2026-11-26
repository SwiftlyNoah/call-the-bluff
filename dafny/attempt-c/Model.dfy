// Model.dfy: finite two-player zero-sum extensive-form games with chance,
// behavioral strategy profiles and expected utility, over exact reals.
//
// Infosets are numbered 0 .. |owner|-1. Actions at an infoset are numbered
// 0 .. nAct[I]-1, and action i at a decision node leads to kids[i].
// A profile is one row of action probabilities per infoset (both players).

module Model {

  datatype Node =
    | Leaf(payoff: real)                              // payoff to player 0
    | Chance(probs: seq<real>, kids: seq<Node>)       // probs[i] leads to kids[i]
    | Dec(player: nat, inf: nat, kids: seq<Node>)     // action i leads to kids[i]

  datatype Game = Game(root: Node, owner: seq<nat>, nAct: seq<nat>)

  type Profile = seq<seq<real>>

  // Total accessor: a missing entry reads as probability 0.
  function Get(s: Profile, i: nat, a: nat): real
  {
    if i < |s| && a < |s[i]| then s[i][a] else 0.0
  }

  // Utility of player p (0 or 1) when player 0 gets x (zero-sum).
  function Util(p: nat, x: real): real
  {
    if p == 0 then x else -x
  }

  // Weight on the i-th child of a non-leaf node under profile s.
  function Coef(n: Node, s: Profile, i: nat): real
    requires !n.Leaf?
  {
    if n.Chance? then (if i < |n.probs| then n.probs[i] else 0.0)
    else Get(s, n.inf, i)
  }

  // Expected utility to player p of the subtree n when everyone plays s.
  function EU(n: Node, s: Profile, p: nat): real
    decreases n, 1
  {
    if n.Leaf? then Util(p, n.payoff) else EUKids(n, s, p, 0)
  }

  function EUKids(n: Node, s: Profile, p: nat, i: nat): real
    requires !n.Leaf?
    decreases n, 0, |n.kids| - i
  {
    if i >= |n.kids| then 0.0
    else Coef(n, s, i) * EU(n.kids[i], s, p) + EUKids(n, s, p, i + 1)
  }

  // ---------- well-formedness ----------

  predicate WFNode(n: Node, owner: seq<nat>, nAct: seq<nat>)
    decreases n
  {
    match n
    case Leaf(_) => true
    case Chance(probs, kids) =>
      |probs| == |kids| && |kids| >= 1 &&
      forall i :: 0 <= i < |kids| ==> WFNode(kids[i], owner, nAct)
    case Dec(pl, inf, kids) =>
      inf < |owner| && inf < |nAct| && owner[inf] == pl && pl <= 1 &&
      |kids| == nAct[inf] && |kids| >= 1 &&
      forall i :: 0 <= i < |kids| ==> WFNode(kids[i], owner, nAct)
  }

  predicate WF(g: Game)
  {
    |g.owner| == |g.nAct| &&
    (forall I :: 0 <= I < |g.nAct| ==> g.nAct[I] >= 1 && g.owner[I] <= 1) &&
    WFNode(g.root, g.owner, g.nAct)
  }

  // Sum of r[k..].
  function RowSumFrom(r: seq<real>, k: nat): real
    decreases |r| - k
  {
    if k >= |r| then 0.0 else r[k] + RowSumFrom(r, k + 1)
  }

  predicate ValidRow(r: seq<real>, n: nat)
  {
    |r| == n && (forall a :: 0 <= a < n ==> r[a] >= 0.0) && RowSumFrom(r, 0) == 1.0
  }

  // A behavioral strategy profile: every infoset row is a probability distribution.
  predicate ValidProfile(g: Game, s: Profile)
  {
    |s| == |g.nAct| && forall I :: 0 <= I < |s| ==> ValidRow(s[I], g.nAct[I])
  }

  // t plays exactly like s at every infoset of the opponent of p.
  ghost predicate AgreeOpp(g: Game, s: Profile, t: Profile, p: nat)
  {
    forall I: nat, a: nat :: I < |g.owner| && g.owner[I] != p ==> Get(s, I, a) == Get(t, I, a)
  }

  // ---------- pure strategies ----------

  // A pure strategy picks one action index per infoset (only p's entries matter).
  predicate PureOk(g: Game, pi: seq<nat>)
  {
    |pi| == |g.nAct| && forall I :: 0 <= I < |pi| ==> pi[I] < g.nAct[I]
  }

  function PureRow(a: nat, n: nat): seq<real>
  {
    seq(n, j => if j == a then 1.0 else 0.0)
  }

  // The profile in which p plays the pure strategy pi and the opponent plays s.
  function Override(g: Game, s: Profile, pi: seq<nat>, p: nat): Profile
    requires |pi| == |g.nAct| && |s| == |g.nAct| && |g.owner| == |g.nAct|
  {
    seq(|g.nAct|, I requires 0 <= I < |g.nAct| =>
      if g.owner[I] == p then PureRow(pi[I], g.nAct[I]) else s[I])
  }
}
