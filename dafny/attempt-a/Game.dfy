// The game model: finite two-player zero-sum extensive-form games with chance,
// as an inductive tree, plus the semantics (expected utility) of behavioral and
// pure strategies.

include "Math.dfy"

module Game {
  import opened Math

  // A game tree. Payoffs are to player 0; player 1 gets the negation.
  // At a decision node, action a leads to kids[a]. Infosets are numbered 0..|A|-1.
  datatype Node =
    | Terminal(payoff: real)
    | Chance(probs: seq<real>, kids: seq<Node>)
    | Decision(player: nat, infoset: nat, kids: seq<Node>)

  // A behavioral strategy (or profile): row I is the distribution over the
  // actions of infoset I.
  type Strat = seq<seq<real>>

  function Row(m: Strat, I: nat): seq<real>
  {
    if I < |m| then m[I] else []
  }

  // Utility to player p of a payoff to player 0.
  function U(p: nat, x: real): real
  {
    if p == 0 then x else -x
  }

  // Expected utility to player p when p plays behavioral strategy b at p's own
  // decision nodes and every other decision node follows sigma.
  // The profile's own value to p is Val(h, p, sigma, sigma).
  function Val(h: Node, p: nat, b: Strat, sigma: Strat): real
    decreases h
  {
    match h
    case Terminal(x) => U(p, x)
    case Chance(ps, ks) =>
      Dot(ps, seq(|ks|, i requires 0 <= i < |ks| => Val(ks[i], p, b, sigma)))
    case Decision(q, I, ks) =>
      Dot(if q == p then Row(b, I) else Row(sigma, I),
          seq(|ks|, i requires 0 <= i < |ks| => Val(ks[i], p, b, sigma)))
  }

  // Expected utility to player p when p plays the pure strategy s (infoset to
  // action index) and every other decision node follows sigma.
  ghost function PVal(h: Node, p: nat, s: nat -> nat, sigma: Strat): real
    decreases h
  {
    match h
    case Terminal(x) => U(p, x)
    case Chance(ps, ks) =>
      Dot(ps, seq(|ks|, i requires 0 <= i < |ks| => PVal(ks[i], p, s, sigma)))
    case Decision(q, I, ks) =>
      if q == p then
        (if s(I) < |ks| then PVal(ks[s(I)], p, s, sigma) else 0.0)
      else
        Dot(Row(sigma, I), seq(|ks|, i requires 0 <= i < |ks| => PVal(ks[i], p, s, sigma)))
  }

  // Number of nodes in a tree (the termination measure).
  ghost function Size(h: Node): nat
    decreases h
  {
    match h
    case Terminal(_) => 1
    case Chance(_, ks) => 1 + SumNF(seq(|ks|, i requires 0 <= i < |ks| => Size(ks[i])), 0)
    case Decision(_, _, ks) => 1 + SumNF(seq(|ks|, i requires 0 <= i < |ks| => Size(ks[i])), 0)
  }

  // Structural well-formedness against an arity table A (A[I] = number of
  // actions at infoset I): every infoset is in range, every node of an
  // infoset has exactly A[I] >= 1 children, chance nodes have one probability
  // per child and at least one child.
  predicate WFN(h: Node, A: seq<nat>)
    decreases h
  {
    match h
    case Terminal(_) => true
    case Chance(ps, ks) =>
      |ks| >= 1 && |ps| == |ks| && forall i :: 0 <= i < |ks| ==> WFN(ks[i], A)
    case Decision(q, I, ks) =>
      I < |A| && |ks| == A[I] && |ks| >= 1 && forall i :: 0 <= i < |ks| ==> WFN(ks[i], A)
  }

  // Semantic sanity (checked at runtime, not needed by the best response
  // theorems): players are 0 or 1 and chance distributions are exact.
  predicate Sane(h: Node)
    decreases h
  {
    match h
    case Terminal(_) => true
    case Chance(ps, ks) =>
      ValidRow(ps, |ks|) && forall i :: 0 <= i < |ks| ==> Sane(ks[i])
    case Decision(q, _, ks) =>
      q <= 1 && forall i :: 0 <= i < |ks| ==> Sane(ks[i])
  }

  predicate ValidRow(r: seq<real>, n: nat)
  {
    |r| == n && (forall j :: 0 <= j < n ==> r[j] >= 0.0) && Sum(r) == 1.0
  }

  // A behavioral strategy: an exact probability distribution at every infoset.
  predicate ValidBeh(b: Strat, A: seq<nat>)
  {
    |b| == |A| && forall I :: 0 <= I < |A| ==> ValidRow(b[I], A[I])
  }

  // A pure strategy: a legal action at every infoset.
  ghost predicate ValidPure(s: nat -> nat, A: seq<nat>)
  {
    forall I :: 0 <= I < |A| ==> s(I) < A[I]
  }

  function OneHotRow(k: nat, n: nat): seq<real>
  {
    seq(n, a requires 0 <= a < n => if a == k then 1.0 else 0.0)
  }

  // The behavioral strategy that plays the pure strategy s.
  ghost function OneHot(s: nat -> nat, A: seq<nat>): Strat
  {
    seq(|A|, I requires 0 <= I < |A| => OneHotRow(s(I), A[I]))
  }

  lemma OneHotValid(s: nat -> nat, A: seq<nat>)
    requires ValidPure(s, A)
    ensures ValidBeh(OneHot(s, A), A)
  {
    var b := OneHot(s, A);
    forall I | 0 <= I < |A| ensures ValidRow(b[I], A[I]) {
      SumOneHot(b[I], s(I), 0);
    }
  }

  // Pure strategy semantics agree with the one-hot behavioral semantics.
  lemma PureIsOneHot(h: Node, p: nat, s: nat -> nat, sigma: Strat, A: seq<nat>)
    requires WFN(h, A) && ValidPure(s, A)
    ensures PVal(h, p, s, sigma) == Val(h, p, OneHot(s, A), sigma)
    decreases h
  {
    var b := OneHot(s, A);
    match h
    case Terminal(_) =>
    case Chance(ps, ks) =>
      var pv := seq(|ks|, i requires 0 <= i < |ks| => PVal(ks[i], p, s, sigma));
      var vv := seq(|ks|, i requires 0 <= i < |ks| => Val(ks[i], p, b, sigma));
      forall i | 0 <= i < |ks| ensures pv[i] == vv[i] {
        PureIsOneHot(ks[i], p, s, sigma, A);
      }
      assert pv == vv;
    case Decision(q, I, ks) =>
      var pv := seq(|ks|, i requires 0 <= i < |ks| => PVal(ks[i], p, s, sigma));
      var vv := seq(|ks|, i requires 0 <= i < |ks| => Val(ks[i], p, b, sigma));
      forall i | 0 <= i < |ks| ensures pv[i] == vv[i] {
        PureIsOneHot(ks[i], p, s, sigma, A);
      }
      assert pv == vv;
      if q == p {
        assert Row(b, I) == OneHotRow(s(I), A[I]);
        DotOneHot(Row(b, I), vv, s(I), 0);
      }
  }

  // Zero sum: under one profile, player 1's utility is minus player 0's.
  lemma ZeroSum(h: Node, sigma: Strat)
    ensures Val(h, 1, sigma, sigma) == -Val(h, 0, sigma, sigma)
    decreases h
  {
    match h
    case Terminal(_) =>
    case Chance(ps, ks) =>
      var v0 := seq(|ks|, i requires 0 <= i < |ks| => Val(ks[i], 0, sigma, sigma));
      var v1 := seq(|ks|, i requires 0 <= i < |ks| => Val(ks[i], 1, sigma, sigma));
      forall i | 0 <= i < |ks| ensures v1[i] == -v0[i] {
        ZeroSum(ks[i], sigma);
      }
      DotNeg(ps, v0, v1);
    case Decision(q, I, ks) =>
      var v0 := seq(|ks|, i requires 0 <= i < |ks| => Val(ks[i], 0, sigma, sigma));
      var v1 := seq(|ks|, i requires 0 <= i < |ks| => Val(ks[i], 1, sigma, sigma));
      forall i | 0 <= i < |ks| ensures v1[i] == -v0[i] {
        ZeroSum(ks[i], sigma);
      }
      DotNeg(Row(sigma, I), v0, v1);
  }

  lemma SizePos(h: Node)
    ensures Size(h) >= 1
  {
  }

  // Each child is strictly smaller than its parent.
  lemma KidSmaller(h: Node, a: nat)
    requires !h.Terminal? && a < |h.kids|
    ensures Size(h.kids[a]) < Size(h)
  {
    var ks := h.kids;
    var sz := seq(|ks|, i requires 0 <= i < |ks| => Size(ks[i]));
    SumNGe(sz, 0, a);
  }
}
