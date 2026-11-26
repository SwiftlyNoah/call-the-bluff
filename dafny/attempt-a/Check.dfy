// The checker: validate a game and a profile, compute the profile's value, both
// players' best responses, NashConv and exploitability, and state what is proved
// about the verdict in CheckSound.

include "BestResponse.dfy"

module Checker {
  import opened Math
  import opened Game
  import opened BestResponse

  // A game: the tree and the arity of every infoset (A[I] = number of actions).
  datatype GameT = GameT(root: Node, A: seq<nat>)

  datatype Verdict = Verdict(
    value_p0: real,
    br_value_p0: real,
    br_value_p1: real,
    nash_conv: real,
    exploitability: real,
    br0: seq<nat>,   // br0[I] = action of p0's best response at infoset I
    br1: seq<nat>)

  datatype Result = Ok(verdict: Verdict) | Fail(reason: string)

  function BR(root: Node, p: nat, sigma: Strat, A: seq<nat>): Res
    requires WFN(root, A)
  {
    BRV([(1.0, root)], [], p, sigma, A)
  }

  // The first action recorded for infoset I, or 0 when I was never reached.
  function Lookup(ch: seq<(nat, nat)>, I: nat): nat
    decreases |ch|
  {
    if |ch| == 0 then 0 else if ch[0].0 == I then ch[0].1 else Lookup(ch[1..], I)
  }

  // Specified by Lookup; computed in one backward pass over ch with an array, so
  // the compiled checker is linear in |A| + |ch| instead of |A| * |ch| (the
  // Lookup form took 19 s on a 2000-infoset chain).
  function BuildBR(ch: seq<(nat, nat)>, A: seq<nat>): seq<nat>
  {
    seq(|A|, I requires 0 <= I < |A| => Lookup(ch, I))
  }
  by method
  {
    var a := new nat[|A|](_ => 0);
    var j := |ch|;
    while j > 0
      invariant 0 <= j <= |ch|
      invariant forall I :: 0 <= I < |A| ==> a[I] == Lookup(ch[j..], I)
    {
      j := j - 1;
      assert ch[j..][1..] == ch[j + 1..];
      if ch[j].0 < |A| {
        a[ch[j].0] := ch[j].1;
      }
    }
    assert ch[0..] == ch;
    return a[..];
  }

  // Runtime check: the recorded choices name a legal action at every infoset
  // and never disagree with the assembled strategy. Perfect recall implies it.
  predicate Consistent(ch: seq<(nat, nat)>, br: seq<nat>, A: seq<nat>)
  {
    |br| == |A|
    && (forall I :: 0 <= I < |A| ==> br[I] < A[I])
    && (forall j :: 0 <= j < |ch| ==> ch[j].0 < |br| && br[ch[j].0] == ch[j].1)
  }

  ghost function ToFun(br: seq<nat>): nat -> nat
  {
    (I: nat) => if I < |br| then br[I] else 0
  }

  function Check(g: GameT, prof: Strat): Result
  {
    if !WFN(g.root, g.A) then
      Fail("malformed game tree: arity table, infoset ids or chance children do not match")
    else if !Sane(g.root) then
      Fail("malformed game: a player is not 0 or 1, or a chance distribution is not exact")
    else if !ValidBeh(prof, g.A) then
      Fail("profile is not an exact distribution at every infoset")
    else
      var r0 := BR(g.root, 0, prof, g.A);
      var r1 := BR(g.root, 1, prof, g.A);
      var s0 := BuildBR(r0.ch, g.A);
      var s1 := BuildBR(r1.ch, g.A);
      if !Consistent(r0.ch, s0, g.A) || !Consistent(r1.ch, s1, g.A) then
        Fail("best response choices conflict at an infoset: the game lacks perfect recall")
      else
        var nc := r0.v + r1.v;
        Ok(Verdict(Val(g.root, 0, prof, prof), r0.v, r1.v, nc, nc / 2.0, s0, s1))
  }

  // ---------------------------------------------------------------- theorems

  // The best response theorem for one player p against sigma.
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
  {
    var L: seq<Item> := [(1.0, root)];
    assert AllWF(L, A);
    var r := BR(root, p, sigma, A);
    forall b: Strat | ValidBeh(b, A) ensures Val(root, p, b, sigma) <= r.v {
      UpperBound(L, [], p, b, sigma, A);
      reveal Mul();
    }
    forall s: nat -> nat | ValidPure(s, A) ensures PVal(root, p, s, sigma) <= r.v {
      PureIsOneHot(root, p, s, sigma, A);
      OneHotValid(s, A);
      UpperBound(L, [], p, OneHot(s, A), sigma, A);
      reveal Mul();
    }
    var br := BuildBR(r.ch, A);
    if Consistent(r.ch, br, A) {
      var s := ToFun(br);
      assert ValidPure(s, A);
      assert Agrees(s, r.ch);
      Attain(L, [], p, s, sigma, A);
      PureIsOneHot(root, p, s, sigma, A);
      reveal Mul();
    }
  }

  // Everything an Ok verdict guarantees.
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
  {
    var v := Check(g, prof).verdict;
    BRTheorem(g.root, 0, prof, g.A);
    BRTheorem(g.root, 1, prof, g.A);
    ZeroSum(g.root, prof);
    assert Val(g.root, 0, prof, prof) <= v.br_value_p0;
    assert Val(g.root, 1, prof, prof) <= v.br_value_p1;
  }
}
