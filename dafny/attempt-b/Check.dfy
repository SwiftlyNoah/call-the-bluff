// Check.dfy: the theorems about the best response and the verified checker.

include "Model.dfy"
include "BestResponse.dfy"

module Checker {
  import opened GameModel
  import opened BestResponse

  // The best-response value of player p against the opponent's part of sigma.
  function BRValue(g: Game, p: nat, sigma: Strat): real
    requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma)
  {
    BR(g, p, sigma, [WN(g.root, 1.0)], []).0
  }

  // THEOREM 1a (behavioral upper bound): no behavioral strategy of p does better than BRValue.
  lemma BehavioralUpperBound(g: Game, p: nat, sigma: Strat, beta: Strat)
    requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma) && ValidBeh(g, p, beta)
    ensures Val(g, p, beta, sigma, g.root) <= BRValue(g, p, sigma)
  {
    var S := [WN(g.root, 1.0)];
    BRBound(g, p, beta, sigma, S, []);
    VSHead(g, p, beta, sigma, S);
    VSNil(g, p, beta, sigma, S[1..]);
    VSNil(g, p, beta, sigma, []);
  }

  // THEOREM 1b (pure upper bound): no pure strategy of p does better than BRValue.
  lemma PureUpperBound(g: Game, p: nat, sigma: Strat, pi: seq<nat>)
    requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma) && PureValid(g, p, pi)
    ensures Val(g, p, OneHot(g, pi), sigma, g.root) <= BRValue(g, p, sigma)
  {
    OneHotValid(g, p, pi);
    BehavioralUpperBound(g, p, sigma, OneHot(g, pi));
  }

  // Turn the algorithm's infoset -> action choices into a total pure strategy for p.
  function ToPure(g: Game, ch: map<nat, nat>): seq<nat>
    requires Shape(g)
    ensures |ToPure(g, ch)| == |g.infosetPlayer|
  {
    seq(|g.infosetPlayer|, I requires 0 <= I < |g.infosetPlayer| =>
      if I in ch && ch[I] < g.infosetActions[I] then ch[I] else 0)
  }

  lemma ToPureValid(g: Game, p: nat, ch: map<nat, nat>)
    requires WF(g)
    ensures PureValid(g, p, ToPure(g, ch))
  {
    forall I | 0 <= I < |g.infosetPlayer| ensures g.infosetActions[I] > 0 { WFInfoset(g, I); }
  }

  // value is p's best-response value against sigma, attained by the pure strategy pi.
  ghost predicate IsBestResponse(g: Game, p: nat, sigma: Strat, value: real, pi: seq<nat>)
    requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma)
  {
    PureValid(g, p, pi)
    && Val(g, p, OneHot(g, pi), sigma, g.root) == value
    && (forall beta | ValidBeh(g, p, beta) :: Val(g, p, beta, sigma, g.root) <= value)
    && (forall pi' | PureValid(g, p, pi') :: Val(g, p, OneHot(g, pi'), sigma, g.root) <= value)
  }

  datatype Verdict = Verdict(
    valueP0: real,         // expected payoff to player 0 under the profile
    brValue0: real,        // best player 0 can get against the profile's player 1
    brValue1: real,        // best player 1 can get against the profile's player 0 (p1's utility)
    nashConv: real,        // brValue0 + brValue1
    exploitability: real,  // nashConv / 2
    bestResponse0: seq<nat>,
    bestResponse1: seq<nat>)

  datatype Result = Ok(v: Verdict) | Err(msg: string)

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

  // Best response of player p, checked for attainment at run time.
  method BestResponseOf(g: Game, p: nat, sigma: Strat) returns (ok: bool, value: real, pi: seq<nat>)
    requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma)
    ensures ok ==> IsBestResponse(g, p, sigma, value, pi)
  {
    var res := BR(g, p, sigma, [WN(g.root, 1.0)], []);
    value := res.0;
    pi := ToPure(g, res.1);
    ToPureValid(g, p, res.1);
    OneHotValid(g, p, pi);
    var attained := Val(g, p, OneHot(g, pi), sigma, g.root);
    ok := attained == value;
    forall beta | ValidBeh(g, p, beta) ensures Val(g, p, beta, sigma, g.root) <= value {
      BehavioralUpperBound(g, p, sigma, beta);
    }
    forall pi' | PureValid(g, p, pi') ensures Val(g, p, OneHot(g, pi'), sigma, g.root) <= value {
      PureUpperBound(g, p, sigma, pi');
    }
  }

  // THEOREM 2 (the checker): whenever Check returns Ok, the verdict is Sound: value_p0 is the
  // profile's exact value, each br value is the maximum over all behavioral and all pure
  // strategies, attained by the returned pure strategy, and nash_conv = br0 + br1 >= 0.
  method Check(g: Game, sigma: Strat) returns (r: Result)
    ensures r.Ok? ==> WF(g) && ValidBeh(g, 0, sigma) && ValidBeh(g, 1, sigma) && Sound(g, sigma, r.v)
  {
    if !WF(g) { return Err("game is not well formed"); }
    if !ValidBeh(g, 0, sigma) || !ValidBeh(g, 1, sigma) {
      return Err("profile is not a behavioral strategy profile of this game");
    }
    var v0 := Val(g, 0, sigma, sigma, g.root);
    ZeroSum(g, sigma, g.root);
    var ok0, b0, pi0 := BestResponseOf(g, 0, sigma);
    if !ok0 { return Err("player 0's best response value is not attained (game without perfect recall?)"); }
    var ok1, b1, pi1 := BestResponseOf(g, 1, sigma);
    if !ok1 { return Err("player 1's best response value is not attained (game without perfect recall?)"); }
    // nash_conv >= 0: sigma itself is a behavioral strategy for each player.
    assert Val(g, 0, sigma, sigma, g.root) <= b0;
    assert Val(g, 1, sigma, sigma, g.root) <= b1;
    var nc := b0 + b1;
    r := Ok(Verdict(v0, b0, b1, nc, nc / 2.0, pi0, pi1));
  }

  // LEMMA: nash_conv >= 0 for every Ok verdict (a restatement of part of Sound, for the README).
  lemma NashConvNonNegative(g: Game, sigma: Strat, v: Verdict)
    requires WF(g) && ValidBeh(g, 0, sigma) && ValidBeh(g, 1, sigma) && Sound(g, sigma, v)
    ensures v.nashConv >= 0.0 && v.exploitability >= 0.0
  {}
}
