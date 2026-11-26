// Check.dfy: the best-response value, the core theorems, and Check(game, profile).

include "Model.dfy"
include "SeqForm.dfy"
include "Proofs.dfy"
include "Recall.dfy"

module Checker {
  import opened Model
  import opened SeqForm
  import opened Proofs
  import opened Recall

  // ---------- best response ----------

  // The best value player p can get against the opponent's part of s.
  function BRValue(g: Game, s: Profile, p: nat): real
  {
    var Z := Terms(g.root, s, p);
    V(g.nAct, Z, MaxLen(Z))
  }

  // A pure strategy for p made of the per-infoset choices behind BRValue.
  function BRPure(g: Game, s: Profile, p: nat): seq<nat>
  {
    var Z := Terms(g.root, s, p);
    PickV(g.nAct, Z, MaxLen(Z), seq(|g.nAct|, _ => 0))
  }

  // ---------- THEOREM 1 (upper bound, behavioral) ----------
  // Every behavioral strategy of p, played against the opponent's part of s,
  // earns p at most BRValue(g, s, p). (t is any valid profile that agrees with s
  // at every infoset of the opponent; its rows at p's infosets are arbitrary
  // probability distributions, i.e. an arbitrary behavioral strategy of p.)
  lemma BRUpperBound(g: Game, s: Profile, t: Profile, p: nat)
    requires WF(g) && ValidProfile(g, t) && AgreeOpp(g, s, t, p)
    ensures EU(g.root, t, p) <= BRValue(g, s, p)
  {
    var Z := Terms(g.root, s, p);
    TermsAgree(g, g.root, s, t, p);
    assert Z == Terms(g.root, t, p);
    Bridge(g.root, t, p);
    TermsSteps(g.root, s, p, g.owner, g.nAct);
    assert RowsOk(t, g.nAct) by { reveal RowsOk(); }
    VBound(g.nAct, Z, MaxLen(Z), t);
  }

  // ---------- THEOREM 1' (upper bound, pure) ----------
  // Every pure strategy pi of p earns p at most BRValue(g, s, p) against s.
  lemma PureUpperBound(g: Game, s: Profile, pi: seq<nat>, p: nat)
    requires WF(g) && ValidProfile(g, s) && PureOk(g, pi)
    ensures EU(g.root, Override(g, s, pi, p), p) <= BRValue(g, s, p)
  {
    var t := Override(g, s, pi, p);
    forall I | 0 <= I < |t| ensures ValidRow(t[I], g.nAct[I]) {
      if g.owner[I] == p {
        PureRowValid(pi[I], g.nAct[I]);
      }
    }
    assert ValidProfile(g, t);
    forall I: nat, a: nat | I < |g.owner| && g.owner[I] != p ensures Get(s, I, a) == Get(t, I, a) {
    }
    BRUpperBound(g, s, t, p);
  }

  lemma PureRowSum(a: nat, n: nat, k: nat)
    requires a < n && k <= n
    ensures RowSumFrom(PureRow(a, n), k) == if k <= a then 1.0 else 0.0
    decreases n - k
  {
    if k < n { PureRowSum(a, n, k + 1); }
  }

  lemma PureRowValid(a: nat, n: nat)
    requires a < n
    ensures ValidRow(PureRow(a, n), n)
  {
    PureRowSum(a, n, 0);
  }

  // ---------- zero-sum ----------

  lemma ZeroSum(n: Node, s: Profile)
    ensures EU(n, s, 1) == -EU(n, s, 0)
    decreases n, 1
  {
    if !n.Leaf? { ZeroSumKids(n, s, 0); }
  }

  lemma ZeroSumKids(n: Node, s: Profile, i: nat)
    requires !n.Leaf?
    ensures EUKids(n, s, 1, i) == -EUKids(n, s, 0, i)
    decreases n, 0, |n.kids| - i
  {
    if i < |n.kids| {
      ZeroSum(n.kids[i], s);
      ZeroSumKids(n, s, i + 1);
    }
  }

  // ---------- THEOREM 2 (nash_conv is nonnegative) ----------
  lemma NashConvNonneg(g: Game, s: Profile)
    requires WF(g) && ValidProfile(g, s)
    ensures BRValue(g, s, 0) + BRValue(g, s, 1) >= 0.0
  {
    BRUpperBound(g, s, s, 0);
    BRUpperBound(g, s, s, 1);
    ZeroSum(g.root, s);
  }

  // ---------- perfect recall ----------

  // Perfect recall for player p: every infoset of p is reached by exactly one
  // sequence of p's own (infoset, action) moves. Stated over the p-sequences of
  // the terminals, which do not depend on the profile (TermsPaths below).
  ghost predicate PerfectRecall(g: Game, p: nat)
  {
    PRZ(Terms(g.root, [], p))
  }

  ghost predicate SamePaths(A: seq<Term>, B: seq<Term>)
  {
    |A| == |B| && forall j :: 0 <= j < |A| ==> A[j].path == B[j].path
  }

  lemma SamePathsAppend(A: seq<Term>, B: seq<Term>, C: seq<Term>, D: seq<Term>)
    requires SamePaths(A, C) && SamePaths(B, D)
    ensures SamePaths(A + B, C + D)
  {
  }

  lemma TermsPaths(n: Node, s: Profile, s2: Profile, p: nat)
    ensures SamePaths(Terms(n, s, p), Terms(n, s2, p))
    decreases n, 2
  {
    if !n.Leaf? { TermsPathsKids(n, s, s2, p, 0); }
  }

  lemma TermsPathsKids(n: Node, s: Profile, s2: Profile, p: nat, i: nat)
    requires !n.Leaf?
    ensures SamePaths(TermsKids(n, s, p, i), TermsKids(n, s2, p, i))
    decreases n, 1, |n.kids| - i
  {
    if i < |n.kids| {
      TermsPaths(n.kids[i], s, s2, p);
      TermsPathsKids(n, s, s2, p, i + 1);
      SamePathsAppend(KidTerms(n, s, p, i), TermsKids(n, s, p, i + 1),
                      KidTerms(n, s2, p, i), TermsKids(n, s2, p, i + 1));
    }
  }

  lemma PRZTransfer(A: seq<Term>, B: seq<Term>)
    requires SamePaths(A, B) && PRZ(A)
    ensures PRZ(B)
  {
    reveal PRZ();
    forall t1, t2, k1, k2 | t1 in B && t2 in B && 0 <= k1 < |t1.path| && 0 <= k2 < |t2.path| &&
      t1.path[k1].inf == t2.path[k2].inf
      ensures t1.path[..k1] == t2.path[..k2]
    {
      var j1 :| 0 <= j1 < |B| && B[j1] == t1;
      var j2 :| 0 <= j2 < |B| && B[j2] == t2;
      assert A[j1] in A && A[j2] in A;
    }
  }

  // Every step in p's terminal sequences is a move of p.
  ghost predicate OwnedBy(Z: seq<Term>, owner: seq<nat>, p: nat)
  {
    forall t, k :: t in Z && 0 <= k < |t.path| ==> t.path[k].inf < |owner| && owner[t.path[k].inf] == p
  }

  lemma TermsOwned(n: Node, s: Profile, p: nat, owner: seq<nat>, nAct: seq<nat>)
    requires WFNode(n, owner, nAct)
    ensures OwnedBy(Terms(n, s, p), owner, p)
    decreases n, 2
  {
    if !n.Leaf? { TermsOwnedKids(n, s, p, owner, nAct, 0); }
  }

  lemma TermsOwnedKids(n: Node, s: Profile, p: nat, owner: seq<nat>, nAct: seq<nat>, i: nat)
    requires !n.Leaf? && WFNode(n, owner, nAct)
    ensures OwnedBy(TermsKids(n, s, p, i), owner, p)
    decreases n, 1, |n.kids| - i
  {
    if i < |n.kids| {
      var T := Terms(n.kids[i], s, p);
      TermsOwned(n.kids[i], s, p, owner, nAct);
      TermsOwnedKids(n, s, p, owner, nAct, i + 1);
      var K := KidTerms(n, s, p, i);
      forall u, k | u in K && 0 <= k < |u.path|
        ensures u.path[k].inf < |owner| && owner[u.path[k].inf] == p
      {
        var j :| 0 <= j < |K| && K[j] == u;
        assert T[j] in T;
        if n.Dec? && n.player == p {
          assert u.path == [Step(n.inf, i)] + T[j].path;
          if k > 0 { assert u.path[k] == T[j].path[k - 1]; }
        }
      }
    }
  }

  // ---------- THEOREM 3 (attainment under perfect recall) ----------
  // With perfect recall for p, the pure strategy BRPure(g, s, p) is legal and
  // earns p exactly BRValue(g, s, p). So the runtime attainment check in Check
  // never fails on a perfect-recall game.
  lemma BRAttained(g: Game, s: Profile, p: nat)
    requires WF(g) && ValidProfile(g, s) && PerfectRecall(g, p)
    ensures PureOk(g, BRPure(g, s, p))
    ensures EU(g.root, Override(g, s, BRPure(g, s, p), p), p) == BRValue(g, s, p)
  {
    var Z := Terms(g.root, s, p);
    var d := MaxLen(Z);
    var pi0: seq<nat> := seq(|g.nAct|, _ => 0);
    var pF := BRPure(g, s, p);
    assert pF == PickV(g.nAct, Z, d, pi0);
    assert LegalPi(pi0, g.nAct);
    PickVLegal(g.nAct, Z, d, pi0);
    assert PureOk(g, pF);
    TermsPaths(g.root, [], s, p);
    PRZTransfer(Terms(g.root, [], p), Z);
    var t := Override(g, s, pF, p);
    forall I: nat, a: nat | I < |g.owner| && g.owner[I] != p ensures Get(s, I, a) == Get(t, I, a) {
    }
    TermsAgree(g, g.root, s, t, p);
    Bridge(g.root, t, p);
    TermsSteps(g.root, s, p, g.owner, g.nAct);
    TermsOwned(g.root, s, p, g.owner, g.nAct);
    forall u, k | u in Z && 0 <= k < |u.path|
      ensures u.path[k].inf < |pF| &&
        Get(t, u.path[k].inf, u.path[k].act) == (if u.path[k].act == pF[u.path[k].inf] then 1.0 else 0.0)
    {
      assert PathOk(u.path, g.nAct);
      var J := u.path[k].inf;
      assert g.owner[J] == p;
      assert t[J] == PureRow(pF[J], g.nAct[J]);
    }
    assert IndOn(t, pF, Z);
    PickVAttains(g.nAct, Z, d, pi0, t);
  }

  // ---------- the checker ----------

  datatype Result =
    | Ok(value_p0: real, br0: real, br1: real, nash_conv: real, expl: real,
         pi0: seq<nat>, pi1: seq<nat>)
    | Err(msg: string)

  method Check(g: Game, s: Profile) returns (r: Result)
    ensures r.Ok? ==> WF(g) && ValidProfile(g, s)
    ensures r.Ok? ==> r.value_p0 == EU(g.root, s, 0)
    ensures r.Ok? ==> r.br0 == BRValue(g, s, 0) && r.br1 == BRValue(g, s, 1)
    // upper bound: no behavioral (hence no pure) strategy of p beats br_p
    ensures r.Ok? ==> forall t :: ValidProfile(g, t) && AgreeOpp(g, s, t, 0) ==> EU(g.root, t, 0) <= r.br0
    ensures r.Ok? ==> forall t :: ValidProfile(g, t) && AgreeOpp(g, s, t, 1) ==> EU(g.root, t, 1) <= r.br1
    ensures r.Ok? ==> forall pi :: PureOk(g, pi) ==> EU(g.root, Override(g, s, pi, 0), 0) <= r.br0
    ensures r.Ok? ==> forall pi :: PureOk(g, pi) ==> EU(g.root, Override(g, s, pi, 1), 1) <= r.br1
    // attainment: the returned pure strategies earn exactly br_p
    ensures r.Ok? ==> PureOk(g, r.pi0) && EU(g.root, Override(g, s, r.pi0, 0), 0) == r.br0
    ensures r.Ok? ==> PureOk(g, r.pi1) && EU(g.root, Override(g, s, r.pi1, 1), 1) == r.br1
    // nash_conv and exploitability
    ensures r.Ok? ==> r.nash_conv == r.br0 + r.br1 && r.nash_conv >= 0.0
    ensures r.Ok? ==> r.expl == r.nash_conv / 2.0 && r.expl >= 0.0
    // completeness: on a well-formed perfect-recall game with a valid profile, Check succeeds
    ensures WF(g) && ValidProfile(g, s) && PerfectRecall(g, 0) && PerfectRecall(g, 1) ==> r.Ok?
  {
    if !WF(g) {
      return Err("malformed game");
    }
    if !ValidProfile(g, s) {
      return Err("profile is not a behavioral strategy profile for this game");
    }
    var v0 := EU(g.root, s, 0);
    var br0 := BRValue(g, s, 0);
    var br1 := BRValue(g, s, 1);
    var pi0 := BRPure(g, s, 0);
    var pi1 := BRPure(g, s, 1);
    if WF(g) && ValidProfile(g, s) && PerfectRecall(g, 0) && PerfectRecall(g, 1) {
      BRAttained(g, s, 0);
      BRAttained(g, s, 1);
    }
    if !PureOk(g, pi0) || !PureOk(g, pi1) {
      return Err("internal: best response is not a pure strategy");
    }
    var e0 := EU(g.root, Override(g, s, pi0, 0), 0);
    var e1 := EU(g.root, Override(g, s, pi1, 1), 1);
    if e0 != br0 || e1 != br1 {
      return Err("best response value not attained (does the game have perfect recall?)");
    }
    forall t | ValidProfile(g, t) && AgreeOpp(g, s, t, 0) ensures EU(g.root, t, 0) <= br0 {
      BRUpperBound(g, s, t, 0);
    }
    forall t | ValidProfile(g, t) && AgreeOpp(g, s, t, 1) ensures EU(g.root, t, 1) <= br1 {
      BRUpperBound(g, s, t, 1);
    }
    forall pi | PureOk(g, pi) ensures EU(g.root, Override(g, s, pi, 0), 0) <= br0 {
      PureUpperBound(g, s, pi, 0);
    }
    forall pi | PureOk(g, pi) ensures EU(g.root, Override(g, s, pi, 1), 1) <= br1 {
      PureUpperBound(g, s, pi, 1);
    }
    NashConvNonneg(g, s);
    var nc := br0 + br1;
    r := Ok(v0, br0, br1, nc, nc / 2.0, pi0, pi1);
  }
}
