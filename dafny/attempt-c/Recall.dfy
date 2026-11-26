// Recall.dfy: under perfect recall the pure strategy behind V attains V.
//
// Perfect recall of player p, in sequence form: every infoset of p is reached
// by exactly one sequence of p's own moves. Over the terminal list this reads:
// whenever infoset J occurs at position k1 of one terminal's p-sequence and at
// position k2 of another's (or the same one's), the prefixes before J agree.

include "Model.dfy"
include "SeqForm.dfy"
include "Proofs.dfy"

module Recall {
  import opened Model
  import opened SeqForm
  import opened Proofs

  opaque ghost predicate PRZ(Z: seq<Term>)
  {
    forall t1, t2, k1, k2 ::
      (t1 in Z && t2 in Z && 0 <= k1 < |t1.path| && 0 <= k2 < |t2.path| &&
       t1.path[k1].inf == t2.path[k2].inf) ==> t1.path[..k1] == t2.path[..k2]
  }

  opaque ghost function Infs(Z: seq<Term>): set<nat>
  {
    set t, k | t in Z && 0 <= k < |t.path| :: t.path[k].inf
  }

  // ---------- subsets and subterms ----------

  lemma PRZSubset(Z: seq<Term>, Z2: seq<Term>)
    requires PRZ(Z) && forall t :: t in Z2 ==> t in Z
    ensures PRZ(Z2)
  {
    reveal PRZ();
  }

  lemma InfsSubset(Z: seq<Term>, Z2: seq<Term>)
    requires forall t :: t in Z2 ==> t in Z
    ensures Infs(Z2) <= Infs(Z)
  {
    reveal Infs();
  }

  // Every term of Sub(In, a) comes from a term of In with head action a.
  lemma SubOrigin(In: seq<Term>, a: nat, u: Term)
    requires u in Sub(In, a)
    ensures exists t :: t in In && |t.path| > 0 && t.path[0].act == a && u.w == t.w && u.path == t.path[1..]
  {
    var F := Filter(In, HeadAct(a));
    var r := Strip(F);
    var j :| 0 <= j < |r| && r[j] == u;
    assert F[j] in F;
  }

  lemma PRZSub(In: seq<Term>, a: nat)
    requires PRZ(In)
    ensures PRZ(Sub(In, a))
  {
    reveal PRZ();
    var S := Sub(In, a);
    forall u1, u2, k1, k2 | u1 in S && u2 in S && 0 <= k1 < |u1.path| && 0 <= k2 < |u2.path| &&
      u1.path[k1].inf == u2.path[k2].inf
      ensures u1.path[..k1] == u2.path[..k2]
    {
      SubOrigin(In, a, u1);
      SubOrigin(In, a, u2);
      var t1 :| t1 in In && |t1.path| > 0 && t1.path[0].act == a && u1.path == t1.path[1..];
      var t2 :| t2 in In && |t2.path| > 0 && t2.path[0].act == a && u2.path == t2.path[1..];
      assert t1.path[k1 + 1] == u1.path[k1];
      assert t2.path[k2 + 1] == u2.path[k2];
      assert t1.path[..k1 + 1] == t2.path[..k2 + 1];
      assert u1.path[..k1] == t1.path[..k1 + 1][1..];
      assert u2.path[..k2] == t2.path[..k2 + 1][1..];
    }
  }

  lemma InfsSub(In: seq<Term>, a: nat, J: nat)
    requires J in Infs(Sub(In, a))
    ensures exists t, k :: t in In && 1 <= k < |t.path| && t.path[0].act == a && t.path[k].inf == J
  {
    reveal Infs();
    var S := Sub(In, a);
    var u, k :| u in S && 0 <= k < |u.path| && u.path[k].inf == J;
    SubOrigin(In, a, u);
    var t :| t in In && |t.path| > 0 && t.path[0].act == a && u.path == t.path[1..];
    assert t.path[k + 1] == u.path[k];
  }

  lemma InfsSubIn(In: seq<Term>, a: nat)
    ensures Infs(Sub(In, a)) <= Infs(In)
  {
    reveal Infs();
    forall J | J in Infs(Sub(In, a)) ensures J in Infs(In) {
      InfsSub(In, a, J);
    }
  }

  // ---------- disjointness consequences of perfect recall ----------

  // The head infoset I does not occur again below its own actions.
  lemma HeadNotInSub(L: seq<Term>, I: nat, a: nat)
    requires PRZ(L)
    ensures I !in Infs(Sub(Filter(L, HeadInf(I)), a))
  {
    reveal PRZ();
    var In := Filter(L, HeadInf(I));
    if I in Infs(Sub(In, a)) {
      InfsSub(In, a, I);
      var t, k :| t in In && 1 <= k < |t.path| && t.path[0].act == a && t.path[k].inf == I;
      assert t in L;
      assert t.path[..k] == t.path[..0];
    }
  }

  // Different actions at I lead to disjoint sets of infosets.
  lemma SubsDisjoint(L: seq<Term>, I: nat, a: nat, b: nat)
    requires PRZ(L) && a != b
    ensures Infs(Sub(Filter(L, HeadInf(I)), a)) !! Infs(Sub(Filter(L, HeadInf(I)), b))
  {
    reveal PRZ();
    var In := Filter(L, HeadInf(I));
    forall J | J in Infs(Sub(In, a)) && J in Infs(Sub(In, b)) ensures false {
      InfsSub(In, a, J);
      InfsSub(In, b, J);
      var t1, k1 :| t1 in In && 1 <= k1 < |t1.path| && t1.path[0].act == a && t1.path[k1].inf == J;
      var t2, k2 :| t2 in In && 1 <= k2 < |t2.path| && t2.path[0].act == b && t2.path[k2].inf == J;
      assert t1 in L && t2 in L;
      assert t1.path[..k1] == t2.path[..k2];
      assert t1.path[..k1][0] == t1.path[0];
      assert t2.path[..k2][0] == t2.path[0];
    }
  }

  // I does not occur in the rest of L.
  lemma HeadNotInOut(L: seq<Term>, I: nat)
    requires PRZ(L) && |L| > 0 && |L[0].path| > 0 && L[0].path[0].inf == I
    ensures I !in Infs(Filter(L, NotHeadInf(I)))
  {
    reveal Infs();
    reveal PRZ();
    var Out := Filter(L, NotHeadInf(I));
    if I in Infs(Out) {
      var t, k :| t in Out && 0 <= k < |t.path| && t.path[k].inf == I;
      assert t in L && L[0] in L;
      assert t.path[..k] == L[0].path[..0];
      assert k == 0;
    }
  }

  // The infosets below I and the infosets in the rest of L are disjoint.
  lemma SubOutDisjoint(L: seq<Term>, I: nat, a: nat)
    requires PRZ(L)
    ensures Infs(Sub(Filter(L, HeadInf(I)), a)) !! Infs(Filter(L, NotHeadInf(I)))
  {
    reveal Infs();
    reveal PRZ();
    var In := Filter(L, HeadInf(I));
    var Out := Filter(L, NotHeadInf(I));
    forall J | J in Infs(Sub(In, a)) && J in Infs(Out) ensures false {
      InfsSub(In, a, J);
      var t1, k1 :| t1 in In && 1 <= k1 < |t1.path| && t1.path[0].act == a && t1.path[k1].inf == J;
      var t2, k2 :| t2 in Out && 0 <= k2 < |t2.path| && t2.path[k2].inf == J;
      assert t1 in L && t2 in L;
      assert t1.path[..k1] == t2.path[..k2];
      assert t1.path[..k1][0] == t1.path[0];
      assert t2.path[..k2][0] == t2.path[0];
    }
  }

  // ---------- frame: the pickers only write infosets that occur in their input ----------

  ghost predicate NotBelow(In: seq<Term>, a: nat, n: nat, J: nat)
  {
    forall a' :: a <= a' < n ==> J !in Infs(Sub(In, a'))
  }

  lemma PickVFrame(nAct: seq<nat>, Z: seq<Term>, d: nat, pi: seq<nat>, J: nat)
    requires PathsLe(Z, d)
    ensures |PickV(nAct, Z, d, pi)| == |pi|
    ensures J !in Infs(Z) && J < |pi| ==> PickV(nAct, Z, d, pi)[J] == pi[J]
    decreases d, 2, 0, 0
  {
    reveal PickV();
    if |Z| > 0 {
      InfsSubset(Z, Filter(Z, NonEmpty));
      PickHFrame(nAct, Filter(Z, NonEmpty), d, pi, J);
    }
  }

  lemma PickHFrame(nAct: seq<nat>, L: seq<Term>, d: nat, pi: seq<nat>, J: nat)
    requires PathsLe(L, d) && (forall t :: t in L ==> |t.path| > 0)
    ensures |PickH(nAct, L, d, pi)| == |pi|
    ensures J !in Infs(L) && J < |pi| ==> PickH(nAct, L, d, pi)[J] == pi[J]
    decreases d, 1, |L|, 0
  {
    reveal PickH();
    if |L| > 0 {
      var I := L[0].path[0].inf;
      assert L[0] in L;
      assert I in Infs(L) by { reveal Infs(); }
      var n := NA(nAct, I);
      OutShorter(L, I);
      var In := Filter(L, HeadInf(I));
      var Out := Filter(L, NotHeadInf(I));
      InfsSubset(L, In);
      InfsSubset(L, Out);
      var pi1 :=
        if n == 0 then pi
        else
          var best := ArgMaxV(nAct, In, 0, n, d).0;
          PickActs(nAct, In, 0, n, d, if I < |pi| then pi[I := best] else pi);
      if n != 0 {
        var best := ArgMaxV(nAct, In, 0, n, d).0;
        var pi0 := if I < |pi| then pi[I := best] else pi;
        forall a' | 0 <= a' < n ensures J !in Infs(L) ==> J !in Infs(Sub(In, a')) {
          InfsSubIn(In, a');
        }
        PickActsFrame(nAct, In, 0, n, d, pi0, J);
      }
      PickHFrame(nAct, Out, d, pi1, J);
      assert PickH(nAct, L, d, pi) == PickH(nAct, Out, d, pi1);
    }
  }

  lemma PickActsFrame(nAct: seq<nat>, In: seq<Term>, a: nat, n: nat, d: nat, pi: seq<nat>, J: nat)
    requires a <= n && 1 <= d && PathsLe(In, d)
    ensures |PickActs(nAct, In, a, n, d, pi)| == |pi|
    ensures NotBelow(In, a, n, J) && J < |pi| ==> PickActs(nAct, In, a, n, d, pi)[J] == pi[J]
    decreases d, 0, 0, n - a
  {
    reveal PickActs();
    if a < n {
      SubLe(In, a, d);
      PickVFrame(nAct, Sub(In, a), d - 1, pi, J);
      PickActsFrame(nAct, In, a + 1, n, d, PickV(nAct, Sub(In, a), d - 1, pi), J);
    }
  }

  // Running the actions a..n-1 is running a..b-1 and then b..n-1.
  lemma PickActsSplit(nAct: seq<nat>, In: seq<Term>, a: nat, b: nat, n: nat, d: nat, pi: seq<nat>)
    requires a <= b <= n && 1 <= d && PathsLe(In, d)
    ensures PickActs(nAct, In, a, n, d, pi) == PickActs(nAct, In, b, n, d, PickActs(nAct, In, a, b, d, pi))
    decreases b - a
  {
    reveal PickActs();
    if a < b {
      SubLe(In, a, d);
      PickActsSplit(nAct, In, a + 1, b, n, d, PickV(nAct, Sub(In, a), d - 1, pi));
    }
  }

  // The actions after b never write an infoset below b.
  lemma PickActsFrameAfter(nAct: seq<nat>, L: seq<Term>, I: nat, In: seq<Term>, a: nat, n: nat, d: nat, pi: seq<nat>, b: nat, J: nat)
    requires PRZ(L) && In == Filter(L, HeadInf(I))
    requires b < a <= n && 1 <= d && PathsLe(In, d)
    requires J in Infs(Sub(In, b))
    ensures |PickActs(nAct, In, a, n, d, pi)| == |pi|
    ensures J < |pi| ==> PickActs(nAct, In, a, n, d, pi)[J] == pi[J]
    decreases n - a
  {
    reveal PickActs();
    if a < n {
      SubLe(In, a, d);
      SubsDisjoint(L, I, a, b);
      assert J !in Infs(Sub(In, a));
      var p1 := PickV(nAct, Sub(In, a), d - 1, pi);
      PickVFrame(nAct, Sub(In, a), d - 1, pi, J);
      PickActsFrameAfter(nAct, L, I, In, a + 1, n, d, p1, b, J);
      assert PickActs(nAct, In, a, n, d, pi) == PickActs(nAct, In, a + 1, n, d, p1);
    }
  }

  // ---------- the picked actions are legal ----------

  ghost predicate LegalPi(pi: seq<nat>, nAct: seq<nat>)
  {
    |pi| == |nAct| && forall I :: 0 <= I < |pi| ==> pi[I] < nAct[I]
  }

  lemma ArgMaxRange(nAct: seq<nat>, In: seq<Term>, a: nat, n: nat, d: nat)
    requires a < n && 1 <= d && PathsLe(In, d)
    ensures a <= ArgMaxV(nAct, In, a, n, d).0 < n
    ensures ArgMaxV(nAct, In, a, n, d).1 == MaxV(nAct, In, a, n, d)
    ensures SubLeHolds(In, ArgMaxV(nAct, In, a, n, d).0, d) &&
      V(nAct, Sub(In, ArgMaxV(nAct, In, a, n, d).0), d - 1) == ArgMaxV(nAct, In, a, n, d).1
    decreases n - a
  {
    reveal ArgMaxV();
    reveal MaxV();
    SubLe(In, a, d);
    if a + 1 < n {
      ArgMaxRange(nAct, In, a + 1, n, d);
    }
  }

  // PathsLe(Sub(In, b), d - 1), as a predicate usable inside an ensures clause.
  ghost predicate SubLeHolds(In: seq<Term>, b: nat, d: nat)
  {
    1 <= d && PathsLe(Sub(In, b), d - 1)
  }

  lemma PickVLegal(nAct: seq<nat>, Z: seq<Term>, d: nat, pi: seq<nat>)
    requires PathsLe(Z, d) && LegalPi(pi, nAct)
    ensures LegalPi(PickV(nAct, Z, d, pi), nAct)
    decreases d, 2, 0, 0
  {
    reveal PickV();
    if |Z| > 0 { PickHLegal(nAct, Filter(Z, NonEmpty), d, pi); }
  }

  lemma PickHLegal(nAct: seq<nat>, L: seq<Term>, d: nat, pi: seq<nat>)
    requires PathsLe(L, d) && (forall t :: t in L ==> |t.path| > 0) && LegalPi(pi, nAct)
    ensures LegalPi(PickH(nAct, L, d, pi), nAct)
    decreases d, 1, |L|, 0
  {
    reveal PickH();
    if |L| > 0 {
      var I := L[0].path[0].inf;
      assert L[0] in L;
      var n := NA(nAct, I);
      OutShorter(L, I);
      var In := Filter(L, HeadInf(I));
      var Out := Filter(L, NotHeadInf(I));
      var pi1 :=
        if n == 0 then pi
        else
          var best := ArgMaxV(nAct, In, 0, n, d).0;
          PickActs(nAct, In, 0, n, d, if I < |pi| then pi[I := best] else pi);
      if n != 0 {
        var best := ArgMaxV(nAct, In, 0, n, d).0;
        ArgMaxRange(nAct, In, 0, n, d);
        var pi0 := if I < |pi| then pi[I := best] else pi;
        assert LegalPi(pi0, nAct);
        PickActsLegal(nAct, In, 0, n, d, pi0);
      }
      PickHLegal(nAct, Out, d, pi1);
      assert PickH(nAct, L, d, pi) == PickH(nAct, Out, d, pi1);
    }
  }

  lemma PickActsLegal(nAct: seq<nat>, In: seq<Term>, a: nat, n: nat, d: nat, pi: seq<nat>)
    requires a <= n && 1 <= d && PathsLe(In, d) && LegalPi(pi, nAct)
    ensures LegalPi(PickActs(nAct, In, a, n, d, pi), nAct)
    decreases d, 0, 0, n - a
  {
    reveal PickActs();
    if a < n {
      SubLe(In, a, d);
      PickVLegal(nAct, Sub(In, a), d - 1, pi);
      PickActsLegal(nAct, In, a + 1, n, d, PickV(nAct, Sub(In, a), d - 1, pi));
    }
  }

  // ---------- attainment ----------

  // beta plays the pure strategy pf at every step that occurs in Z.
  ghost predicate IndOn(beta: Profile, pf: seq<nat>, Z: seq<Term>)
  {
    forall t, k :: t in Z && 0 <= k < |t.path| ==>
      t.path[k].inf < |pf| &&
      Get(beta, t.path[k].inf, t.path[k].act) == (if t.path[k].act == pf[t.path[k].inf] then 1.0 else 0.0)
  }

  lemma SumActInd(In: seq<Term>, I: nat, a: nat, n: nat, beta: Profile, c: nat)
    requires c < n && a <= n
    requires forall a' :: a <= a' < n ==> |Sub(In, a')| == 0 || Get(beta, I, a') == (if a' == c then 1.0 else 0.0)
    ensures SumAct(In, I, a, n, beta) == (if a <= c then BSum(Sub(In, c), beta) else 0.0)
    decreases n - a
  {
    reveal SumAct();
    if a < n {
      SumActInd(In, I, a + 1, n, beta, c);
      MulDef(Get(beta, I, a), BSum(Sub(In, a), beta));
      if |Sub(In, a)| == 0 {
        BSumNil(beta);
        assert Sub(In, a) == [];
      }
    }
  }

  lemma PickVAttains(nAct: seq<nat>, Z: seq<Term>, d: nat, pi: seq<nat>, beta: Profile)
    requires PathsLe(Z, d) && StepsOk(Z, nAct) && PRZ(Z) && |pi| == |nAct|
    requires IndOn(beta, PickV(nAct, Z, d, pi), Z)
    ensures BSum(Z, beta) == V(nAct, Z, d)
    decreases d, 2, 0, 0
  {
    reveal V();
    BSumNil(beta);
    if |Z| > 0 {
      var NE := Filter(Z, NonEmpty);
      Split(Z, Any, IsEmpty, NonEmpty, beta);
      FilterAll(Z, Any);
      EmptyW(Filter(Z, IsEmpty), beta);
      PRZSubset(Z, NE);
      assert PickV(nAct, Z, d, pi) == PickH(nAct, NE, d, pi) by { reveal PickV(); }
      PickHAttains(nAct, NE, d, pi, beta);
    }
  }

  lemma {:isolate_assertions} PickHAttains(nAct: seq<nat>, L: seq<Term>, d: nat, pi: seq<nat>, beta: Profile)
    requires PathsLe(L, d) && (forall t :: t in L ==> |t.path| > 0)
    requires StepsOk(L, nAct) && PRZ(L) && |pi| == |nAct|
    requires IndOn(beta, PickH(nAct, L, d, pi), L)
    ensures BSum(L, beta) == VH(nAct, L, d)
    decreases d, 1, |L|, 0
  {
    reveal VH();
    BSumNil(beta);
    if |L| > 0 {
      assert L[0] in L;
      var I := L[0].path[0].inf;
      assert PathOk(L[0].path, nAct);
      var n := NA(nAct, I);
      assert 1 <= n && I < |pi|;
      var In := Filter(L, HeadInf(I));
      var Out := Filter(L, NotHeadInf(I));
      OutShorter(L, I);
      ArgMaxRange(nAct, In, 0, n, d);
      var best := ArgMaxV(nAct, In, 0, n, d).0;
      var pi0 := pi[I := best];
      var pi2 := PickActs(nAct, In, 0, n, d, pi0);
      var pF := PickH(nAct, L, d, pi);
      assert HF: pF == PickH(nAct, Out, d, pi2) by { reveal PickH(); }
      PickActsFrame(nAct, In, 0, n, d, pi0, I);
      PickHFrame(nAct, Out, d, pi2, I);
      assert |pi2| == |pi| && |pF| == |pi| by { reveal HF; }

      // the rest of L
      Split(L, Any, HeadInf(I), NotHeadInf(I), beta);
      FilterAll(L, Any);
      PRZSubset(L, Out);
      assert IndOn(beta, PickH(nAct, Out, d, pi2), Out) by { reveal HF; }
      PickHAttains(nAct, Out, d, pi2, beta);

      // pF picks best at I
      forall a' | 0 <= a' < n ensures I !in Infs(Sub(In, a')) {
        HeadNotInSub(L, I, a');
      }
      HeadNotInOut(L, I);
      assert pF[I] == best by { reveal HF; }

      // the terms through I collapse onto the chosen action
      forall t | t in In ensures |t.path| > 0 && t.path[0].inf == I && t.path[0].act < n {
        assert PathOk(t.path, nAct);
      }
      ActSplit(In, I, n, beta);
      forall a' | 0 <= a' < n
        ensures |Sub(In, a')| == 0 || Get(beta, I, a') == (if a' == best then 1.0 else 0.0)
      {
        if |Sub(In, a')| > 0 {
          var u := Sub(In, a')[0];
          assert u in Sub(In, a');
          SubOrigin(In, a', u);
          var t :| t in In && |t.path| > 0 && t.path[0].act == a' && u.path == t.path[1..];
          assert t in L;
          assert t.path[0].inf == I;
        }
      }
      SumActInd(In, I, 0, n, beta, best);
      var Sb := Sub(In, best);

      // the subtree under the chosen action is played by PickV on it
      SubLe(In, best, d);
      var pb := PickActs(nAct, In, 0, best, d, pi0);
      PickActsSplit(nAct, In, 0, best, n, d, pi0);
      var pv := PickV(nAct, Sb, d - 1, pb);
      assert pi2 == PickActs(nAct, In, best + 1, n, d, pv) by { reveal PickActs(); }
      PickActsFrame(nAct, In, 0, best, d, pi0, 0);
      PickVFrame(nAct, Sb, d - 1, pb, 0);
      forall J | J in Infs(Sb) ensures J < |pv| && pF[J] == pv[J] {
        InfsSub(In, best, J);
        var t, k :| t in In && 1 <= k < |t.path| && t.path[0].act == best && t.path[k].inf == J;
        assert t in L;
        assert PathOk(t.path, nAct);
        assert J < |pi|;
        SubOutDisjoint(L, I, best);
        PickHFrame(nAct, Out, d, pi2, J);
        PickActsFrameAfter(nAct, L, I, In, best + 1, n, d, pv, best, J);
        reveal HF;
      }
      forall u, k | u in Sb && 0 <= k < |u.path|
        ensures u.path[k].inf < |pv| &&
          Get(beta, u.path[k].inf, u.path[k].act) == (if u.path[k].act == pv[u.path[k].inf] then 1.0 else 0.0)
      {
        assert u.path[k].inf in Infs(Sb) by { reveal Infs(); }
        SubOrigin(In, best, u);
        var t :| t in In && |t.path| > 0 && t.path[0].act == best && u.path == t.path[1..];
        assert t in L;
        assert t.path[k + 1] == u.path[k];
      }
      assert PRZ(In) by { PRZSubset(L, In); }
      PRZSub(In, best);
      SubSteps(In, best, nAct);
      PickVAttains(nAct, Sb, d - 1, pb, beta);
    }
  }
}

