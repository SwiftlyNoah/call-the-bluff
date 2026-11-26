// Proofs.dfy: (1) the bridge: p's expected utility in the game tree equals the
// sequence-form bilinear sum; (2) the bound: for every behavioral strategy of p,
// the bilinear sum is at most V (linearity in p's realization plan plus
// per-infoset maximization).

include "Model.dfy"
include "SeqForm.dfy"

module Proofs {
  import opened Model
  import opened SeqForm

  // ---------- arithmetic helpers ----------

  lemma MulLe(c: real, x: real, y: real)
    requires c >= 0.0 && x <= y
    ensures c * x <= c * y
  {
  }

  // ---------- linearity of BSum ----------

  lemma BSumNil(s: Profile)
    ensures BSum([], s) == 0.0
  {
    reveal BSum();
  }

  lemma BSumAppend(A: seq<Term>, B: seq<Term>, s: Profile)
    ensures BSum(A + B, s) == BSum(A, s) + BSum(B, s)
  {
    reveal BSum();
    if |A| == 0 {
      assert A + B == B;
    } else {
      assert (A + B)[0] == A[0];
      assert (A + B)[1..] == A[1..] + B;
      BSumAppend(A[1..], B, s);
    }
  }

  lemma BSumScale(c: real, A: seq<Term>, s: Profile)
    ensures BSum(Scale(c, A), s) == c * BSum(A, s)
  {
    reveal BSum();
    if |A| > 0 {
      var r := Scale(c, A);
      var r1 := Scale(c, A[1..]);
      forall j | 0 <= j < |r1| ensures r[1..][j] == r1[j] {
        assert A[1..][j] == A[j + 1];
      }
      assert r[1..] == r1;
      BSumScale(c, A[1..], s);
      var R := Reach(s, A[0].path);
      calc {
        BSum(r, s);
        r[0].w * R + BSum(r[1..], s);
        (c * A[0].w) * R + c * BSum(A[1..], s);
        c * (A[0].w * R + BSum(A[1..], s));
        c * BSum(A, s);
      }
    }
  }

  lemma ReachCons(s: Profile, st: Step, path: seq<Step>)
    ensures Reach(s, [st] + path) == Get(s, st.inf, st.act) * Reach(s, path)
  {
    assert ([st] + path)[0] == st;
    assert ([st] + path)[1..] == path;
  }

  lemma BSumPrepend(st: Step, A: seq<Term>, s: Profile)
    ensures BSum(Prepend(st, A), s) == Get(s, st.inf, st.act) * BSum(A, s)
  {
    reveal BSum();
    if |A| > 0 {
      var r := Prepend(st, A);
      var G := Get(s, st.inf, st.act);
      assert r[1..] == Prepend(st, A[1..]);
      BSumPrepend(st, A[1..], s);
      ReachCons(s, st, A[0].path);
      var R := Reach(s, A[0].path);
      calc {
        BSum(r, s);
        r[0].w * Reach(s, [st] + A[0].path) + BSum(r[1..], s);
        A[0].w * (G * R) + G * BSum(A[1..], s);
        G * (A[0].w * R + BSum(A[1..], s));
        G * BSum(A, s);
      }
    }
  }

  // ---------- the bridge: tree expected utility = bilinear sum ----------

  lemma Bridge(n: Node, s: Profile, p: nat)
    ensures BSum(Terms(n, s, p), s) == EU(n, s, p)
    decreases n, 2
  {
    if n.Leaf? {
      var Z := Terms(n, s, p);
      assert Z[1..] == [];
      reveal BSum();
    } else {
      BridgeKids(n, s, p, 0);
    }
  }

  lemma BridgeKids(n: Node, s: Profile, p: nat, i: nat)
    requires !n.Leaf?
    ensures BSum(TermsKids(n, s, p, i), s) == EUKids(n, s, p, i)
    decreases n, 1, |n.kids| - i
  {
    BSumNil(s);
    if i < |n.kids| {
      var T := Terms(n.kids[i], s, p);
      Bridge(n.kids[i], s, p);
      BridgeKids(n, s, p, i + 1);
      BSumAppend(KidTerms(n, s, p, i), TermsKids(n, s, p, i + 1), s);
      if n.Dec? && n.player == p {
        BSumPrepend(Step(n.inf, i), T, s);
      } else {
        BSumScale(Coef(n, s, i), T, s);
      }
    }
  }

  // ---------- the terminal list only depends on the opponent's play ----------

  lemma TermsAgree(g: Game, n: Node, s: Profile, t: Profile, p: nat)
    requires WFNode(n, g.owner, g.nAct) && AgreeOpp(g, s, t, p)
    ensures Terms(n, s, p) == Terms(n, t, p)
    decreases n, 2
  {
    if !n.Leaf? {
      TermsAgreeKids(g, n, s, t, p, 0);
    }
  }

  lemma TermsAgreeKids(g: Game, n: Node, s: Profile, t: Profile, p: nat, i: nat)
    requires !n.Leaf? && WFNode(n, g.owner, g.nAct) && AgreeOpp(g, s, t, p)
    ensures TermsKids(n, s, p, i) == TermsKids(n, t, p, i)
    decreases n, 1, |n.kids| - i
  {
    if i < |n.kids| {
      TermsAgree(g, n.kids[i], s, t, p);
      TermsAgreeKids(g, n, s, t, p, i + 1);
      if n.Dec? && n.player != p {
        assert Coef(n, s, i) == Coef(n, t, i);
      }
    }
  }

  // ---------- p's steps are in range ----------

  ghost predicate PathOk(path: seq<Step>, nAct: seq<nat>)
  {
    forall k :: 0 <= k < |path| ==> path[k].inf < |nAct| && path[k].act < nAct[path[k].inf]
  }

  ghost predicate StepsOk(Z: seq<Term>, nAct: seq<nat>)
  {
    forall t :: t in Z ==> PathOk(t.path, nAct)
  }

  lemma StepsAppend(A: seq<Term>, B: seq<Term>, nAct: seq<nat>)
    requires StepsOk(A, nAct) && StepsOk(B, nAct)
    ensures StepsOk(A + B, nAct)
  {
  }

  lemma TermsSteps(n: Node, s: Profile, p: nat, owner: seq<nat>, nAct: seq<nat>)
    requires WFNode(n, owner, nAct)
    ensures StepsOk(Terms(n, s, p), nAct)
    decreases n, 2
  {
    if !n.Leaf? {
      TermsStepsKids(n, s, p, owner, nAct, 0);
    }
  }

  lemma TermsStepsKids(n: Node, s: Profile, p: nat, owner: seq<nat>, nAct: seq<nat>, i: nat)
    requires !n.Leaf? && WFNode(n, owner, nAct)
    ensures StepsOk(TermsKids(n, s, p, i), nAct)
    decreases n, 1, |n.kids| - i
  {
    if i < |n.kids| {
      var T := Terms(n.kids[i], s, p);
      TermsSteps(n.kids[i], s, p, owner, nAct);
      TermsStepsKids(n, s, p, owner, nAct, i + 1);
      var K := KidTerms(n, s, p, i);
      if n.Dec? && n.player == p {
        var st := Step(n.inf, i);
        forall u | u in K ensures PathOk(u.path, nAct) {
          var j :| 0 <= j < |K| && K[j] == u;
          assert T[j] in T;
          assert u.path == [st] + T[j].path;
        }
      } else {
        forall u | u in K ensures PathOk(u.path, nAct) {
          var j :| 0 <= j < |K| && K[j] == u;
          assert T[j] in T;
        }
      }
      StepsAppend(K, TermsKids(n, s, p, i + 1), nAct);
    }
  }

  lemma SubSteps(Z: seq<Term>, a: nat, nAct: seq<nat>)
    requires StepsOk(Z, nAct)
    ensures StepsOk(Sub(Z, a), nAct)
  {
    var F := Filter(Z, HeadAct(a));
    var r := Strip(F);
    forall t | t in r ensures PathOk(t.path, nAct) {
      var j :| 0 <= j < |r| && r[j] == t;
      assert F[j] in F;
      var path := F[j].path;
      assert PathOk(path, nAct);
      forall k | 0 <= k < |t.path| ensures t.path[k].inf < |nAct| && t.path[k].act < nAct[t.path[k].inf] {
        assert t.path[k] == path[k + 1];
      }
    }
  }

  // ---------- partition lemmas ----------

  lemma BSumCons(x: Term, R: seq<Term>, s: Profile)
    ensures BSum([x] + R, s) == x.w * Reach(s, x.path) + BSum(R, s)
  {
    reveal BSum();
    assert ([x] + R)[0] == x;
    assert ([x] + R)[1..] == R;
  }

  lemma BSumFilterCons(Z: seq<Term>, q: Pred, s: Profile)
    requires |Z| > 0
    ensures BSum(Filter(Z, q), s) ==
      (if Holds(q, Z[0]) then Z[0].w * Reach(s, Z[0].path) else 0.0) + BSum(Filter(Z[1..], q), s)
  {
    var R := Filter(Z[1..], q);
    if Holds(q, Z[0]) {
      assert Filter(Z, q) == [Z[0]] + R;
      BSumCons(Z[0], R, s);
    } else {
      assert Filter(Z, q) == R;
    }
  }

  lemma Split(Z: seq<Term>, q: Pred, q1: Pred, q2: Pred, s: Profile)
    requires forall t :: t in Z ==> Holds(q, t) == (Holds(q1, t) || Holds(q2, t))
    requires forall t :: t in Z ==> !(Holds(q1, t) && Holds(q2, t))
    ensures BSum(Filter(Z, q), s) == BSum(Filter(Z, q1), s) + BSum(Filter(Z, q2), s)
  {
    BSumNil(s);
    if |Z| > 0 {
      assert Z[0] in Z;
      assert forall t :: t in Z[1..] ==> t in Z;
      Split(Z[1..], q, q1, q2, s);
      BSumFilterCons(Z, q, s);
      BSumFilterCons(Z, q1, s);
      BSumFilterCons(Z, q2, s);
    }
  }

  lemma FilterAll(Z: seq<Term>, q: Pred)
    requires forall t :: t in Z ==> Holds(q, t)
    ensures Filter(Z, q) == Z
  {
    if |Z| > 0 {
      assert Z[0] in Z;
      assert forall t :: t in Z[1..] ==> t in Z;
      FilterAll(Z[1..], q);
      assert [Z[0]] + Z[1..] == Z;
    }
  }

  lemma FilterNone(Z: seq<Term>, q: Pred)
    requires forall t :: t in Z ==> !Holds(q, t)
    ensures Filter(Z, q) == []
  {
    if |Z| > 0 {
      assert Z[0] in Z;
      assert forall t :: t in Z[1..] ==> t in Z;
      FilterNone(Z[1..], q);
    }
  }

  lemma EmptyW(Z: seq<Term>, s: Profile)
    requires forall t :: t in Z ==> |t.path| == 0
    ensures BSum(Z, s) == WSum(Z)
  {
    reveal BSum();
    if |Z| > 0 {
      assert Z[0] in Z;
      assert forall t :: t in Z[1..] ==> t in Z;
      EmptyW(Z[1..], s);
    }
  }

  // ---------- splitting p's first infoset by action ----------

  // Products in the per-infoset argument go through an opaque Mul, so the
  // solver sees them as plain terms; the arithmetic facts are in small lemmas.
  opaque ghost function Mul(x: real, y: real): real { x * y }

  lemma MulDef(x: real, y: real)
    ensures Mul(x, y) == x * y
  {
    reveal Mul();
  }

  opaque ghost function SumAct(In: seq<Term>, I: nat, a: nat, n: nat, s: Profile): real
    decreases n - a
  {
    if a >= n then 0.0
    else Mul(Get(s, I, a), BSum(Sub(In, a), s)) + SumAct(In, I, a + 1, n, s)
  }

  lemma StripSum(F: seq<Term>, I: nat, a: nat, s: Profile)
    requires forall t :: t in F ==> |t.path| > 0 && t.path[0] == Step(I, a)
    ensures BSum(F, s) == Get(s, I, a) * BSum(Strip(F), s)
  {
    reveal BSum();
    if |F| > 0 {
      var r := Strip(F);
      var G := Get(s, I, a);
      assert F[0] in F;
      assert forall t :: t in F[1..] ==> t in F;
      assert r[1..] == Strip(F[1..]);
      StripSum(F[1..], I, a, s);
      var R := Reach(s, Tail(F[0].path));
      assert Reach(s, F[0].path) == G * R;
      calc {
        BSum(F, s);
        F[0].w * (G * R) + G * BSum(r[1..], s);
        G * (F[0].w * R + BSum(r[1..], s));
        G * BSum(r, s);
      }
    }
  }

  lemma Lin(Ge: real, BF: real, Ge1: real, GX: real, S: real, S1: real)
    requires Ge == BF + Ge1 && BF == GX && Ge1 == S1 && S == GX + S1
    ensures Ge == S
  {
  }

  lemma SumActUnfold(In: seq<Term>, I: nat, a: nat, n: nat, s: Profile)
    requires a < n
    ensures SumAct(In, I, a, n, s) == Mul(Get(s, I, a), BSum(Sub(In, a), s)) + SumAct(In, I, a + 1, n, s)
  {
    reveal SumAct();
  }

  lemma GeSum(In: seq<Term>, I: nat, a: nat, n: nat, s: Profile)
    requires a <= n
    requires forall t :: t in In ==> |t.path| > 0 && t.path[0].inf == I && t.path[0].act < n
    ensures BSum(Filter(In, HeadActGe(a)), s) == SumAct(In, I, a, n, s)
    decreases n - a
  {
    if a == n {
      FilterNone(In, HeadActGe(a));
      assert BSum([], s) == 0.0 by { reveal BSum(); }
      assert SumAct(In, I, a, n, s) == 0.0 by { reveal SumAct(); }
      assert BSum(Filter(In, HeadActGe(a)), s) == 0.0;
    } else {
      var F := Filter(In, HeadAct(a));
      var Ge := BSum(Filter(In, HeadActGe(a)), s);
      var Ge1 := BSum(Filter(In, HeadActGe(a + 1)), s);
      Split(In, HeadActGe(a), HeadAct(a), HeadActGe(a + 1), s);
      assert Ge == BSum(F, s) + Ge1;
      forall t | t in F ensures |t.path| > 0 && t.path[0] == Step(I, a) {
        assert t in In;
      }
      StripSum(F, I, a, s);
      GeSum(In, I, a + 1, n, s);
      SumActUnfold(In, I, a, n, s);
      assert A1: Ge == BSum(F, s) + Ge1;
      assert A2: Ge1 == SumAct(In, I, a + 1, n, s);
      MulDef(Get(s, I, a), BSum(Sub(In, a), s));
      assert A3: BSum(F, s) == Mul(Get(s, I, a), BSum(Sub(In, a), s));
      assert A4: SumAct(In, I, a, n, s) == Mul(Get(s, I, a), BSum(Sub(In, a), s)) + SumAct(In, I, a + 1, n, s);
      assert Ge == SumAct(In, I, a, n, s) by {
        reveal A1, A2, A3, A4;
        Lin(Ge, BSum(F, s), Ge1, Mul(Get(s, I, a), BSum(Sub(In, a), s)),
            SumAct(In, I, a, n, s), SumAct(In, I, a + 1, n, s));
      }
    }
  }

  lemma ActSplit(In: seq<Term>, I: nat, n: nat, s: Profile)
    requires forall t :: t in In ==> |t.path| > 0 && t.path[0].inf == I && t.path[0].act < n
    ensures BSum(In, s) == SumAct(In, I, 0, n, s)
  {
    FilterAll(In, HeadActGe(0));
    GeSum(In, I, 0, n, s);
  }

  // ---------- the bound ----------

  opaque ghost predicate RowsOk(s: Profile, nAct: seq<nat>)
  {
    |s| == |nAct| && forall I :: 0 <= I < |s| ==> ValidRow(s[I], nAct[I])
  }

  lemma RowOf(s: Profile, nAct: seq<nat>, I: nat)
    requires RowsOk(s, nAct) && I < |nAct|
    ensures I < |s| && ValidRow(s[I], nAct[I])
  {
    reveal RowsOk();
  }

  lemma RowNonneg(r: seq<real>, k: nat)
    requires forall a :: 0 <= a < |r| ==> r[a] >= 0.0
    ensures RowSumFrom(r, k) >= 0.0
    decreases |r| - k
  {
    if k < |r| { RowNonneg(r, k + 1); }
  }

  // All the arithmetic of one step of MaxBound, on plain reals.
  lemma StepArith(b: real, B: real, v: real, S: real, S1: real, R: real, R1: real,
                  M: real, M1: real, last: bool)
    requires b >= 0.0 && B <= v
    requires S == Mul(b, B) + S1 && R == b + R1
    requires last ==> S1 == 0.0 && R1 == 0.0 && M == v
    requires !last ==> R1 >= 0.0 && S1 <= Mul(M1, R1) && M == Max(v, M1)
    ensures S <= Mul(M, R)
  {
    reveal Mul();
    MulLe(b, B, v);
    if last {
      Comm(b, v);
    } else {
      MulLe(b, v, M);
      MulLe(R1, M1, M);
      Comm(M1, R1);
      Comm(M, R1);
      Comm(b, M);
      Distrib(M, b, R1);
    }
  }

  lemma Comm(x: real, y: real)
    ensures x * y == y * x
  {
  }

  lemma Distrib(m: real, x: real, y: real)
    ensures m * (x + y) == m * x + m * y
  {
  }

  lemma VBound(nAct: seq<nat>, Z: seq<Term>, d: nat, s: Profile)
    requires PathsLe(Z, d) && StepsOk(Z, nAct) && RowsOk(s, nAct)
    ensures BSum(Z, s) <= V(nAct, Z, d)
    decreases d, 2, 0, 0
  {
    reveal V();
    BSumNil(s);
    if |Z| > 0 {
      Split(Z, Any, IsEmpty, NonEmpty, s);
      FilterAll(Z, Any);
      EmptyW(Filter(Z, IsEmpty), s);
      VHBound(nAct, Filter(Z, NonEmpty), d, s);
      reveal V();
    }
  }

  lemma VHBound(nAct: seq<nat>, L: seq<Term>, d: nat, s: Profile)
    requires PathsLe(L, d) && (forall t :: t in L ==> |t.path| > 0)
    requires StepsOk(L, nAct) && RowsOk(s, nAct)
    ensures BSum(L, s) <= VH(nAct, L, d)
    decreases d, 1, |L|, 0
  {
    reveal VH();
    BSumNil(s);
    if |L| > 0 {
      assert L[0] in L;
      var I := L[0].path[0].inf;
      assert PathOk(L[0].path, nAct);
      var n := NA(nAct, I);
      assert n >= 1;
      var In := Filter(L, HeadInf(I));
      var Out := Filter(L, NotHeadInf(I));
      Split(L, Any, HeadInf(I), NotHeadInf(I), s);
      FilterAll(L, Any);
      OutShorter(L, I);
      VHBound(nAct, Out, d, s);
      forall t | t in In ensures |t.path| > 0 && t.path[0].inf == I && t.path[0].act < n {
        assert PathOk(t.path, nAct);
      }
      ActSplit(In, I, n, s);
      RowOf(s, nAct, I);
      MaxBound(nAct, In, I, 0, n, d, s);
      assert RowSumFrom(s[I], 0) == 1.0;
      MulDef(MaxV(nAct, In, 0, n, d), 1.0);
      reveal VH();
    }
  }

  lemma MaxVUnfold(nAct: seq<nat>, In: seq<Term>, a: nat, n: nat, d: nat)
    requires a < n && 1 <= d && PathsLe(In, d)
    ensures SubLeHolds0(In, a, d)
    ensures a + 1 == n ==> MaxV(nAct, In, a, n, d) == V(nAct, Sub(In, a), d - 1)
    ensures a + 1 < n ==> MaxV(nAct, In, a, n, d) == Max(V(nAct, Sub(In, a), d - 1), MaxV(nAct, In, a + 1, n, d))
  {
    reveal MaxV();
    SubLe(In, a, d);
  }

  ghost predicate SubLeHolds0(In: seq<Term>, a: nat, d: nat)
  {
    1 <= d && PathsLe(Sub(In, a), d - 1)
  }

  lemma RowSumUnfold(r: seq<real>, k: nat)
    requires k < |r|
    ensures RowSumFrom(r, k) == r[k] + RowSumFrom(r, k + 1)
    ensures k + 1 == |r| ==> RowSumFrom(r, k + 1) == 0.0
  {
  }

  lemma SumActLast(In: seq<Term>, I: nat, n: nat, s: Profile)
    ensures SumAct(In, I, n, n, s) == 0.0
  {
    reveal SumAct();
  }

  lemma MaxBound(nAct: seq<nat>, In: seq<Term>, I: nat, a: nat, n: nat, d: nat, s: Profile)
    requires a < n && 1 <= d && PathsLe(In, d) && StepsOk(In, nAct) && RowsOk(s, nAct)
    requires I < |nAct| && n == nAct[I] && I < |s| && ValidRow(s[I], n)
    ensures SumAct(In, I, a, n, s) <= Mul(MaxV(nAct, In, a, n, d), RowSumFrom(s[I], a))
    decreases d, 0, 0, n - a
  {
    SubLe(In, a, d);
    SubSteps(In, a, nAct);
    VBound(nAct, Sub(In, a), d - 1, s);
    SumActUnfold(In, I, a, n, s);
    RowSumUnfold(s[I], a);
    MaxVUnfold(nAct, In, a, n, d);
    assert Get(s, I, a) == s[I][a] && s[I][a] >= 0.0;
    if a + 1 == n {
      SumActLast(In, I, n, s);
      StepArith(Get(s, I, a), BSum(Sub(In, a), s), V(nAct, Sub(In, a), d - 1),
                SumAct(In, I, a, n, s), SumAct(In, I, a + 1, n, s),
                RowSumFrom(s[I], a), RowSumFrom(s[I], a + 1),
                MaxV(nAct, In, a, n, d), 0.0, true);
    } else {
      MaxBound(nAct, In, I, a + 1, n, d, s);
      RowNonneg(s[I], a + 1);
      StepArith(Get(s, I, a), BSum(Sub(In, a), s), V(nAct, Sub(In, a), d - 1),
                SumAct(In, I, a, n, s), SumAct(In, I, a + 1, n, s),
                RowSumFrom(s[I], a), RowSumFrom(s[I], a + 1),
                MaxV(nAct, In, a, n, d), MaxV(nAct, In, a + 1, n, d), false);
    }
  }
}
