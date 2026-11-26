// SeqForm.dfy: the sequence-form view of a game from player p's side, and the
// best response by per-infoset maximization over p's sequences.
//
// Terms(n, s, p) lists every terminal z under n as Term(w, path) where
//   w    = (chance reach of z) * (opponent reach of z under s) * (p's payoff at z)
//   path = p's own sequence to z: the (infoset, action) pairs p plays on the way.
// For a behavioral strategy t of p, p's expected utility is the bilinear sum
//   BSum(Z, t) = sum over z of w(z) * Reach(t, path(z))
// where Reach(t, path) is t's realization weight of the sequence path(z).
// BSum is linear in the realization plan of p, so it is maximized by a
// per-infoset choice of the best continuation sequence: that is V.

include "Model.dfy"

module SeqForm {
  import opened Model

  datatype Step = Step(inf: nat, act: nat)
  datatype Term = Term(w: real, path: seq<Step>)

  // ---------- realization weights and the bilinear sum ----------

  function Reach(s: Profile, path: seq<Step>): real
  {
    if |path| == 0 then 1.0 else Get(s, path[0].inf, path[0].act) * Reach(s, path[1..])
  }

  opaque ghost function BSum(Z: seq<Term>, s: Profile): real
  {
    if |Z| == 0 then 0.0 else Z[0].w * Reach(s, Z[0].path) + BSum(Z[1..], s)
  }

  function WSum(Z: seq<Term>): real
  {
    if |Z| == 0 then 0.0 else Z[0].w + WSum(Z[1..])
  }

  function Scale(c: real, Z: seq<Term>): (r: seq<Term>)
    ensures |r| == |Z|
    ensures forall j :: 0 <= j < |Z| ==> r[j] == Term(c * Z[j].w, Z[j].path)
  {
    seq(|Z|, j requires 0 <= j < |Z| => Term(c * Z[j].w, Z[j].path))
  }

  function Prepend(st: Step, Z: seq<Term>): (r: seq<Term>)
    ensures |r| == |Z|
    ensures forall j :: 0 <= j < |Z| ==> r[j] == Term(Z[j].w, [st] + Z[j].path)
  {
    seq(|Z|, j requires 0 <= j < |Z| => Term(Z[j].w, [st] + Z[j].path))
  }

  function Tail(path: seq<Step>): seq<Step>
  {
    if |path| == 0 then [] else path[1..]
  }

  function Strip(Z: seq<Term>): (r: seq<Term>)
    ensures |r| == |Z|
    ensures forall j :: 0 <= j < |Z| ==> r[j] == Term(Z[j].w, Tail(Z[j].path))
  {
    seq(|Z|, j requires 0 <= j < |Z| => Term(Z[j].w, Tail(Z[j].path)))
  }

  // ---------- from the game tree to the terminal list ----------

  function KidTerms(n: Node, s: Profile, p: nat, i: nat): seq<Term>
    requires !n.Leaf? && i < |n.kids|
    decreases n, 0
  {
    if n.Dec? && n.player == p then Prepend(Step(n.inf, i), Terms(n.kids[i], s, p))
    else Scale(Coef(n, s, i), Terms(n.kids[i], s, p))
  }

  function Terms(n: Node, s: Profile, p: nat): seq<Term>
    decreases n, 2
  {
    if n.Leaf? then [Term(Util(p, n.payoff), [])] else TermsKids(n, s, p, 0)
  }

  function TermsKids(n: Node, s: Profile, p: nat, i: nat): seq<Term>
    requires !n.Leaf?
    decreases n, 1, |n.kids| - i
  {
    if i >= |n.kids| then [] else KidTerms(n, s, p, i) + TermsKids(n, s, p, i + 1)
  }

  // ---------- filters ----------

  datatype Pred = Any | IsEmpty | NonEmpty | HeadInf(I: nat) | NotHeadInf(I: nat)
                | HeadAct(a: nat) | HeadActGe(a: nat)

  predicate Holds(q: Pred, t: Term)
  {
    match q
    case Any => true
    case IsEmpty => |t.path| == 0
    case NonEmpty => |t.path| > 0
    case HeadInf(I) => |t.path| > 0 && t.path[0].inf == I
    case NotHeadInf(I) => !(|t.path| > 0 && t.path[0].inf == I)
    case HeadAct(a) => |t.path| > 0 && t.path[0].act == a
    case HeadActGe(a) => |t.path| > 0 && t.path[0].act >= a
  }

  function Filter(Z: seq<Term>, q: Pred): (r: seq<Term>)
    ensures |r| <= |Z|
    ensures forall t :: t in r ==> t in Z && Holds(q, t)
  {
    if |Z| == 0 then []
    else if Holds(q, Z[0]) then [Z[0]] + Filter(Z[1..], q)
    else Filter(Z[1..], q)
  }

  // The terms below p's sequence (I, a), with that step removed.
  function Sub(Z: seq<Term>, a: nat): seq<Term>
  {
    Strip(Filter(Z, HeadAct(a)))
  }

  ghost predicate PathsLe(Z: seq<Term>, d: nat)
  {
    forall t :: t in Z ==> |t.path| <= d
  }

  function MaxLen(Z: seq<Term>): (r: nat)
    ensures PathsLe(Z, r)
  {
    if |Z| == 0 then 0
    else
      var m := MaxLen(Z[1..]);
      assert forall t :: t in Z ==> t == Z[0] || t in Z[1..];
      if |Z[0].path| > m then |Z[0].path| else m
  }

  lemma SubLe(Z: seq<Term>, a: nat, d: nat)
    requires PathsLe(Z, d) && 1 <= d
    ensures PathsLe(Sub(Z, a), d - 1)
  {
    var F := Filter(Z, HeadAct(a));
    var r := Strip(F);
    forall t | t in r ensures |t.path| <= d - 1 {
      var j :| 0 <= j < |r| && r[j] == t;
      assert F[j] in F;
    }
  }

  function NA(nAct: seq<nat>, I: nat): nat
  {
    if I < |nAct| then nAct[I] else 0
  }

  function Max(x: real, y: real): real { if x >= y then x else y }

  // ---------- the best-response value ----------

  // V(Z): the most p can get from the terms Z, choosing per infoset.
  opaque function V(nAct: seq<nat>, Z: seq<Term>, d: nat): real
    requires PathsLe(Z, d)
    decreases d, 2, 0, 0
  {
    if |Z| == 0 then 0.0
    else WSum(Filter(Z, IsEmpty)) + VH(nAct, Filter(Z, NonEmpty), d)
  }

  lemma OutShorter(L: seq<Term>, I: nat)
    requires |L| > 0 && |L[0].path| > 0 && L[0].path[0].inf == I
    ensures |Filter(L, NotHeadInf(I))| < |L|
  {
  }

  // L: terms whose path starts with a step of p. Group them by p's first infoset I.
  opaque function VH(nAct: seq<nat>, L: seq<Term>, d: nat): real
    requires PathsLe(L, d) && forall t :: t in L ==> |t.path| > 0
    decreases d, 1, |L|, 0
  {
    if |L| == 0 then 0.0
    else
      var I := L[0].path[0].inf;
      var n := NA(nAct, I);
      OutShorter(L, I);
      assert L[0] in L;
      (if n == 0 then 0.0 else MaxV(nAct, Filter(L, HeadInf(I)), 0, n, d))
      + VH(nAct, Filter(L, NotHeadInf(I)), d)
  }

  // max over actions a' in [a, n) of V(Sub(In, a')).
  opaque function MaxV(nAct: seq<nat>, In: seq<Term>, a: nat, n: nat, d: nat): real
    requires a < n && 1 <= d && PathsLe(In, d)
    decreases d, 0, 0, n - a
  {
    SubLe(In, a, d);
    var v := V(nAct, Sub(In, a), d - 1);
    if a + 1 == n then v else Max(v, MaxV(nAct, In, a + 1, n, d))
  }

  // ---------- the pure best response (the choices behind V) ----------

  opaque function ArgMaxV(nAct: seq<nat>, In: seq<Term>, a: nat, n: nat, d: nat): (r: (nat, real))
    requires a < n && 1 <= d && PathsLe(In, d)
    decreases d, 0, 0, n - a
  {
    SubLe(In, a, d);
    var v := V(nAct, Sub(In, a), d - 1);
    if a + 1 == n then (a, v)
    else
      var b := ArgMaxV(nAct, In, a + 1, n, d);
      if v >= b.1 then (a, v) else b
  }

  opaque function PickV(nAct: seq<nat>, Z: seq<Term>, d: nat, pi: seq<nat>): seq<nat>
    requires PathsLe(Z, d)
    decreases d, 2, 0, 0
  {
    if |Z| == 0 then pi else PickH(nAct, Filter(Z, NonEmpty), d, pi)
  }

  opaque function PickH(nAct: seq<nat>, L: seq<Term>, d: nat, pi: seq<nat>): seq<nat>
    requires PathsLe(L, d) && forall t :: t in L ==> |t.path| > 0
    decreases d, 1, |L|, 0
  {
    if |L| == 0 then pi
    else
      var I := L[0].path[0].inf;
      var n := NA(nAct, I);
      OutShorter(L, I);
      assert L[0] in L;
      var In := Filter(L, HeadInf(I));
      var pi1 :=
        if n == 0 then pi
        else
          var best := ArgMaxV(nAct, In, 0, n, d).0;
          PickActs(nAct, In, 0, n, d, if I < |pi| then pi[I := best] else pi);
      PickH(nAct, Filter(L, NotHeadInf(I)), d, pi1)
  }

  opaque function PickActs(nAct: seq<nat>, In: seq<Term>, a: nat, n: nat, d: nat, pi: seq<nat>): seq<nat>
    requires a <= n && 1 <= d && PathsLe(In, d)
    decreases d, 0, 0, n - a
  {
    if a >= n then pi
    else
      SubLe(In, a, d);
      PickActs(nAct, In, a + 1, n, d, PickV(nAct, Sub(In, a), d - 1, pi))
  }
}
