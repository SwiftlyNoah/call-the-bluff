// Best response by infoset aggregation, with its proofs.
//
// The algorithm keeps a weighted list L of nodes still to expand and a list P
// of pending decision nodes of the responder p. Chance and opponent nodes are
// expanded into their children (weights multiplied by the chance or opponent
// probability). Responder nodes wait in P. When L is empty, the pending nodes
// are grouped by infoset; for the first infoset I, the value of each action a
// is the best response value of the group advanced along a, the best action is
// taken, and the rest of P is handled the same way.
//
// Theorems (proved below, no assumptions):
//   UpperBound: for every behavioral strategy b of p (an exact distribution at
//     every infoset), LVal(L, b) + LVal(P, b) <= BRV(L, P).v.
//   Attain: if a pure strategy s agrees with every (infoset, action) choice the
//     algorithm recorded, s attains BRV(L, P).v exactly.
//   ScanEq, BRGEq: Scan then BRG, the form the checker compiles and runs (Scan
//     is a loop), return exactly what BRV returns.
// Neither theorem needs perfect recall. The recorded choices must also be
// consistent (no infoset chosen twice with different actions); perfect recall
// should guarantee that, but it is not proved here (fuzz-tested only), so the
// checker tests consistency at run time and refuses the input otherwise.

include "Game.dfy"

module BestResponse {
  import opened Math
  import opened Game

  type Item = (real, Node)

  // Weighted expected utility of a list of nodes.
  ghost function LVal(L: seq<Item>, p: nat, b: Strat, sigma: Strat): real
    decreases |L|
  {
    if |L| == 0 then 0.0 else Mul(L[0].0, Val(L[0].1, p, b, sigma)) + LVal(L[1..], p, b, sigma)
  }

  ghost function M(L: seq<Item>): nat
    decreases |L|
  {
    if |L| == 0 then 0 else Size(L[0].1) + M(L[1..])
  }

  ghost predicate AllWF(L: seq<Item>, A: seq<nat>)
  {
    forall i :: 0 <= i < |L| ==> WFN(L[i].1, A)
  }

  ghost predicate AllP(L: seq<Item>, p: nat)
  {
    forall i :: 0 <= i < |L| ==> L[i].1.Decision? && L[i].1.player == p
  }

  // The children of a node, weighted by w * d[i].
  function Expand(w: real, d: seq<real>, ks: seq<Node>, i: nat): seq<Item>
    decreases |ks| - i
  {
    if i >= |ks| then [] else [(Mul(w, At(d, i)), ks[i])] + Expand(w, d, ks, i + 1)
  }

  // Pending nodes at infoset I, and the others.
  function Same(P: seq<Item>, I: nat): seq<Item>
    decreases |P|
  {
    if |P| == 0 then []
    else (if P[0].1.Decision? && P[0].1.infoset == I then [P[0]] else []) + Same(P[1..], I)
  }

  function Other(P: seq<Item>, I: nat): seq<Item>
    decreases |P|
  {
    if |P| == 0 then []
    else (if P[0].1.Decision? && P[0].1.infoset == I then [] else [P[0]]) + Other(P[1..], I)
  }

  ghost predicate AllHaveKid(G: seq<Item>, a: nat)
  {
    forall i :: 0 <= i < |G| ==> G[i].1.Decision? && a < |G[i].1.kids|
  }

  // Every node of the group advanced along action a.
  function Adv(G: seq<Item>, a: nat): seq<Item>
    requires AllHaveKid(G, a)
    decreases |G|
  {
    if |G| == 0 then [] else [(G[0].0, G[0].1.kids[a])] + Adv(G[1..], a)
  }

  datatype Res = Res(v: real, ch: seq<(nat, nat)>)

  function Argmax(rs: seq<Res>): nat
    requires |rs| > 0
    ensures Argmax(rs) < |rs|
    ensures forall i :: 0 <= i < |rs| ==> rs[i].v <= rs[Argmax(rs)].v
    decreases |rs|
  {
    if |rs| == 1 then 0
    else
      var k := Argmax(rs[1..]);
      assert forall i :: 1 <= i < |rs| ==> rs[i] == rs[1..][i - 1];
      if rs[0].v >= rs[k + 1].v then 0 else k + 1
  }

  // ---------------------------------------------------------------- list facts

  lemma LValAppend(X: seq<Item>, Y: seq<Item>, p: nat, b: Strat, sigma: Strat)
    ensures LVal(X + Y, p, b, sigma) == LVal(X, p, b, sigma) + LVal(Y, p, b, sigma)
    decreases |X|
  {
    if |X| == 0 {
      assert X + Y == Y;
    } else {
      assert (X + Y)[1..] == X[1..] + Y;
      LValAppend(X[1..], Y, p, b, sigma);
    }
  }

  lemma MAppend(X: seq<Item>, Y: seq<Item>)
    ensures M(X + Y) == M(X) + M(Y)
    decreases |X|
  {
    if |X| == 0 {
      assert X + Y == Y;
    } else {
      assert (X + Y)[1..] == X[1..] + Y;
      MAppend(X[1..], Y);
    }
  }

  lemma ExpandStep(w: real, c: real, v: real, r: real)
    ensures Mul(Mul(w, c), v) + Mul(w, r) == Mul(w, Mul(c, v) + r)
  {
    reveal Mul();
    assert w * (c * v + r) == (w * c) * v + w * r;
  }

  lemma LValExpand(w: real, d: seq<real>, ks: seq<Node>, i: nat, p: nat, b: Strat, sigma: Strat)
    ensures LVal(Expand(w, d, ks, i), p, b, sigma)
      == Mul(w, DotF(d, seq(|ks|, j requires 0 <= j < |ks| => Val(ks[j], p, b, sigma)), i))
    decreases |ks| - i
  {
    var kv := seq(|ks|, j requires 0 <= j < |ks| => Val(ks[j], p, b, sigma));
    if i < |ks| {
      LValExpand(w, d, ks, i + 1, p, b, sigma);
      var E := Expand(w, d, ks, i);
      assert E[1..] == Expand(w, d, ks, i + 1);
      ExpandStep(w, At(d, i), kv[i], DotF(d, kv, i + 1));
    } else {
      reveal Mul();
    }
  }

  lemma MExpand(w: real, d: seq<real>, ks: seq<Node>, i: nat)
    ensures M(Expand(w, d, ks, i)) == SumNF(seq(|ks|, j requires 0 <= j < |ks| => Size(ks[j])), i)
    decreases |ks| - i
  {
    if i < |ks| {
      MExpand(w, d, ks, i + 1);
      assert Expand(w, d, ks, i)[1..] == Expand(w, d, ks, i + 1);
    }
  }

  lemma ExpandWF(w: real, d: seq<real>, ks: seq<Node>, i: nat, A: seq<nat>)
    requires forall j :: 0 <= j < |ks| ==> WFN(ks[j], A)
    ensures AllWF(Expand(w, d, ks, i), A)
    decreases |ks| - i
  {
    if i < |ks| {
      ExpandWF(w, d, ks, i + 1, A);
      var E := Expand(w, d, ks, i);
      assert E[1..] == Expand(w, d, ks, i + 1);
      forall k | 0 <= k < |E| ensures WFN(E[k].1, A) {
        if k > 0 { assert E[k] == E[1..][k - 1]; }
      }
    }
  }

  // Expanding a chance or opponent node lowers the measure and keeps the list
  // well formed.
  lemma ExpandFacts(L: seq<Item>, d: seq<real>, A: seq<nat>)
    requires |L| > 0 && !L[0].1.Terminal? && AllWF(L, A)
    ensures AllWF(Expand(L[0].0, d, L[0].1.kids, 0) + L[1..], A)
    ensures M(Expand(L[0].0, d, L[0].1.kids, 0) + L[1..]) < M(L)
  {
    var h := L[0].1;
    var E := Expand(L[0].0, d, h.kids, 0);
    assert forall j :: 0 <= j < |h.kids| ==> WFN(h.kids[j], A);
    ExpandWF(L[0].0, d, h.kids, 0, A);
    MExpand(L[0].0, d, h.kids, 0);
    MAppend(E, L[1..]);
    var sz := seq(|h.kids|, j requires 0 <= j < |h.kids| => Size(h.kids[j]));
    assert Size(h) == 1 + SumNF(sz, 0);
    forall k | 0 <= k < |E + L[1..]| ensures WFN((E + L[1..])[k].1, A) {
      if k >= |E| { assert (E + L[1..])[k] == L[k - |E| + 1]; }
    }
  }

  lemma TailFacts(L: seq<Item>, A: seq<nat>)
    requires |L| > 0 && AllWF(L, A)
    ensures AllWF(L[1..], A) && M(L[1..]) < M(L)
  {
    SizePos(L[0].1);
    forall k | 0 <= k < |L[1..]| ensures WFN(L[1..][k].1, A) {
      assert L[1..][k] == L[k + 1];
    }
  }

  lemma PushFacts(L: seq<Item>, P: seq<Item>, p: nat, A: seq<nat>)
    requires |L| > 0 && AllWF(L, A) && AllWF(P, A) && AllP(P, p)
    requires L[0].1.Decision? && L[0].1.player == p
    ensures AllWF(P + [L[0]], A) && AllP(P + [L[0]], p)
    ensures M(P + [L[0]]) == M(P) + Size(L[0].1)
  {
    MAppend(P, [L[0]]);
  }

  lemma SplitFacts(P: seq<Item>, I: nat, p: nat, A: seq<nat>, b: Strat, sigma: Strat)
    requires AllWF(P, A) && AllP(P, p)
    ensures AllWF(Same(P, I), A) && AllP(Same(P, I), p)
    ensures AllWF(Other(P, I), A) && AllP(Other(P, I), p)
    ensures forall i :: 0 <= i < |Same(P, I)| ==> Same(P, I)[i].1.infoset == I
    ensures M(Same(P, I)) + M(Other(P, I)) == M(P)
    ensures LVal(Same(P, I), p, b, sigma) + LVal(Other(P, I), p, b, sigma) == LVal(P, p, b, sigma)
    ensures |P| > 0 && P[0].1.infoset == I ==> |Same(P, I)| > 0
    decreases |P|
  {
    if |P| > 0 {
      var T := P[1..];
      assert AllWF(T, A) && AllP(T, p) by {
        forall k | 0 <= k < |T| ensures WFN(T[k].1, A) && T[k].1.Decision? && T[k].1.player == p {
          assert T[k] == P[k + 1];
        }
      }
      SplitFacts(T, I, p, A, b, sigma);
      var hd := if P[0].1.Decision? && P[0].1.infoset == I then [P[0]] else [];
      var od := if P[0].1.Decision? && P[0].1.infoset == I then [] else [P[0]];
      assert Same(P, I) == hd + Same(T, I);
      assert Other(P, I) == od + Other(T, I);
      MAppend(hd, Same(T, I));
      MAppend(od, Other(T, I));
      LValAppend(hd, Same(T, I), p, b, sigma);
      LValAppend(od, Other(T, I), p, b, sigma);
      reveal Mul();
      var S := Same(P, I);
      forall k | 0 <= k < |S|
        ensures WFN(S[k].1, A) && S[k].1.Decision? && S[k].1.player == p && S[k].1.infoset == I
      {
        if |hd| == 1 && k > 0 { assert S[k] == Same(T, I)[k - 1]; }
      }
      var O := Other(P, I);
      forall k | 0 <= k < |O| ensures WFN(O[k].1, A) && O[k].1.Decision? && O[k].1.player == p {
        if |od| == 1 && k > 0 { assert O[k] == Other(T, I)[k - 1]; }
      }
    }
  }

  lemma AdvFacts(G: seq<Item>, a: nat, A: seq<nat>)
    requires AllWF(G, A) && AllHaveKid(G, a)
    ensures AllWF(Adv(G, a), A)
    ensures |G| > 0 ==> M(Adv(G, a)) < M(G)
    ensures M(Adv(G, a)) <= M(G)
    decreases |G|
  {
    if |G| > 0 {
      var T := G[1..];
      assert AllWF(T, A) && AllHaveKid(T, a) by {
        forall k | 0 <= k < |T| ensures WFN(T[k].1, A) && T[k].1.Decision? && a < |T[k].1.kids| {
          assert T[k] == G[k + 1];
        }
      }
      AdvFacts(T, a, A);
      KidSmaller(G[0].1, a);
      var X := Adv(G, a);
      assert X[1..] == Adv(T, a);
      assert WFN(G[0].1, A);
      forall k | 0 <= k < |X| ensures WFN(X[k].1, A) {
        if k > 0 { assert X[k] == Adv(T, a)[k - 1]; }
      }
    }
  }

  // Grouping a responder infoset: the group's value is the b-weighted mix of
  // its advanced lists.
  lemma LValGroup(G: seq<Item>, I: nat, n: nat, p: nat, b: Strat, sigma: Strat)
    requires forall i :: 0 <= i < |G| ==>
      G[i].1.Decision? && G[i].1.player == p && G[i].1.infoset == I && |G[i].1.kids| == n
    ensures forall a :: 0 <= a < n ==> AllHaveKid(G, a)
    ensures LVal(G, p, b, sigma)
      == Dot(Row(b, I), seq(n, a requires 0 <= a < n => LVal(Adv(G, a), p, b, sigma)))
    decreases |G|
  {
    var d := Row(b, I);
    var Y := seq(n, a requires 0 <= a < n => LVal(Adv(G, a), p, b, sigma));
    if |G| == 0 {
      DotZero(d, Y, 0);
    } else {
      var w := G[0].0;
      var h := G[0].1;
      var T := G[1..];
      assert forall i :: 0 <= i < |T| ==> T[i] == G[i + 1];
      LValGroup(T, I, n, p, b, sigma);
      var Y' := seq(n, a requires 0 <= a < n => LVal(Adv(T, a), p, b, sigma));
      var kv := seq(|h.kids|, i requires 0 <= i < |h.kids| => Val(h.kids[i], p, b, sigma));
      assert Val(h, p, b, sigma) == Dot(d, kv);
      forall a | 0 <= a < n ensures Y[a] == Mul(w, kv[a]) + Y'[a] {
        var X := Adv(G, a);
        assert X[1..] == Adv(T, a);
      }
      DotLin(d, kv, Y', Y, w, 0);
    }
  }

  // ---------------------------------------------------------------- the algorithm

  function BRV(L: seq<Item>, P: seq<Item>, p: nat, sigma: Strat, A: seq<nat>): Res
    requires AllWF(L, A) && AllWF(P, A) && AllP(P, p)
    decreases M(L) + M(P), 1, |L|
  {
    if |L| > 0 then
      var w := L[0].0;
      var h := L[0].1;
      TailFacts(L, A);
      if w == 0.0 then
        BRV(L[1..], P, p, sigma, A)
      else
        match h
        case Terminal(x) =>
          var r := BRV(L[1..], P, p, sigma, A);
          Res(Mul(w, U(p, x)) + r.v, r.ch)
        case Chance(ps, ks) =>
          ExpandFacts(L, ps, A);
          BRV(Expand(w, ps, ks, 0) + L[1..], P, p, sigma, A)
        case Decision(q, I, ks) =>
          if q == p then
            PushFacts(L, P, p, A);
            BRV(L[1..], P + [L[0]], p, sigma, A)
          else
            ExpandFacts(L, Row(sigma, I), A);
            BRV(Expand(w, Row(sigma, I), ks, 0) + L[1..], P, p, sigma, A)
    else if |P| == 0 then
      Res(0.0, [])
    else
      var I := P[0].1.infoset;
      var G := Same(P, I);
      var R := Other(P, I);
      SplitFacts(P, I, p, A, [], sigma);
      assert WFN(P[0].1, A);
      var n := A[I];
      assert forall i :: 0 <= i < |G| ==> WFN(G[i].1, A) && |G[i].1.kids| == n;
      var rs := Branches(G, I, n, p, sigma, A);
      var k := Argmax(rs);
      var rest := BRV([], R, p, sigma, A);
      Res(rs[k].v + rest.v, [(I, k)] + rs[k].ch + rest.ch)
  }

  ghost predicate GroupOK(G: seq<Item>, I: nat, n: nat, p: nat, A: seq<nat>)
  {
    |G| > 0 && n >= 1 && AllWF(G, A) && AllP(G, p)
    && forall i :: 0 <= i < |G| ==> G[i].1.infoset == I && |G[i].1.kids| == n
  }

  function Branches(G: seq<Item>, I: nat, n: nat, p: nat, sigma: Strat, A: seq<nat>): seq<Res>
    requires GroupOK(G, I, n, p, A)
    ensures |Branches(G, I, n, p, sigma, A)| == n
    decreases M(G), 0, 0
  {
    assert forall a :: 0 <= a < n ==> AllHaveKid(G, a);
    assert forall a :: 0 <= a < n ==> AllWF(Adv(G, a), A) && M(Adv(G, a)) < M(G) by {
      forall a | 0 <= a < n ensures AllWF(Adv(G, a), A) && M(Adv(G, a)) < M(G) {
        AdvFacts(G, a, A);
      }
    }
    seq(n, a requires 0 <= a < n => BRV(Adv(G, a), [], p, sigma, A))
  }

  // ---------------------------------------------------------------- the compiled form

  // What the checker runs, proved equal to BRV (ScanEq, BRGEq, BranchesEq).
  // BRV needs one stack frame per list item, so the compiled BRV ran out of
  // stack on a chance node with about a million outcomes. Scan does BRV's work
  // on L (terminals, chance and opponent nodes, pushing responder nodes onto P)
  // and calls only itself, each time in tail position, so Dafny compiles it to
  // a loop. BRG then groups the pending nodes by infoset as BRV does; its
  // recursion depth is the number of responder levels and pending infosets,
  // not the number of nodes.
  function Scan(L: seq<Item>, P: seq<Item>, p: nat, sigma: Strat, A: seq<nat>, v: real)
    : (r: (seq<Item>, real))
    requires AllWF(L, A) && AllWF(P, A) && AllP(P, p)
    ensures AllWF(r.0, A) && AllP(r.0, p) && M(r.0) <= M(L) + M(P)
    decreases M(L) + M(P), |L|
  {
    if |L| == 0 then (P, v)
    else
      var w := L[0].0;
      var h := L[0].1;
      TailFacts(L, A);
      if w == 0.0 then
        Scan(L[1..], P, p, sigma, A, v)
      else
        match h
        case Terminal(x) =>
          Scan(L[1..], P, p, sigma, A, v + Mul(w, U(p, x)))
        case Chance(ps, ks) =>
          ExpandFacts(L, ps, A);
          Scan(Expand(w, ps, ks, 0) + L[1..], P, p, sigma, A, v)
        case Decision(q, I, ks) =>
          if q == p then
            PushFacts(L, P, p, A);
            MAppend(L[..1], L[1..]);
            assert L == L[..1] + L[1..];
            Scan(L[1..], P + [L[0]], p, sigma, A, v)
          else
            ExpandFacts(L, Row(sigma, I), A);
            Scan(Expand(w, Row(sigma, I), ks, 0) + L[1..], P, p, sigma, A, v)
  }

  function BRG(P: seq<Item>, p: nat, sigma: Strat, A: seq<nat>, v: real, ch: seq<(nat, nat)>): Res
    requires AllWF(P, A) && AllP(P, p)
    decreases M(P), 1
  {
    if |P| == 0 then Res(v, ch)
    else
      var I := P[0].1.infoset;
      var G := Same(P, I);
      var R := Other(P, I);
      SplitFacts(P, I, p, A, [], sigma);
      assert WFN(P[0].1, A);
      var n := A[I];
      assert forall i :: 0 <= i < |G| ==> WFN(G[i].1, A) && |G[i].1.kids| == n;
      SizePos(P[0].1);
      MAppend(G[..1], G[1..]);
      assert G == G[..1] + G[1..];
      var rs := BranchesA(G, I, n, p, sigma, A);
      var k := Argmax(rs);
      BRG(R, p, sigma, A, v + rs[k].v, ch + ([(I, k)] + rs[k].ch))
  }

  function BranchesA(G: seq<Item>, I: nat, n: nat, p: nat, sigma: Strat, A: seq<nat>): seq<Res>
    requires GroupOK(G, I, n, p, A)
    ensures |BranchesA(G, I, n, p, sigma, A)| == n
    decreases M(G), 0
  {
    assert forall a :: 0 <= a < n ==> AllHaveKid(G, a);
    assert forall a :: 0 <= a < n ==> AllWF(Adv(G, a), A) && M(Adv(G, a)) < M(G) by {
      forall a | 0 <= a < n ensures AllWF(Adv(G, a), A) && M(Adv(G, a)) < M(G) {
        AdvFacts(G, a, A);
      }
    }
    seq(n, a requires 0 <= a < n =>
      var s := Scan(Adv(G, a), [], p, sigma, A, 0.0);
      BRG(s.0, p, sigma, A, s.1, []))
  }

  lemma ScanEq(L: seq<Item>, P: seq<Item>, p: nat, sigma: Strat, A: seq<nat>, v: real)
    requires AllWF(L, A) && AllWF(P, A) && AllP(P, p)
    ensures var r := Scan(L, P, p, sigma, A, v);
      v + BRV(L, P, p, sigma, A).v == r.1 + BRV([], r.0, p, sigma, A).v
      && BRV(L, P, p, sigma, A).ch == BRV([], r.0, p, sigma, A).ch
    decreases M(L) + M(P), |L|
  {
    if |L| > 0 {
      var w := L[0].0;
      var h := L[0].1;
      TailFacts(L, A);
      if w == 0.0 {
        ScanEq(L[1..], P, p, sigma, A, v);
      } else {
        match h
        case Terminal(x) =>
          ScanEq(L[1..], P, p, sigma, A, v + Mul(w, U(p, x)));
        case Chance(ps, ks) =>
          ExpandFacts(L, ps, A);
          ScanEq(Expand(w, ps, ks, 0) + L[1..], P, p, sigma, A, v);
        case Decision(q, I, ks) =>
          if q == p {
            PushFacts(L, P, p, A);
            MAppend(L[..1], L[1..]);
            assert L == L[..1] + L[1..];
            ScanEq(L[1..], P + [L[0]], p, sigma, A, v);
          } else {
            ExpandFacts(L, Row(sigma, I), A);
            ScanEq(Expand(w, Row(sigma, I), ks, 0) + L[1..], P, p, sigma, A, v);
          }
      }
    }
  }

  lemma BRGEq(P: seq<Item>, p: nat, sigma: Strat, A: seq<nat>, v: real, ch: seq<(nat, nat)>)
    requires AllWF(P, A) && AllP(P, p)
    ensures BRG(P, p, sigma, A, v, ch)
      == Res(v + BRV([], P, p, sigma, A).v, ch + BRV([], P, p, sigma, A).ch)
    decreases M(P), 1
  {
    if |P| == 0 {
      assert ch + [] == ch;
    } else {
      var I := P[0].1.infoset;
      var G := Same(P, I);
      var R := Other(P, I);
      SplitFacts(P, I, p, A, [], sigma);
      assert WFN(P[0].1, A);
      var n := A[I];
      assert forall i :: 0 <= i < |G| ==> WFN(G[i].1, A) && |G[i].1.kids| == n;
      SizePos(P[0].1);
      MAppend(G[..1], G[1..]);
      assert G == G[..1] + G[1..];
      BranchesEq(G, I, n, p, sigma, A);
      var rs := Branches(G, I, n, p, sigma, A);
      var k := Argmax(rs);
      var c := [(I, k)] + rs[k].ch;
      BRGEq(R, p, sigma, A, v + rs[k].v, ch + c);
      var rest := BRV([], R, p, sigma, A);
      assert BRV([], P, p, sigma, A) == Res(rs[k].v + rest.v, [(I, k)] + rs[k].ch + rest.ch);
      assert (ch + c) + rest.ch == ch + ([(I, k)] + rs[k].ch + rest.ch);
    }
  }

  lemma BranchesEq(G: seq<Item>, I: nat, n: nat, p: nat, sigma: Strat, A: seq<nat>)
    requires GroupOK(G, I, n, p, A)
    ensures BranchesA(G, I, n, p, sigma, A) == Branches(G, I, n, p, sigma, A)
    decreases M(G), 0
  {
    assert forall a :: 0 <= a < n ==> AllHaveKid(G, a);
    var X := BranchesA(G, I, n, p, sigma, A);
    var Y := Branches(G, I, n, p, sigma, A);
    forall a | 0 <= a < n ensures X[a] == Y[a] {
      AdvFacts(G, a, A);
      var s := Scan(Adv(G, a), [], p, sigma, A, 0.0);
      ScanEq(Adv(G, a), [], p, sigma, A, 0.0);
      BRGEq(s.0, p, sigma, A, s.1, []);
      var r := BRV(Adv(G, a), [], p, sigma, A);
      assert [] + r.ch == r.ch;
    }
  }

  // ---------------------------------------------------------------- theorems

  lemma UpperBound(L: seq<Item>, P: seq<Item>, p: nat, b: Strat, sigma: Strat, A: seq<nat>)
    requires AllWF(L, A) && AllWF(P, A) && AllP(P, p) && ValidBeh(b, A)
    ensures LVal(L, p, b, sigma) + LVal(P, p, b, sigma) <= BRV(L, P, p, sigma, A).v
    decreases M(L) + M(P), 1, |L|
  {
    if |L| > 0 {
      var w := L[0].0;
      var h := L[0].1;
      TailFacts(L, A);
      if w == 0.0 {
        UpperBound(L[1..], P, p, b, sigma, A);
        reveal Mul();
      } else {
        match h
        case Terminal(x) =>
          UpperBound(L[1..], P, p, b, sigma, A);
        case Chance(ps, ks) =>
          ExpandFacts(L, ps, A);
          var E := Expand(w, ps, ks, 0);
          UpperBound(E + L[1..], P, p, b, sigma, A);
          LValAppend(E, L[1..], p, b, sigma);
          LValExpand(w, ps, ks, 0, p, b, sigma);
        case Decision(q, I, ks) =>
          if q == p {
            PushFacts(L, P, p, A);
            UpperBound(L[1..], P + [L[0]], p, b, sigma, A);
            LValAppend(P, [L[0]], p, b, sigma);
            reveal Mul();
          } else {
            ExpandFacts(L, Row(sigma, I), A);
            var E := Expand(w, Row(sigma, I), ks, 0);
            UpperBound(E + L[1..], P, p, b, sigma, A);
            LValAppend(E, L[1..], p, b, sigma);
            LValExpand(w, Row(sigma, I), ks, 0, p, b, sigma);
          }
      }
    } else if |P| > 0 {
      var I := P[0].1.infoset;
      var G := Same(P, I);
      var R := Other(P, I);
      SplitFacts(P, I, p, A, b, sigma);
      SplitFacts(P, I, p, A, [], sigma);
      assert WFN(P[0].1, A);
      var n := A[I];
      assert forall i :: 0 <= i < |G| ==> WFN(G[i].1, A) && |G[i].1.kids| == n;
      var rs := Branches(G, I, n, p, sigma, A);
      var k := Argmax(rs);
      assert BRV(L, P, p, sigma, A).v == rs[k].v + BRV([], R, p, sigma, A).v;
      UpperBound([], R, p, b, sigma, A);
      LValGroup(G, I, n, p, b, sigma);
      var d := Row(b, I);
      var Y := seq(n, a requires 0 <= a < n => LVal(Adv(G, a), p, b, sigma));
      var Z := seq(n, a requires 0 <= a < n => rs[a].v);
      forall a | 0 <= a < n ensures Y[a] <= Z[a] {
        AdvFacts(G, a, A);
        UpperBound(Adv(G, a), [], p, b, sigma, A);
      }
      assert ValidRow(d, n);
      DotMono(d, Y, Z, 0);
      DotBound(d, Z, rs[k].v, 0);
    }
  }

  ghost predicate Agrees(s: nat -> nat, ch: seq<(nat, nat)>)
  {
    forall j :: 0 <= j < |ch| ==> s(ch[j].0) == ch[j].1
  }

  lemma Attain(L: seq<Item>, P: seq<Item>, p: nat, s: nat -> nat, sigma: Strat, A: seq<nat>)
    requires AllWF(L, A) && AllWF(P, A) && AllP(P, p) && ValidPure(s, A)
    requires Agrees(s, BRV(L, P, p, sigma, A).ch)
    ensures LVal(L, p, OneHot(s, A), sigma) + LVal(P, p, OneHot(s, A), sigma)
      == BRV(L, P, p, sigma, A).v
    decreases M(L) + M(P), 1, |L|
  {
    var b := OneHot(s, A);
    if |L| > 0 {
      var w := L[0].0;
      var h := L[0].1;
      TailFacts(L, A);
      if w == 0.0 {
        Attain(L[1..], P, p, s, sigma, A);
        reveal Mul();
      } else {
        match h
        case Terminal(x) =>
          Attain(L[1..], P, p, s, sigma, A);
        case Chance(ps, ks) =>
          ExpandFacts(L, ps, A);
          var E := Expand(w, ps, ks, 0);
          Attain(E + L[1..], P, p, s, sigma, A);
          LValAppend(E, L[1..], p, b, sigma);
          LValExpand(w, ps, ks, 0, p, b, sigma);
        case Decision(q, I, ks) =>
          if q == p {
            PushFacts(L, P, p, A);
            Attain(L[1..], P + [L[0]], p, s, sigma, A);
            LValAppend(P, [L[0]], p, b, sigma);
            reveal Mul();
          } else {
            ExpandFacts(L, Row(sigma, I), A);
            var E := Expand(w, Row(sigma, I), ks, 0);
            Attain(E + L[1..], P, p, s, sigma, A);
            LValAppend(E, L[1..], p, b, sigma);
            LValExpand(w, Row(sigma, I), ks, 0, p, b, sigma);
          }
      }
    } else if |P| > 0 {
      var I := P[0].1.infoset;
      var G := Same(P, I);
      var R := Other(P, I);
      SplitFacts(P, I, p, A, b, sigma);
      SplitFacts(P, I, p, A, [], sigma);
      assert WFN(P[0].1, A);
      var n := A[I];
      assert forall i :: 0 <= i < |G| ==> WFN(G[i].1, A) && |G[i].1.kids| == n;
      var rs := Branches(G, I, n, p, sigma, A);
      var k := Argmax(rs);
      var rest := BRV([], R, p, sigma, A);
      var ch := [(I, k)] + rs[k].ch + rest.ch;
      assert BRV(L, P, p, sigma, A) == Res(rs[k].v + rest.v, ch);
      assert s(I) == k by { assert ch[0] == (I, k); }
      assert Agrees(s, rs[k].ch) by {
        forall j | 0 <= j < |rs[k].ch| ensures s(rs[k].ch[j].0) == rs[k].ch[j].1 {
          assert ch[1 + j] == rs[k].ch[j];
        }
      }
      assert Agrees(s, rest.ch) by {
        forall j | 0 <= j < |rest.ch| ensures s(rest.ch[j].0) == rest.ch[j].1 {
          assert ch[1 + |rs[k].ch| + j] == rest.ch[j];
        }
      }
      Attain([], R, p, s, sigma, A);
      LValGroup(G, I, n, p, b, sigma);
      var d := Row(b, I);
      var Y := seq(n, a requires 0 <= a < n => LVal(Adv(G, a), p, b, sigma));
      AdvFacts(G, k, A);
      Attain(Adv(G, k), [], p, s, sigma, A);
      assert d == OneHotRow(s(I), A[I]);
      DotOneHot(d, Y, k, 0);
    }
  }
}
