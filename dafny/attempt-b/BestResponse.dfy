// BestResponse.dfy: a best-response computation for player p against the opponent's behavioral
// strategy sigma, and the proof that its value is an upper bound on p's expected utility for
// EVERY behavioral strategy of p (hence every pure one).
//
// The algorithm works on weighted node lists. P holds pending nodes (not yet classified); R holds
// p's decision nodes that are ready. Chance and opponent nodes in P are expanded in place
// (weights multiplied by probabilities), terminals are cashed in, p's decision nodes move to R.
// When P is empty, R's nodes at the infoset of R[0] are decided together: one action for the
// whole infoset, chosen to maximize the best-response value of the successors. Because every node
// of an infoset in one group shares a single action, the upper bound holds in ANY game; with
// perfect recall the returned pure strategy also attains it, which Check verifies at run time.

include "Model.dfy"

module BestResponse {
  import opened GameModel

  datatype WN = WN(node: nat, w: real)   // a node with a weight (opponent-and-chance reach)

  predicate NodesIn(g: Game, S: seq<WN>)
  {
    forall j | 0 <= j < |S| :: S[j].node < |g.nodes|
  }

  predicate AllPDec(g: Game, p: nat, S: seq<WN>)
  {
    NodesIn(g, S)
    && forall j | 0 <= j < |S| :: g.nodes[S[j].node].Decision? && g.nodes[S[j].node].player == p
  }

  predicate AllAt(g: Game, p: nat, S: seq<WN>, I: nat)
  {
    AllPDec(g, p, S) && forall j | 0 <= j < |S| :: g.nodes[S[j].node].infoset == I
  }

  // ---------- termination measure: total subtree size of a node list (ghost) ----------

  ghost function Size(g: Game, n: nat): nat
    requires WF(g) && n < |g.nodes|
    decreases g.height[n], 1, 0
  {
    if g.nodes[n].Terminal? then 1 else 1 + SizeKids(g, n, 0)
  }

  ghost function SizeKids(g: Game, n: nat, i: nat): nat
    requires WF(g) && n < |g.nodes| && !g.nodes[n].Terminal? && i <= |g.nodes[n].kids|
    decreases g.height[n], 0, |g.nodes[n].kids| - i
  {
    if i == |g.nodes[n].kids| then 0
    else KidFacts(g, n, i); Size(g, g.nodes[n].kids[i]) + SizeKids(g, n, i + 1)
  }

  ghost function Meas(g: Game, S: seq<WN>): nat
    requires WF(g) && NodesIn(g, S)
  {
    if |S| == 0 then 0 else Size(g, S[0].node) + Meas(g, S[1..])
  }

  lemma MeasCat(g: Game, A: seq<WN>, B: seq<WN>)
    requires WF(g) && NodesIn(g, A) && NodesIn(g, B)
    ensures NodesIn(g, A + B) && Meas(g, A + B) == Meas(g, A) + Meas(g, B)
  {
    if |A| > 0 {
      assert (A + B)[1..] == A[1..] + B;
      MeasCat(g, A[1..], B);
    } else {
      assert A + B == B;
    }
  }

  lemma SizeKidsGe(g: Game, n: nat, i: nat, a: nat)
    requires WF(g) && n < |g.nodes| && !g.nodes[n].Terminal? && i <= a < |g.nodes[n].kids|
    ensures g.nodes[n].kids[a] < |g.nodes| && Size(g, g.nodes[n].kids[a]) <= SizeKids(g, n, i)
    decreases a - i
  {
    KidFacts(g, n, a);
    if i < a { SizeKidsGe(g, n, i + 1, a); }
  }

  // ---------- list operations used by the algorithm ----------

  // Children of n with weights w * ws[j], for j >= i.
  function Expand(g: Game, n: nat, ws: seq<real>, w: real, i: nat): seq<WN>
    requires WF(g) && n < |g.nodes| && !g.nodes[n].Terminal?
    requires |ws| == |g.nodes[n].kids| && i <= |ws|
    ensures NodesIn(g, Expand(g, n, ws, w, i))
    decreases |ws| - i
  {
    if i == |ws| then []
    else KidFacts(g, n, i); [WN(g.nodes[n].kids[i], w * ws[i])] + Expand(g, n, ws, w, i + 1)
  }

  lemma ExpandFacts(g: Game, n: nat, ws: seq<real>, w: real, i: nat)
    requires WF(g) && n < |g.nodes| && !g.nodes[n].Terminal?
    requires |ws| == |g.nodes[n].kids| && i <= |ws|
    ensures NodesIn(g, Expand(g, n, ws, w, i))
    ensures Meas(g, Expand(g, n, ws, w, i)) == SizeKids(g, n, i)
    decreases |ws| - i
  {
    if i < |ws| {
      KidFacts(g, n, i);
      ExpandFacts(g, n, ws, w, i + 1);
      var E := Expand(g, n, ws, w, i);
      assert E[1..] == Expand(g, n, ws, w, i + 1);
    }
  }

  // The successors of a group of p's nodes at infoset I after action a.
  function Succ(g: Game, p: nat, S: seq<WN>, I: nat, a: nat): seq<WN>
    requires WF(g) && AllAt(g, p, S, I) && I < |g.infosetActions| && a < g.infosetActions[I]
    ensures |Succ(g, p, S, I, a)| == |S| && NodesIn(g, Succ(g, p, S, I, a))
  {
    if |S| == 0 then []
    else
      WFNode(g, S[0].node);
      KidFacts(g, S[0].node, a);
      [WN(g.nodes[S[0].node].kids[a], S[0].w)] + Succ(g, p, S[1..], I, a)
  }

  lemma SuccFacts(g: Game, p: nat, S: seq<WN>, I: nat, a: nat)
    requires WF(g) && AllAt(g, p, S, I) && I < |g.infosetActions| && a < g.infosetActions[I]
    ensures NodesIn(g, Succ(g, p, S, I, a))
    ensures Meas(g, Succ(g, p, S, I, a)) + |S| <= Meas(g, S)
  {
    if |S| > 0 {
      var n := S[0].node;
      WFNode(g, n);
      SizeKidsGe(g, n, 0, a);
      SuccFacts(g, p, S[1..], I, a);
      var X := Succ(g, p, S, I, a);
      assert X[1..] == Succ(g, p, S[1..], I, a);
    }
  }

  function FilterIn(g: Game, p: nat, S: seq<WN>, I: nat): seq<WN>
    requires AllPDec(g, p, S)
  {
    if |S| == 0 then []
    else if g.nodes[S[0].node].infoset == I then [S[0]] + FilterIn(g, p, S[1..], I)
    else FilterIn(g, p, S[1..], I)
  }

  function FilterOut(g: Game, p: nat, S: seq<WN>, I: nat): seq<WN>
    requires AllPDec(g, p, S)
  {
    if |S| == 0 then []
    else if g.nodes[S[0].node].infoset != I then [S[0]] + FilterOut(g, p, S[1..], I)
    else FilterOut(g, p, S[1..], I)
  }

  lemma FilterFacts(g: Game, p: nat, S: seq<WN>, I: nat)
    requires WF(g) && AllPDec(g, p, S)
    ensures AllAt(g, p, FilterIn(g, p, S, I), I)
    ensures AllPDec(g, p, FilterOut(g, p, S, I))
    ensures Meas(g, FilterIn(g, p, S, I)) + Meas(g, FilterOut(g, p, S, I)) == Meas(g, S)
    ensures |S| > 0 && g.nodes[S[0].node].infoset == I ==> |FilterIn(g, p, S, I)| > 0
  {
    if |S| > 0 {
      FilterFacts(g, p, S[1..], I);
      var A := FilterIn(g, p, S, I);
      var B := FilterOut(g, p, S, I);
      if g.nodes[S[0].node].infoset == I {
        assert A[1..] == FilterIn(g, p, S[1..], I);
      } else {
        assert B[1..] == FilterOut(g, p, S[1..], I);
      }
    }
  }

  // ---------- the best-response algorithm (compiled) ----------

  // Returns (value, choices): the best-response value of the weighted lists P and R, and the
  // infoset -> action choices that the algorithm made.
  opaque function BR(g: Game, p: nat, sigma: Strat, P: seq<WN>, R: seq<WN>): (real, map<nat, nat>)
    requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma) && NodesIn(g, P) && AllPDec(g, p, R)
    decreases Meas(g, P) + Meas(g, R), |P|, 1, 0
  {
    if |P| > 0 then
      var h := P[0];
      WFNode(g, h.node);
      assert Meas(g, P) == Size(g, h.node) + Meas(g, P[1..]);
      match g.nodes[h.node]
      case Terminal(u) =>
        var r := BR(g, p, sigma, P[1..], R);
        (h.w * U(p, u) + r.0, r.1)
      case Chance(pr, kids) =>
        var E := Expand(g, h.node, pr, h.w, 0);
        ExpandFacts(g, h.node, pr, h.w, 0);
        MeasCat(g, E, P[1..]);
        BR(g, p, sigma, E + P[1..], R)
      case Decision(q, I, kids) =>
        if q == p then
          MeasCat(g, R, [h]);
          BR(g, p, sigma, P[1..], R + [h])
        else
          var E := Expand(g, h.node, sigma[I], h.w, 0);
          ExpandFacts(g, h.node, sigma[I], h.w, 0);
          MeasCat(g, E, P[1..]);
          BR(g, p, sigma, E + P[1..], R)
    else if |R| == 0 then (0.0, map[])
    else
      var I := g.nodes[R[0].node].infoset;
      WFNode(g, R[0].node);
      WFInfoset(g, I);
      var SI := FilterIn(g, p, R, I);
      var rest := FilterOut(g, p, R, I);
      FilterFacts(g, p, R, I);
      var best := Best(g, p, sigma, SI, I, 0);
      var r := BR(g, p, sigma, [], rest);
      (best.0 + r.0, (r.1 + best.2)[I := best.1])
  }

  // Maximizes over actions a' in [a, k) the best-response value of Succ(SI, a').
  // Returns (value, action, choices of that branch).
  function Best(g: Game, p: nat, sigma: Strat, SI: seq<WN>, I: nat, a: nat): (real, nat, map<nat, nat>)
    requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma) && AllAt(g, p, SI, I) && |SI| > 0
    requires I < |g.infosetActions| && a < g.infosetActions[I]
    decreases Meas(g, SI), 0, 0, g.infosetActions[I] - a
  {
    SuccFacts(g, p, SI, I, a);
    var c := BR(g, p, sigma, Succ(g, p, SI, I, a), []);
    if a + 1 == g.infosetActions[I] then (c.0, a, c.1)
    else
      var r := Best(g, p, sigma, SI, I, a + 1);
      if c.0 >= r.0 then (c.0, a, c.1) else r
  }

  // ---------- the value of a weighted list under a strategy of p ----------

  // VS(S) = sum over j of S[j].w * Val(S[j].node). Opaque: unfolded only through VSHead.
  opaque function VS(g: Game, p: nat, beta: Strat, sigma: Strat, S: seq<WN>): real
    requires Inv(g, p, beta, sigma) && NodesIn(g, S)
  {
    if |S| == 0 then 0.0 else S[0].w * Val(g, p, beta, sigma, S[0].node) + VS(g, p, beta, sigma, S[1..])
  }

  lemma VSNil(g: Game, p: nat, beta: Strat, sigma: Strat, S: seq<WN>)
    requires Inv(g, p, beta, sigma) && NodesIn(g, S) && |S| == 0
    ensures VS(g, p, beta, sigma, S) == 0.0
  {
    reveal VS();
  }

  lemma VSHead(g: Game, p: nat, beta: Strat, sigma: Strat, S: seq<WN>)
    requires Inv(g, p, beta, sigma) && NodesIn(g, S) && |S| > 0
    ensures VS(g, p, beta, sigma, S)
         == S[0].w * Val(g, p, beta, sigma, S[0].node) + VS(g, p, beta, sigma, S[1..])
  {
    reveal VS();
  }

  lemma VSCons(g: Game, p: nat, beta: Strat, sigma: Strat, x: WN, T: seq<WN>)
    requires Inv(g, p, beta, sigma) && x.node < |g.nodes| && NodesIn(g, T)
    ensures NodesIn(g, [x] + T)
    ensures VS(g, p, beta, sigma, [x] + T) == x.w * Val(g, p, beta, sigma, x.node) + VS(g, p, beta, sigma, T)
  {
    assert ([x] + T)[1..] == T;
    VSHead(g, p, beta, sigma, [x] + T);
  }

  lemma VSCat(g: Game, p: nat, beta: Strat, sigma: Strat, A: seq<WN>, B: seq<WN>)
    requires Inv(g, p, beta, sigma) && NodesIn(g, A) && NodesIn(g, B)
    ensures NodesIn(g, A + B)
    ensures VS(g, p, beta, sigma, A + B) == VS(g, p, beta, sigma, A) + VS(g, p, beta, sigma, B)
  {
    if |A| > 0 {
      assert A + B == [A[0]] + (A[1..] + B);
      assert A == [A[0]] + A[1..];
      VSCat(g, p, beta, sigma, A[1..], B);
      VSCons(g, p, beta, sigma, A[0], A[1..] + B);
      VSCons(g, p, beta, sigma, A[0], A[1..]);
    } else {
      assert A + B == B;
      VSNil(g, p, beta, sigma, A);
    }
  }

  // small arithmetic facts, kept out of the big contexts
  lemma Distrib3(w: real, c: real, v: real, rest: real)
    ensures (w * c) * v + w * rest == w * (c * v + rest)
  {}

  lemma Distrib5(c: real, w: real, v: real, xs: real, sk: real, r1: real)
    ensures c * (w * v + xs) + (w * sk + r1) == w * (c * v + sk) + (c * xs + r1)
  {}

  lemma MulLe(c: real, x: real, y: real)
    requires c >= 0.0 && x <= y
    ensures c * x <= c * y
  {}

  lemma BoundStep(c: real, x: real, r: real, M: real, t: real)
    requires c >= 0.0 && x <= M && r <= M * t
    ensures c * x + r <= M * (c + t)
  {
    MulLe(c, x, M);
  }

  lemma ValInner(g: Game, p: nat, beta: Strat, sigma: Strat, n: nat)
    requires Inv(g, p, beta, sigma) && n < |g.nodes| && !g.nodes[n].Terminal?
    ensures Val(g, p, beta, sigma, n) == SumKids(g, p, beta, sigma, n, Row(g, p, beta, sigma, n), 0)
  {}

  lemma VSExpand(g: Game, p: nat, beta: Strat, sigma: Strat, n: nat, ws: seq<real>, w: real, i: nat)
    requires Inv(g, p, beta, sigma) && n < |g.nodes| && !g.nodes[n].Terminal?
    requires |ws| == |g.nodes[n].kids| && i <= |ws|
    ensures NodesIn(g, Expand(g, n, ws, w, i))
    ensures VS(g, p, beta, sigma, Expand(g, n, ws, w, i)) == w * SumKids(g, p, beta, sigma, n, ws, i)
    decreases |ws| - i
  {
    ExpandFacts(g, n, ws, w, i);
    if i < |ws| {
      KidFacts(g, n, i);
      VSExpand(g, p, beta, sigma, n, ws, w, i + 1);
      var k := g.nodes[n].kids[i];
      var T := Expand(g, n, ws, w, i + 1);
      assert Expand(g, n, ws, w, i) == [WN(k, w * ws[i])] + T;
      VSCons(g, p, beta, sigma, WN(k, w * ws[i]), T);
      var v := Val(g, p, beta, sigma, k);
      var rest := SumKids(g, p, beta, sigma, n, ws, i + 1);
      assert SumKids(g, p, beta, sigma, n, ws, i) == ws[i] * v + rest;
      Distrib3(w, ws[i], v, rest);
    } else {
      VSNil(g, p, beta, sigma, Expand(g, n, ws, w, i));
    }
  }

  lemma VSFilter(g: Game, p: nat, beta: Strat, sigma: Strat, S: seq<WN>, I: nat)
    requires Inv(g, p, beta, sigma) && AllPDec(g, p, S)
    ensures AllAt(g, p, FilterIn(g, p, S, I), I) && AllPDec(g, p, FilterOut(g, p, S, I))
    ensures VS(g, p, beta, sigma, FilterIn(g, p, S, I)) + VS(g, p, beta, sigma, FilterOut(g, p, S, I))
         == VS(g, p, beta, sigma, S)
  {
    FilterFacts(g, p, S, I);
    if |S| > 0 {
      FilterFacts(g, p, S[1..], I);
      VSFilter(g, p, beta, sigma, S[1..], I);
      VSHead(g, p, beta, sigma, S);
      var A' := FilterIn(g, p, S[1..], I);
      var B' := FilterOut(g, p, S[1..], I);
      if g.nodes[S[0].node].infoset == I {
        assert FilterIn(g, p, S, I) == [S[0]] + A';
        assert FilterOut(g, p, S, I) == B';
        VSCons(g, p, beta, sigma, S[0], A');
        assert VS(g, p, beta, sigma, FilterIn(g, p, S, I))
            == S[0].w * Val(g, p, beta, sigma, S[0].node) + VS(g, p, beta, sigma, A');
      } else {
        assert FilterOut(g, p, S, I) == [S[0]] + B';
        assert FilterIn(g, p, S, I) == A';
        VSCons(g, p, beta, sigma, S[0], B');
        assert VS(g, p, beta, sigma, FilterOut(g, p, S, I))
            == S[0].w * Val(g, p, beta, sigma, S[0].node) + VS(g, p, beta, sigma, B');
      }
    } else {
      VSNil(g, p, beta, sigma, S);
      VSNil(g, p, beta, sigma, FilterIn(g, p, S, I));
      VSNil(g, p, beta, sigma, FilterOut(g, p, S, I));
    }
  }

  // ASum(b, S, a) = sum over a <= a' < |b| of b[a'] * VS(Succ(S, a')). Opaque.
  opaque function ASum(g: Game, p: nat, beta: Strat, sigma: Strat, b: seq<real>, S: seq<WN>, I: nat, a: nat): real
    requires Inv(g, p, beta, sigma) && AllAt(g, p, S, I) && I < |g.infosetActions|
    requires |b| == g.infosetActions[I] && a <= |b|
    decreases |b| - a
  {
    if a == |b| then 0.0
    else b[a] * VS(g, p, beta, sigma, Succ(g, p, S, I, a)) + ASum(g, p, beta, sigma, b, S, I, a + 1)
  }

  lemma ASumUnfold(g: Game, p: nat, beta: Strat, sigma: Strat, b: seq<real>, S: seq<WN>, I: nat, a: nat)
    requires Inv(g, p, beta, sigma) && AllAt(g, p, S, I) && I < |g.infosetActions|
    requires |b| == g.infosetActions[I] && a <= |b|
    ensures a == |b| ==> ASum(g, p, beta, sigma, b, S, I, a) == 0.0
    ensures a < |b| ==>
      (ASum(g, p, beta, sigma, b, S, I, a)
       == b[a] * VS(g, p, beta, sigma, Succ(g, p, S, I, a)) + ASum(g, p, beta, sigma, b, S, I, a + 1))
  {
    reveal ASum();
  }

  lemma ASumCons(g: Game, p: nat, beta: Strat, sigma: Strat, b: seq<real>, S: seq<WN>, I: nat, a: nat)
    requires Inv(g, p, beta, sigma) && AllAt(g, p, S, I) && I < |g.infosetActions|
    requires |b| == g.infosetActions[I] && a <= |b| && |S| > 0
    ensures |b| == |g.nodes[S[0].node].kids|
    ensures ASum(g, p, beta, sigma, b, S, I, a)
         == S[0].w * SumKids(g, p, beta, sigma, S[0].node, b, a) + ASum(g, p, beta, sigma, b, S[1..], I, a)
    decreases |b| - a
  {
    var n := S[0].node;
    WFNode(g, n);
    ASumUnfold(g, p, beta, sigma, b, S, I, a);
    ASumUnfold(g, p, beta, sigma, b, S[1..], I, a);
    if a < |b| {
      ASumCons(g, p, beta, sigma, b, S, I, a + 1);
      KidFacts(g, n, a);
      var k := g.nodes[n].kids[a];
      var T := Succ(g, p, S[1..], I, a);
      assert Succ(g, p, S, I, a) == [WN(k, S[0].w)] + T;
      VSCons(g, p, beta, sigma, WN(k, S[0].w), T);
      var w := S[0].w;
      var v := Val(g, p, beta, sigma, k);
      var xs := VS(g, p, beta, sigma, T);
      var sk := SumKids(g, p, beta, sigma, n, b, a + 1);
      var r1 := ASum(g, p, beta, sigma, b, S[1..], I, a + 1);
      assert SumKids(g, p, beta, sigma, n, b, a) == b[a] * v + sk;
      Distrib5(b[a], w, v, xs, sk, r1);
    }
  }

  // The value of a group of p's nodes at one infoset, as a mixture over the infoset's actions.
  lemma VSGroup(g: Game, p: nat, beta: Strat, sigma: Strat, S: seq<WN>, I: nat)
    requires Inv(g, p, beta, sigma) && AllAt(g, p, S, I)
    requires I < |g.infosetActions| && g.infosetPlayer[I] == p
    ensures |beta[I]| == g.infosetActions[I]
    ensures VS(g, p, beta, sigma, S) == ASum(g, p, beta, sigma, beta[I], S, I, 0)
  {
    if |S| > 0 {
      var n := S[0].node;
      WFNode(g, n);
      assert Row(g, p, beta, sigma, n) == beta[I];
      ValInner(g, p, beta, sigma, n);
      VSGroup(g, p, beta, sigma, S[1..], I);
      ASumCons(g, p, beta, sigma, beta[I], S, I, 0);
      VSHead(g, p, beta, sigma, S);
    } else {
      VSNil(g, p, beta, sigma, S);
      reveal ASum();
      assert forall a | 0 <= a < |beta[I]| :: Succ(g, p, S, I, a) == [];
      ASumZero(g, p, beta, sigma, beta[I], S, I, 0);
    }
  }

  lemma ASumZero(g: Game, p: nat, beta: Strat, sigma: Strat, b: seq<real>, S: seq<WN>, I: nat, a: nat)
    requires Inv(g, p, beta, sigma) && AllAt(g, p, S, I) && I < |g.infosetActions|
    requires |b| == g.infosetActions[I] && a <= |b| && |S| == 0
    ensures ASum(g, p, beta, sigma, b, S, I, a) == 0.0
    decreases |b| - a
  {
    ASumUnfold(g, p, beta, sigma, b, S, I, a);
    if a < |b| {
      ASumZero(g, p, beta, sigma, b, S, I, a + 1);
      VSNil(g, p, beta, sigma, Succ(g, p, S, I, a));
    }
  }

  lemma MulEq(M: real, x: real, y: real)
    requires x == y
    ensures M * x == M * y
  {}

  lemma MulZero(M: real)
    ensures M * 0.0 == 0.0
  {}

  lemma SumFrom(b: seq<real>, a: nat)
    requires a < |b|
    ensures Sum(b[a..]) == b[a] + Sum(b[a + 1..])
  {
    assert b[a..][1..] == b[a + 1..];
  }

  // Pure arithmetic, kept opaque so the solver never unrolls it on its own.
  // Dot(b, xs, a) = sum over j >= a of b[j] * xs[j];  SumIdx(b, a) = sum over j >= a of b[j].
  opaque function Dot(b: seq<real>, xs: seq<real>, a: nat): real
    requires |xs| == |b| && a <= |b|
    decreases |b| - a
  {
    if a == |b| then 0.0 else b[a] * xs[a] + Dot(b, xs, a + 1)
  }

  opaque function SumIdx(b: seq<real>, a: nat): real
    requires a <= |b|
    decreases |b| - a
  {
    if a == |b| then 0.0 else b[a] + SumIdx(b, a + 1)
  }

  lemma DotUnfold(b: seq<real>, xs: seq<real>, a: nat)
    requires |xs| == |b| && a <= |b|
    ensures a == |b| ==> Dot(b, xs, a) == 0.0
    ensures a < |b| ==> Dot(b, xs, a) == b[a] * xs[a] + Dot(b, xs, a + 1)
  {
    reveal Dot();
  }

  lemma SumIdxUnfold(b: seq<real>, a: nat)
    requires a <= |b|
    ensures a == |b| ==> SumIdx(b, a) == 0.0
    ensures a < |b| ==> SumIdx(b, a) == b[a] + SumIdx(b, a + 1)
  {
    reveal SumIdx();
  }

  lemma SumIdxIsSum(b: seq<real>, a: nat)
    requires a <= |b|
    ensures SumIdx(b, a) == Sum(b[a..])
    decreases |b| - a
  {
    SumIdxUnfold(b, a);
    if a < |b| {
      SumIdxIsSum(b, a + 1);
      SumFrom(b, a);
    } else {
      assert b[a..] == [];
    }
  }

  lemma DotBound(b: seq<real>, xs: seq<real>, a: nat, M: real)
    requires |xs| == |b| && a <= |b|
    requires forall j | 0 <= j < |b| :: b[j] >= 0.0 && xs[j] <= M
    ensures Dot(b, xs, a) <= M * SumIdx(b, a)
    decreases |b| - a
  {
    DotUnfold(b, xs, a);
    SumIdxUnfold(b, a);
    if a < |b| {
      DotBound(b, xs, a + 1, M);
      BoundStep(b[a], xs[a], Dot(b, xs, a + 1), M, SumIdx(b, a + 1));
    } else {
      MulZero(M);
    }
  }

  lemma ASumIsDot(g: Game, p: nat, beta: Strat, sigma: Strat, b: seq<real>, S: seq<WN>, I: nat, a: nat, xs: seq<real>)
    requires Inv(g, p, beta, sigma) && AllAt(g, p, S, I) && I < |g.infosetActions|
    requires |b| == g.infosetActions[I] && a <= |b| && |xs| == |b|
    requires forall j | 0 <= j < |b| :: xs[j] == VS(g, p, beta, sigma, Succ(g, p, S, I, j))
    ensures ASum(g, p, beta, sigma, b, S, I, a) == Dot(b, xs, a)
    decreases |b| - a
  {
    ASumUnfold(g, p, beta, sigma, b, S, I, a);
    DotUnfold(b, xs, a);
    if a < |b| {
      ASumIsDot(g, p, beta, sigma, b, S, I, a + 1, xs);
    }
  }

  lemma ASumBound(g: Game, p: nat, beta: Strat, sigma: Strat, b: seq<real>, S: seq<WN>, I: nat, M: real)
    requires Inv(g, p, beta, sigma) && AllAt(g, p, S, I) && I < |g.infosetActions|
    requires |b| == g.infosetActions[I]
    requires forall j | 0 <= j < |b| :: b[j] >= 0.0
    requires forall j | 0 <= j < |b| :: VS(g, p, beta, sigma, Succ(g, p, S, I, j)) <= M
    ensures ASum(g, p, beta, sigma, b, S, I, 0) <= M * Sum(b)
  {
    var xs := seq(|b|, j requires 0 <= j < |b| => VS(g, p, beta, sigma, Succ(g, p, S, I, j)));
    ASumIsDot(g, p, beta, sigma, b, S, I, 0, xs);
    DotBound(b, xs, 0, M);
    SumIdxIsSum(b, 0);
    assert b[0..] == b;
    MulEq(M, SumIdx(b, 0), Sum(b));
  }

  // ---------- unfolding BR one step at a time ----------

  lemma BRStep(g: Game, p: nat, sigma: Strat, P: seq<WN>, R: seq<WN>)
    requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma) && NodesIn(g, P) && AllPDec(g, p, R)
    requires |P| > 0 && NodeOK(g, P[0].node)
    ensures g.nodes[P[0].node].Terminal? ==>
      BR(g, p, sigma, P, R).0 == P[0].w * U(p, g.nodes[P[0].node].payoff) + BR(g, p, sigma, P[1..], R).0
    ensures g.nodes[P[0].node].Chance? ==>
      ExpandOK(g, P[0].node, g.nodes[P[0].node].probs) &&
      NodesIn(g, Expand(g, P[0].node, g.nodes[P[0].node].probs, P[0].w, 0) + P[1..]) &&
      BR(g, p, sigma, P, R) == BR(g, p, sigma, Expand(g, P[0].node, g.nodes[P[0].node].probs, P[0].w, 0) + P[1..], R)
    ensures g.nodes[P[0].node].Decision? && g.nodes[P[0].node].player == p ==>
      AllPDec(g, p, R + [P[0]]) && BR(g, p, sigma, P, R) == BR(g, p, sigma, P[1..], R + [P[0]])
    ensures g.nodes[P[0].node].Decision? && g.nodes[P[0].node].player != p ==>
      ExpandOK(g, P[0].node, sigma[g.nodes[P[0].node].infoset]) &&
      NodesIn(g, Expand(g, P[0].node, sigma[g.nodes[P[0].node].infoset], P[0].w, 0) + P[1..]) &&
      BR(g, p, sigma, P, R)
        == BR(g, p, sigma, Expand(g, P[0].node, sigma[g.nodes[P[0].node].infoset], P[0].w, 0) + P[1..], R)
  {
    reveal BR();
    WFNode(g, P[0].node);
    if g.nodes[P[0].node].Chance? {
      ExpandFacts(g, P[0].node, g.nodes[P[0].node].probs, P[0].w, 0);
      MeasCat(g, Expand(g, P[0].node, g.nodes[P[0].node].probs, P[0].w, 0), P[1..]);
    }
    if g.nodes[P[0].node].Decision? && g.nodes[P[0].node].player != p {
      ExpandFacts(g, P[0].node, sigma[g.nodes[P[0].node].infoset], P[0].w, 0);
      MeasCat(g, Expand(g, P[0].node, sigma[g.nodes[P[0].node].infoset], P[0].w, 0), P[1..]);
    }
  }

  predicate ExpandOK(g: Game, n: nat, ws: seq<real>)
    requires n < |g.nodes|
  {
    !g.nodes[n].Terminal? && |ws| == |g.nodes[n].kids|
  }

  lemma BRGroup(g: Game, p: nat, sigma: Strat, R: seq<WN>)
    requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma) && AllPDec(g, p, R) && |R| > 0
    ensures var I := g.nodes[R[0].node].infoset;
      I < |g.infosetActions| && 0 < g.infosetActions[I] && g.infosetPlayer[I] == p
      && AllAt(g, p, FilterIn(g, p, R, I), I) && |FilterIn(g, p, R, I)| > 0
      && AllPDec(g, p, FilterOut(g, p, R, I))
      && BR(g, p, sigma, [], R).0
         == Best(g, p, sigma, FilterIn(g, p, R, I), I, 0).0 + BR(g, p, sigma, [], FilterOut(g, p, R, I)).0
  {
    reveal BR();
    var I := g.nodes[R[0].node].infoset;
    WFNode(g, R[0].node);
    WFInfoset(g, I);
    FilterFacts(g, p, R, I);
  }

  lemma BREmpty(g: Game, p: nat, sigma: Strat)
    requires WF(g) && p <= 1 && Fits(g, Opp(p), sigma)
    ensures BR(g, p, sigma, [], []).0 == 0.0
  {
    reveal BR();
  }

  // ---------- the bound ----------

  // The main bound: for every behavioral strategy beta of p, the value of the lists P and R
  // under beta is at most the computed best-response value.
  lemma BRBound(g: Game, p: nat, beta: Strat, sigma: Strat, P: seq<WN>, R: seq<WN>)
    requires Inv(g, p, beta, sigma) && ValidBeh(g, p, beta) && NodesIn(g, P) && AllPDec(g, p, R)
    ensures VS(g, p, beta, sigma, P) + VS(g, p, beta, sigma, R) <= BR(g, p, sigma, P, R).0
    decreases Meas(g, P) + Meas(g, R), |P|, 2, 0
  {
    if |P| > 0 {
      WFNode(g, P[0].node);
      if g.nodes[P[0].node].Terminal? {
        BRBoundTerm(g, p, beta, sigma, P, R);
      } else if g.nodes[P[0].node].Decision? && g.nodes[P[0].node].player == p {
        BRBoundMove(g, p, beta, sigma, P, R);
      } else {
        BRBoundExpand(g, p, beta, sigma, P, R);
      }
    } else if |R| > 0 {
      BRBoundGroup(g, p, beta, sigma, P, R);
    } else {
      VSNil(g, p, beta, sigma, P);
      VSNil(g, p, beta, sigma, R);
      BREmpty(g, p, sigma);
    }
  }

  lemma BRBoundTerm(g: Game, p: nat, beta: Strat, sigma: Strat, P: seq<WN>, R: seq<WN>)
    requires Inv(g, p, beta, sigma) && ValidBeh(g, p, beta) && NodesIn(g, P) && AllPDec(g, p, R)
    requires |P| > 0 && g.nodes[P[0].node].Terminal?
    ensures VS(g, p, beta, sigma, P) + VS(g, p, beta, sigma, R) <= BR(g, p, sigma, P, R).0
    decreases Meas(g, P) + Meas(g, R), |P|, 1, 0
  {
    var h := P[0];
    WFNode(g, h.node);
    assert Meas(g, P) == Size(g, h.node) + Meas(g, P[1..]);
    VSHead(g, p, beta, sigma, P);
    BRStep(g, p, sigma, P, R);
    BRBound(g, p, beta, sigma, P[1..], R);
  }

  lemma BRBoundMove(g: Game, p: nat, beta: Strat, sigma: Strat, P: seq<WN>, R: seq<WN>)
    requires Inv(g, p, beta, sigma) && ValidBeh(g, p, beta) && NodesIn(g, P) && AllPDec(g, p, R)
    requires |P| > 0 && g.nodes[P[0].node].Decision? && g.nodes[P[0].node].player == p
    ensures VS(g, p, beta, sigma, P) + VS(g, p, beta, sigma, R) <= BR(g, p, sigma, P, R).0
    decreases Meas(g, P) + Meas(g, R), |P|, 1, 0
  {
    var h := P[0];
    WFNode(g, h.node);
    assert Meas(g, P) == Size(g, h.node) + Meas(g, P[1..]);
    VSHead(g, p, beta, sigma, P);
    BRStep(g, p, sigma, P, R);
    MeasCat(g, R, [h]);
    VSCat(g, p, beta, sigma, R, [h]);
    VSCons(g, p, beta, sigma, h, []);
    VSNil(g, p, beta, sigma, []);
    assert [h] + [] == [h];
    BRBound(g, p, beta, sigma, P[1..], R + [h]);
  }

  lemma BRBoundExpand(g: Game, p: nat, beta: Strat, sigma: Strat, P: seq<WN>, R: seq<WN>)
    requires Inv(g, p, beta, sigma) && ValidBeh(g, p, beta) && NodesIn(g, P) && AllPDec(g, p, R)
    requires |P| > 0 && !g.nodes[P[0].node].Terminal?
    requires !(g.nodes[P[0].node].Decision? && g.nodes[P[0].node].player == p)
    ensures VS(g, p, beta, sigma, P) + VS(g, p, beta, sigma, R) <= BR(g, p, sigma, P, R).0
    decreases Meas(g, P) + Meas(g, R), |P|, 1, 0
  {
    var h := P[0];
    WFNode(g, h.node);
    assert Meas(g, P) == Size(g, h.node) + Meas(g, P[1..]);
    VSHead(g, p, beta, sigma, P);
    BRStep(g, p, sigma, P, R);
    var ws := Row(g, p, beta, sigma, h.node);
    assert g.nodes[h.node].Chance? ==> ws == g.nodes[h.node].probs;
    assert g.nodes[h.node].Decision? ==> ws == sigma[g.nodes[h.node].infoset];
    var E := Expand(g, h.node, ws, h.w, 0);
    ExpandFacts(g, h.node, ws, h.w, 0);
    MeasCat(g, E, P[1..]);
    assert BR(g, p, sigma, P, R) == BR(g, p, sigma, E + P[1..], R);
    VSCat(g, p, beta, sigma, E, P[1..]);
    VSExpand(g, p, beta, sigma, h.node, ws, h.w, 0);
    ValInner(g, p, beta, sigma, h.node);
    assert VS(g, p, beta, sigma, P) == VS(g, p, beta, sigma, E + P[1..]);
    BRBound(g, p, beta, sigma, E + P[1..], R);
  }

  lemma BRBoundGroup(g: Game, p: nat, beta: Strat, sigma: Strat, P: seq<WN>, R: seq<WN>)
    requires Inv(g, p, beta, sigma) && ValidBeh(g, p, beta) && NodesIn(g, P) && AllPDec(g, p, R)
    requires |P| == 0 && |R| > 0
    ensures VS(g, p, beta, sigma, P) + VS(g, p, beta, sigma, R) <= BR(g, p, sigma, P, R).0
    decreases Meas(g, P) + Meas(g, R), |P|, 1, 0
  {
    VSNil(g, p, beta, sigma, P);
    BRGroup(g, p, sigma, R);
    var I := g.nodes[R[0].node].infoset;
    var SI := FilterIn(g, p, R, I);
    var rest := FilterOut(g, p, R, I);
    FilterFacts(g, p, R, I);
    VSFilter(g, p, beta, sigma, R, I);
    var M := Best(g, p, sigma, SI, I, 0).0;
    BRBound(g, p, beta, sigma, [], rest);
    VSNil(g, p, beta, sigma, []);
    BestBound(g, p, beta, sigma, SI, I, 0);
    VSGroup(g, p, beta, sigma, SI, I);
    assert IsDist(beta[I]);
    ASumBound(g, p, beta, sigma, beta[I], SI, I, M);
    assert VS(g, p, beta, sigma, SI) <= M;
  }

  lemma BestBound(g: Game, p: nat, beta: Strat, sigma: Strat, SI: seq<WN>, I: nat, a: nat)
    requires Inv(g, p, beta, sigma) && ValidBeh(g, p, beta) && AllAt(g, p, SI, I) && |SI| > 0
    requires I < |g.infosetActions| && a < g.infosetActions[I]
    ensures forall j | a <= j < g.infosetActions[I] ::
      VS(g, p, beta, sigma, Succ(g, p, SI, I, j)) <= Best(g, p, sigma, SI, I, a).0
    decreases Meas(g, SI), 0, 0, g.infosetActions[I] - a
  {
    SuccFacts(g, p, SI, I, a);
    var X := Succ(g, p, SI, I, a);
    BRBound(g, p, beta, sigma, X, []);
    VSNil(g, p, beta, sigma, []);
    var c := BR(g, p, sigma, X, []).0;
    if a + 1 < g.infosetActions[I] {
      BestBound(g, p, beta, sigma, SI, I, a + 1);
      assert Best(g, p, sigma, SI, I, a).0 >= c;
      assert Best(g, p, sigma, SI, I, a).0 >= Best(g, p, sigma, SI, I, a + 1).0;
    } else {
      assert Best(g, p, sigma, SI, I, a).0 == c;
    }
  }
}
