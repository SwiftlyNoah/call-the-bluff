// Exact real (rational) arithmetic over finite sequences: weighted sums and the
// facts about them that the best response proofs use. Everything is indexed from
// a start position i so that the recursion never slices the weight vector.

module Math {

  // Safe indexing: entries past the end read as 0.
  function At(s: seq<real>, i: int): real
  {
    if 0 <= i < |s| then s[i] else 0.0
  }

  // Multiplication behind an opaque wall: the solver sees products as atoms in
  // the big proofs, and the few facts about them are proved in small lemmas below.
  opaque function Mul(a: real, b: real): real
  {
    a * b
  }

  // DotF(d, v, i) = sum over j in [i, |v|) of At(d, j) * v[j].
  function DotF(d: seq<real>, v: seq<real>, i: nat): real
    decreases |v| - i
  {
    if i >= |v| then 0.0 else Mul(At(d, i), v[i]) + DotF(d, v, i + 1)
  }

  function Dot(d: seq<real>, v: seq<real>): real
  {
    DotF(d, v, 0)
  }

  // SumF(d, i) = sum over j in [i, |d|) of d[j].
  function SumF(d: seq<real>, i: nat): real
    decreases |d| - i
  {
    if i >= |d| then 0.0 else d[i] + SumF(d, i + 1)
  }

  function Sum(d: seq<real>): real
  {
    SumF(d, 0)
  }

  function SumNF(s: seq<nat>, i: nat): nat
    decreases |s| - i
  {
    if i >= |s| then 0 else s[i] + SumNF(s, i + 1)
  }

  lemma SumNGe(s: seq<nat>, i: nat, j: nat)
    requires i <= j < |s|
    ensures s[j] <= SumNF(s, i)
    decreases j - i
  {
    if i < j {
      SumNGe(s, i + 1, j);
    }
  }

  // Linearity: if z = w * x + y pointwise then Dot(d, z) = w * Dot(d, x) + Dot(d, y).
  lemma DotLin(d: seq<real>, x: seq<real>, y: seq<real>, z: seq<real>, w: real, i: nat)
    requires |x| == |z| && |y| == |z|
    requires forall j :: 0 <= j < |z| ==> z[j] == Mul(w, x[j]) + y[j]
    ensures DotF(d, z, i) == Mul(w, DotF(d, x, i)) + DotF(d, y, i)
    decreases |z| - i
  {
    if i < |z| {
      DotLin(d, x, y, z, w, i + 1);
      LinStep(At(d, i), w, x[i], y[i], z[i], DotF(d, x, i + 1), DotF(d, y, i + 1), DotF(d, z, i + 1));
    } else {
      reveal Mul();
    }
  }

  lemma LinStep(c: real, w: real, xi: real, yi: real, zi: real, dx: real, dy: real, dz: real)
    requires zi == Mul(w, xi) + yi && dz == Mul(w, dx) + dy
    ensures Mul(c, zi) + dz == Mul(w, Mul(c, xi) + dx) + (Mul(c, yi) + dy)
  {
    reveal Mul();
    assert c * zi == c * (w * xi + yi);
  }

  lemma DotZero(d: seq<real>, v: seq<real>, i: nat)
    requires forall j :: 0 <= j < |v| ==> v[j] == 0.0
    ensures DotF(d, v, i) == 0.0
    decreases |v| - i
  {
    if i < |v| {
      DotZero(d, v, i + 1);
      reveal Mul();
    }
  }

  lemma DotNeg(d: seq<real>, x: seq<real>, z: seq<real>)
    requires |x| == |z|
    requires forall j :: 0 <= j < |z| ==> z[j] == -x[j]
    ensures Dot(d, z) == -Dot(d, x)
  {
    var y := seq(|z|, j => 0.0);
    DotZero(d, y, 0);
    forall j | 0 <= j < |z| ensures z[j] == Mul(-1.0, x[j]) + y[j] {
      reveal Mul();
    }
    DotLin(d, x, y, z, -1.0, 0);
    reveal Mul();
  }

  lemma MulMono(c: real, a: real, b: real)
    requires c >= 0.0 && a <= b
    ensures Mul(c, a) <= Mul(c, b)
  {
    reveal Mul();
    assert c * b - c * a == c * (b - a);
  }

  // Monotonicity in v for nonnegative weights.
  lemma DotMono(d: seq<real>, x: seq<real>, y: seq<real>, i: nat)
    requires |x| == |y|
    requires forall j :: 0 <= j < |d| ==> d[j] >= 0.0
    requires forall j :: 0 <= j < |x| ==> x[j] <= y[j]
    ensures DotF(d, x, i) <= DotF(d, y, i)
    decreases |x| - i
  {
    if i < |x| {
      DotMono(d, x, y, i + 1);
      MulMono(At(d, i), x[i], y[i]);
    }
  }

  lemma BoundStep(di: real, yi: real, m: real, rest: real, srest: real)
    requires di >= 0.0 && yi <= m && rest <= m * srest
    ensures Mul(di, yi) + rest <= m * (di + srest)
  {
    reveal Mul();
    assert di * yi <= di * m by {
      assert di * m - di * yi == di * (m - yi);
    }
    assert m * (di + srest) == m * di + m * srest;
  }

  // A convex combination of values each at most m is at most m * (total weight).
  lemma DotBound(d: seq<real>, y: seq<real>, m: real, i: nat)
    requires |d| == |y|
    requires forall j :: 0 <= j < |d| ==> d[j] >= 0.0
    requires forall j :: 0 <= j < |y| ==> y[j] <= m
    ensures DotF(d, y, i) <= m * SumF(d, i)
    decreases |y| - i
  {
    if i < |y| {
      DotBound(d, y, m, i + 1);
      BoundStep(d[i], y[i], m, DotF(d, y, i + 1), SumF(d, i + 1));
    }
  }

  // Dot with a one-hot weight vector selects one entry.
  lemma DotOneHot(d: seq<real>, v: seq<real>, k: nat, i: nat)
    requires |d| == |v| && k < |v|
    requires forall j :: 0 <= j < |d| ==> d[j] == (if j == k then 1.0 else 0.0)
    ensures DotF(d, v, i) == (if i <= k then v[k] else 0.0)
    decreases |v| - i
  {
    if i < |v| {
      DotOneHot(d, v, k, i + 1);
      reveal Mul();
    }
  }

  lemma SumOneHot(d: seq<real>, k: nat, i: nat)
    requires k < |d|
    requires forall j :: 0 <= j < |d| ==> d[j] == (if j == k then 1.0 else 0.0)
    ensures SumF(d, i) == (if i <= k then 1.0 else 0.0)
    decreases |d| - i
  {
    if i < |d| {
      SumOneHot(d, k, i + 1);
    }
  }
}
