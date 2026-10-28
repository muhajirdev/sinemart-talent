// The ranker since 2026-10-28: a straight line on the ranker model's
// full-size vectors (all 768 numbers, not the 16-number taste space), plus a
// small correction per face learned from that face's own battles:
//
//   score(i) = w . x_i + b_i        P(a beats b) = sigmoid(score(a) - score(b))
//
// w is kept small (l2), b smaller still (lb): a face with few battles stays
// close to what its photo says; a face that keeps winning or losing moves.
// New faces (no battles, e.g. test batches) get b = 0. "Neither" answers stay
// out of the ordering (offline they cost ~3 points) and only place each
// group's "no" line tau afterwards.
// Offline, 30 held-out splits at 1,375 comparisons: 74.4% vs 66.6% for the
// 16-number line with "neither" (docs/context.md, 2026-10-28).

import type { Below, Pick } from "./hills";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
// log(sigmoid(x)) without overflow.
const logSig = (x: number) => (x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x)));

// Minimise f with L-BFGS (backtracking line search). f returns [value, gradient].
export function lbfgs(f: (x: Float64Array) => [number, Float64Array], x0: Float64Array, maxIter = 300, m = 10) {
  const n = x0.length;
  let x = Float64Array.from(x0);
  let [fx, g] = f(x);
  const S: Float64Array[] = [], Y: Float64Array[] = [], rho: number[] = [];
  const dot = (a: Float64Array, b: Float64Array) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; };
  for (let it = 0; it < maxIter; it++) {
    // Two-loop recursion: d = -H g.
    const q = Float64Array.from(g), alpha: number[] = [];
    for (let k = S.length - 1; k >= 0; k--) {
      const a = rho[k] * dot(S[k], q); alpha[k] = a;
      for (let i = 0; i < n; i++) q[i] -= a * Y[k][i];
    }
    const gamma = S.length ? dot(S[S.length - 1], Y[S.length - 1]) / dot(Y[Y.length - 1], Y[Y.length - 1]) : 1 / Math.max(1, Math.sqrt(dot(g, g)));
    for (let i = 0; i < n; i++) q[i] *= gamma;
    for (let k = 0; k < S.length; k++) {
      const b = rho[k] * dot(Y[k], q);
      for (let i = 0; i < n; i++) q[i] += S[k][i] * (alpha[k] - b);
    }
    const d = q.map((v) => -v);
    let slope = dot(g, d);
    if (slope >= 0) { for (let i = 0; i < n; i++) d[i] = -g[i]; slope = -dot(g, g); S.length = Y.length = rho.length = 0; }
    let t = 1, xn = x, fn = fx, gn = g;
    for (let ls = 0; ls < 30; ls++) {
      xn = x.map((v, i) => v + t * d[i]);
      [fn, gn] = f(xn);
      if (fn <= fx + 1e-4 * t * slope) break;
      t *= 0.5;
    }
    const s = xn.map((v, i) => v - x[i]), y = gn.map((v, i) => v - g[i]);
    const sy = dot(s, y);
    if (sy > 1e-12) {
      S.push(s); Y.push(y); rho.push(1 / sy);
      if (S.length > m) { S.shift(); Y.shift(); rho.shift(); }
    }
    const done = Math.abs(fx - fn) <= 1e-10 * Math.max(1, Math.abs(fx));
    x = xn; fx = fn; g = gn;
    if (done || Math.sqrt(dot(g, g)) < 1e-7) break;
  }
  return x;
}

export class FullLine {
  w: Float64Array;
  b: Float64Array; // one correction per face (index into X)
  tau: Float64Array; // "no" line per group, on the centred score
  centre: Float64Array; // per-group mean score, so tau is relative to the average face
  constructor(d: number, nFaces: number, nGroups = 0) {
    this.w = new Float64Array(d);
    this.b = new Float64Array(nFaces);
    this.tau = new Float64Array(nGroups);
    this.centre = new Float64Array(nGroups);
  }
  image(x: number[]) {
    let s = 0;
    for (let i = 0; i < x.length; i++) s += this.w[i] * x[i];
    return s;
  }
  score(X: number[][], i: number) {
    return this.image(X[i]) + this.b[i];
  }
  fit(X: number[][], picks: Pick[], l2 = 3e-3, lb = 0.3, maxIter = 300) {
    const d = this.w.length, N = this.b.length, P = picks.length;
    if (!P) return this;
    const dX = new Float64Array(P * d);
    picks.forEach((p, k) => { for (let j = 0; j < d; j++) dX[k * d + j] = X[p.w][j] - X[p.l][j]; });
    const f = (t: Float64Array): [number, Float64Array] => {
      const g = new Float64Array(d + N);
      let loss = 0;
      for (let k = 0; k < P; k++) {
        const { w, l, y = 1 } = picks[k];
        let m = t[d + w] - t[d + l];
        for (let j = 0; j < d; j++) m += dX[k * d + j] * t[j];
        loss -= y * logSig(m) + (1 - y) * logSig(-m);
        const c = -(y - sigmoid(m)) / P;
        for (let j = 0; j < d; j++) g[j] += c * dX[k * d + j];
        g[d + w] += c; g[d + l] -= c;
      }
      loss /= P;
      for (let j = 0; j < d; j++) { loss += 0.5 * l2 * t[j] * t[j]; g[j] += l2 * t[j]; }
      for (let i = 0; i < N; i++) { loss += (0.5 * lb * t[d + i] * t[d + i]) / P; g[d + i] += (lb * t[d + i]) / P; }
      return [loss, g];
    };
    const t = lbfgs(f, new Float64Array(d + N), maxIter);
    this.w = t.slice(0, d);
    this.b = t.slice(d);
    return this;
  }
  // Place each group's "no" line from the "neither" answers, with the scores
  // fixed: faces in `below` should sit under tau[g]. A pull towards 0 (the
  // group's average face) keeps "clear no" meaning "well below average".
  fitTau(X: number[][], G: number[], below: Below[], nPicks: number, pull = 0.1) {
    const nG = this.tau.length;
    for (let g = 0; g < nG; g++) {
      const s = X.map((_, i) => i).filter((i) => G[i] === g).map((i) => this.score(X, i));
      this.centre[g] = s.length ? s.reduce((a, b) => a + b, 0) / s.length : 0;
    }
    const n = Math.max(nPicks + below.length, 1);
    for (let g = 0; g < nG; g++) {
      const mine = below.filter((b) => b.g === g).map((b) => this.score(X, b.i) - this.centre[g]);
      let tau = 0;
      for (let it = 0; it < 50; it++) {
        // d/dtau of sum -log sigmoid(tau - s) / n + pull/2 tau^2, and its curvature.
        let gr = pull * tau, h = pull;
        for (const s of mine) { const p = sigmoid(tau - s); gr -= (1 - p) / n; h += (p * (1 - p)) / n; }
        tau -= gr / h;
        if (Math.abs(gr) < 1e-10) break;
      }
      this.tau[g] = tau;
    }
    return this;
  }
}

// Stored as model.params_ranker: the line, each face's correction by
// applicant id, and the "no" lines.
export function fullJson(m: FullLine, ids: number[], groups: (string | null)[]) {
  const b: Record<number, number> = {};
  ids.forEach((id, i) => { if (m.b[i]) b[id] = +m.b[i].toFixed(6); });
  return JSON.stringify({ w: Array.from(m.w, (v) => +v.toFixed(8)), b, tau: Array.from(m.tau), centre: Array.from(m.centre), groups });
}

export function fullFrom(p: { w: number[]; b: Record<string, number> }) {
  return {
    score: (id: number, x: number[]) => {
      let s = p.b[id] ?? 0;
      for (let i = 0; i < x.length; i++) s += p.w[i] * x[i];
      return s;
    },
  };
}
