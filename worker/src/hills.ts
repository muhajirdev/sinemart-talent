// The hills model, trained live in the Worker on the team's picks.
// Same model as pipeline/sinemart/bt.py HillsHead, inside the small taste
// space the pipeline precomputes (each applicant's `z`):
//
//   z'    = A z                      learned map of taste space (starts as identity)
//   t_k   = -|z' - q_k|^2 / 2        height on hill k (peaks equally tall)
//   score = log sum_k exp t_k        ~ height on the hill the face stands on
//
// A pick "a beat b" means P = sigmoid(score(a) - score(b)). A straight arrow
// (score = w . z) is fitted on the same picks only to compare against.

export type Pick = { w: number; l: number }; // indices into Z

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

// Small seeded RNG so a retrain is reproducible.
export function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

class Adam {
  m: Float64Array;
  v: Float64Array;
  t = 0;
  constructor(n: number, private lr: number) {
    this.m = new Float64Array(n);
    this.v = new Float64Array(n);
  }
  step(p: Float64Array, g: Float64Array) {
    this.t++;
    const b1 = 1 - 0.9 ** this.t, b2 = 1 - 0.999 ** this.t;
    for (let i = 0; i < p.length; i++) {
      this.m[i] = 0.9 * this.m[i] + 0.1 * g[i];
      this.v[i] = 0.999 * this.v[i] + 0.001 * g[i] * g[i];
      p[i] -= (this.lr * (this.m[i] / b1)) / (Math.sqrt(this.v[i] / b2) + 1e-8);
    }
  }
}

export class Hills {
  r: number;
  K: number;
  A: Float64Array; // r x r
  q: Float64Array; // K x r

  constructor(peaks: number[][]) {
    this.K = peaks.length;
    this.r = peaks[0].length;
    this.A = new Float64Array(this.r * this.r);
    for (let i = 0; i < this.r; i++) this.A[i * this.r + i] = 1;
    this.q = Float64Array.from(peaks.flat());
  }

  // Returns score, hill weights pi (K) and z' - q_k (K x r) for one face.
  private forward(z: number[]) {
    const { r, K, A, q } = this;
    const zp = new Float64Array(r);
    for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) zp[i] += A[i * r + j] * z[j];
    const diff = new Float64Array(K * r);
    const t = new Float64Array(K);
    for (let k = 0; k < K; k++) {
      let d2 = 0;
      for (let i = 0; i < r; i++) {
        const d = zp[i] - q[k * r + i];
        diff[k * r + i] = d;
        d2 += d * d;
      }
      t[k] = -0.5 * d2;
    }
    const tmax = Math.max(...t);
    let sum = 0;
    const pi = new Float64Array(K);
    for (let k = 0; k < K; k++) sum += pi[k] = Math.exp(t[k] - tmax);
    for (let k = 0; k < K; k++) pi[k] /= sum;
    return { score: Math.log(sum) + tmax, pi, diff };
  }

  score(z: number[]) {
    return this.forward(z).score;
  }

  hill(z: number[]) {
    const { pi } = this.forward(z);
    return pi.indexOf(Math.max(...pi));
  }

  fit(Z: number[][], picks: Pick[], steps = 300, lr = 0.02, l2 = 0.05) {
    const { r, K } = this;
    const optA = new Adam(r * r, lr), optQ = new Adam(K * r, lr);
    const n = Math.max(picks.length, 1);
    const used = [...new Set(picks.flatMap((p) => [p.w, p.l]))];
    for (let it = 0; it < steps; it++) {
      const fw: ReturnType<Hills["forward"]>[] = [];
      for (const i of used) fw[i] = this.forward(Z[i]); // only faces that were judged
      const gA = new Float64Array(r * r), gQ = new Float64Array(K * r);
      for (const { w, l } of picks) {
        const g = -(1 - sigmoid(fw[w].score - fw[l].score)) / n; // dL/d(s_w - s_l)
        for (const [idx, sign] of [[w, 1], [l, -1]] as const) {
          const { pi, diff } = fw[idx];
          const gs = g * sign;
          for (let i = 0; i < r; i++) {
            // ds/dz'_i = -sum_k pi_k (z'_i - q_ki) ; ds/dq_ki = pi_k (z'_i - q_ki)
            let dzi = 0;
            for (let k = 0; k < K; k++) {
              const pd = pi[k] * diff[k * r + i];
              dzi -= pd;
              gQ[k * r + i] += gs * pd;
            }
            for (let j = 0; j < r; j++) gA[i * r + j] += gs * dzi * Z[idx][j];
          }
        }
      }
      for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) gA[i * r + j] += l2 * (this.A[i * r + j] - (i === j ? 1 : 0));
      optA.step(this.A, gA);
      optQ.step(this.q, gQ);
    }
    return this;
  }

  toJSON() {
    return { A: Array.from(this.A), q: Array.from(this.q), r: this.r, K: this.K };
  }
}

export class Arrow {
  w: Float64Array;
  constructor(r: number) {
    this.w = new Float64Array(r);
  }
  score(z: number[]) {
    let s = 0;
    for (let i = 0; i < z.length; i++) s += this.w[i] * z[i];
    return s;
  }
  fit(Z: number[][], picks: Pick[], steps = 300, lr = 0.05, l2 = 1e-2) {
    const opt = new Adam(this.w.length, lr);
    const n = Math.max(picks.length, 1);
    for (let it = 0; it < steps; it++) {
      const g = new Float64Array(this.w.length);
      for (const { w, l } of picks) {
        const c = -(1 - sigmoid(this.score(Z[w]) - this.score(Z[l]))) / n;
        for (let i = 0; i < g.length; i++) g[i] += c * (Z[w][i] - Z[l][i]);
      }
      for (let i = 0; i < g.length; i++) g[i] += l2 * this.w[i];
      opt.step(this.w, g);
    }
    return this;
  }
}

export function agreement(scores: number[], picks: Pick[]): number | null {
  if (!picks.length) return null;
  return picks.filter((p) => scores[p.w] > scores[p.l]).length / picks.length;
}

// Hold out applicants once there are enough of them (the real job is scoring
// unseen faces); with only a handful of faces, hold out picks instead.
export function split(picks: Pick[], nFaces: number, rand: () => number) {
  if (nFaces >= 30) {
    const held = Array.from({ length: nFaces }, () => rand() < 0.2);
    return {
      by: "applicant" as const,
      train: picks.filter((p) => !held[p.w] && !held[p.l]),
      test: picks.filter((p) => held[p.w] || held[p.l]),
    };
  }
  const isTest = picks.map(() => rand() < 0.2);
  return { by: "pick" as const, train: picks.filter((_, i) => !isTest[i]), test: picks.filter((_, i) => isTest[i]) };
}

// 70% uncertain (half of them within the same hill), 20% among the top,
// 10% random; sides shuffled. Same-hill battles pin down where each look's
// peak sits; cross-hill battles keep the looks comparable.
export function nextPairs(
  scores: number[], std: number[], n: number, rand: () => number, hill?: number[], topK = 500, near = 5,
) {
  const m = scores.length;
  const order = scores.map((_, i) => i).sort((a, b) => scores[a] - scores[b]);
  const rank = new Array<number>(m);
  order.forEach((i, r) => (rank[i] = r));
  const pairs: { a: number; b: number; kind: string }[] = [];
  const nUnc = Math.round(n * 0.7), nTop = Math.round(n * 0.2), nRand = n - nUnc - nTop;
  const total = std.reduce((x, y) => x + y, 0);
  const pickWeighted = () => {
    if (total <= 0) return Math.floor(rand() * m);
    let u = rand() * total;
    for (let i = 0; i < m; i++) if ((u -= std[i]) <= 0) return i;
    return m - 1;
  };
  const two = (pool: number[]) => {
    const a = pool[Math.floor(rand() * pool.length)];
    let b = a;
    while (b === a) b = pool[Math.floor(rand() * pool.length)];
    return [a, b];
  };
  for (let i = 0; i < nUnc; i++) {
    const a = pickWeighted();
    if (hill && i % 2 === 0) {
      const mine = order.filter((x) => hill[x] === hill[a]); // same hill, in score order
      const r = mine.indexOf(a);
      const cand = mine.slice(Math.max(0, r - near), r + near + 1).filter((x) => x !== a);
      if (cand.length) {
        pairs.push({ a, b: cand[Math.floor(rand() * cand.length)], kind: "same_hill" });
        continue;
      }
    }
    const cand = order.slice(Math.max(0, rank[a] - near), Math.min(m, rank[a] + near + 1)).filter((x) => x !== a);
    pairs.push({ a, b: cand[Math.floor(rand() * cand.length)], kind: "uncertain" });
  }
  const top = order.slice(-Math.min(topK, m));
  for (let i = 0; i < nTop; i++) {
    const [a, b] = two(top);
    pairs.push({ a, b, kind: "top" });
  }
  const all = order;
  for (let i = 0; i < nRand; i++) {
    const [a, b] = two(all);
    pairs.push({ a, b, kind: "random" });
  }
  return pairs.map((p) => (rand() < 0.5 ? { a: p.b, b: p.a, kind: p.kind } : p));
}
