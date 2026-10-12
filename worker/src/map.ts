// A 2D "territory map" of the hills model for the results page.
//
// Faces and peaks live in the learned taste space (z' = A z). We take the
// plane through them that shows the most spread (top 2 principal directions)
// and sample the model's height on that plane. It is a slice of a higher-
// dimensional landscape, so distances are approximate; scores are not.

type Env = { DB: D1Database };

const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);

function topDirections(points: number[][], c: number[], count: number, given: number[][] = []) {
  const r = c.length;
  const cov = Array.from({ length: r }, () => new Array<number>(r).fill(0));
  for (const p of points) {
    const d = p.map((x, i) => x - c[i]);
    for (let i = 0; i < r; i++) for (let j = 0; j < r; j++) cov[i][j] += d[i] * d[j];
  }
  const dirs: number[][] = [...given];
  for (let n = 0; n < count; n++) {
    let v = Array.from({ length: r }, (_, i) => Math.cos(i + n + 1)); // deterministic start
    for (let it = 0; it < 100; it++) {
      for (const u of dirs) {
        const k = dot(v, u);
        v = v.map((x, i) => x - k * u[i]);
      }
      const w = cov.map((row) => dot(row, v));
      const norm = Math.hypot(...w) || 1;
      v = w.map((x) => x / norm);
    }
    dirs.push(v);
  }
  return dirs.slice(given.length);
}

function height(p: number[], q: number[][]) {
  const t = q.map((qk) => -0.5 * qk.reduce((s, x, i) => s + (p[i] - x) ** 2, 0));
  const m = Math.max(...t);
  return m + Math.log(t.reduce((s, x) => s + Math.exp(x - m), 0));
}

// `own` swaps in another model (a rater's own, one hill, or Model 2): its
// params and per-face score/hill, and for Model 2 its own taste space
// (`zs`, `init`).
export async function territory(
  env: Env,
  own?: {
    params: string | null;
    faces: Map<number, { score: number | null; hill: number | null }>;
    zs?: Map<number, number[]>;
    init?: { peaks: number[][]; peak_gender?: string[] };
  },
  gender: string | null = null, // only this group's faces and hills
  gridN = 64,
) {
  const model = await env.DB.prepare("SELECT init, params FROM model WHERE id = 1").first<{ init: string; params: string | null }>();
  if (!model?.init) return { faces: [], peaks: [], grid: null };
  const init = own?.init ?? (JSON.parse(model.init) as { peaks: number[][]; peak_gender?: string[] });
  const peaks0 = init.peaks;
  let pg: (string | null)[] = peaks0.map((_, k) => init.peak_gender?.[k] ?? null);
  const r = peaks0[0].length;
  let A: number[] = Array.from({ length: r * r }, (_, i) => (i % (r + 1) === 0 ? 1 : 0));
  let q = peaks0;
  const params = own ? own.params : model.params;
  if (params) {
    const p = JSON.parse(params) as { A: number[]; q: number[]; K: number; pg?: (string | null)[] };
    if (p.pg) pg = p.pg; // a model with its own peaks (one hill per group) carries their groups
    A = p.A;
    q = Array.from({ length: p.K }, (_, k) => p.q.slice(k * r, (k + 1) * r));
  }

  const { results } = await env.DB.prepare(
    "SELECT id, z, score, look FROM applicants WHERE z IS NOT NULL AND (?1 IS NULL OR gender = ?1)",
  ).bind(gender).all<{ id: number; z: string; score: number | null; look: number | null }>();
  // This group's peaks, keeping their global numbers (k) for colours/labels.
  const keep = q.map((_, k) => gender === null || pg[k] === null || pg[k] === gender);
  const ks = q.map((_, k) => k).filter((k) => keep[k]);
  if (ks.length) q = ks.map((k) => q[k]);
  else ks.push(...q.map((_, k) => k));
  if (own?.zs) {
    for (const f of results) f.z = JSON.stringify(own.zs.get(f.id) ?? []);
  }
  if (own) {
    for (const f of results) {
      const o = own.faces.get(f.id);
      f.score = o?.score ?? null;
      f.look = o?.hill ?? null;
    }
  }
  const zp = results.map((f) => {
    const z = JSON.parse(f.z) as number[];
    return Array.from({ length: r }, (_, i) => z.reduce((s, x, j) => s + A[i * r + j] * x, 0));
  });
  const hillOf = (p: number[]) => {
    const d = q.map((qk) => qk.reduce((s, x, i) => s + (p[i] - x) ** 2, 0));
    return ks[d.indexOf(Math.min(...d))];
  };

  // Pick the plane that pulls the peaks apart first (so each look's
  // territory is visible), then fill in with the faces' own spread.
  const all = [...zp, ...q];
  const mean = (pts: number[][]) => Array.from({ length: r }, (_, i) => pts.reduce((s, p) => s + p[i], 0) / pts.length);
  const c = mean(all);
  let plane: number[][] = [[1], [0]];
  if (r >= 2) {
    const fromPeaks = q.length > 1 ? topDirections(q, mean(q), Math.min(2, q.length - 1)) : [];
    plane = [...fromPeaks, ...topDirections(all, c, 2 - fromPeaks.length, fromPeaks)];
  }
  const [u, v] = plane;
  const xy = (p: number[]) => {
    const d = p.map((x, i) => x - c[i]);
    return [dot(d, u), dot(d, v)];
  };
  const fxy = zp.map(xy), pxy = q.map(xy);
  const xs = [...fxy, ...pxy].map((p) => p[0]), ys = [...fxy, ...pxy].map((p) => p[1]);
  const pad = (lo: number, hi: number) => {
    const s = Math.max(hi - lo, 1) * 0.18;
    return [lo - s, hi + s];
  };
  const [xmin, xmax] = pad(Math.min(...xs), Math.max(...xs));
  const [ymin, ymax] = pad(Math.min(...ys), Math.max(...ys));
  const h: number[] = [];
  for (let j = 0; j < gridN; j++) {
    for (let i = 0; i < gridN; i++) {
      const x = xmin + ((i + 0.5) / gridN) * (xmax - xmin);
      const y = ymax - ((j + 0.5) / gridN) * (ymax - ymin);
      h.push(height(c.map((ci, k) => ci + x * u[k] + y * v[k]), q));
    }
  }

  return {
    faces: results.map((f, i) => ({ id: f.id, x: fxy[i][0], y: fxy[i][1], hill: f.look ?? hillOf(zp[i]), score: f.score })),
    peaks: pxy.map(([x, y], i) => ({ k: ks[i], x, y })),
    grid: { n: gridN, xmin, xmax, ymin, ymax, h },
  };
}
