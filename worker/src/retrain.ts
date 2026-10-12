// Retrain the hills model from every pick in D1, rescore everyone, refill
// the pair queue. Runs automatically every RETRAIN_EVERY picks and on demand.
// The same fit also runs on one rater's picks alone (their personal taste);
// those models are cached in rater_models and never touch the shared scores.

import { Arrow, Hills, Pick, agreement, groupedPairs, rng, split } from "./hills";

export const RETRAIN_EVERY = 10;
export const MIN_PICKS = 5;

type Env = { DB: D1Database };
export type Params = { A: number[]; q: number[]; r: number; K: number; pg?: (string | null)[] };

async function runBatches(db: D1Database, stmts: D1PreparedStatement[], size = 200) {
  for (let i = 0; i < stmts.length; i += size) await db.batch(stmts.slice(i, i + size));
}

// Taste space + picks, optionally only one rater's.
export async function loadData(env: Env, rater: string | null = null) {
  const model = await env.DB.prepare("SELECT init FROM model WHERE id = 1").first<{ init: string }>();
  if (!model?.init) return null;
  const init = JSON.parse(model.init) as { peaks: number[][]; peak_gender?: string[] };
  const peaks = init.peaks;
  const peakGroup: (string | null)[] = init.peak_gender ?? [];
  const faces = (
    await env.DB.prepare("SELECT id, z, gender FROM applicants WHERE z IS NOT NULL ORDER BY id")
      .all<{ id: number; z: string; gender: string | null }>()
  ).results;
  const index = new Map(faces.map((f, i) => [f.id, i]));
  const Z = faces.map((f) => JSON.parse(f.z) as number[]);
  const G = faces.map((f) => f.gender);
  const rows = (
    await env.DB.prepare(
      `SELECT rater, winner_id, CASE WHEN winner_id = left_id THEN right_id ELSE left_id END AS loser_id
       FROM picks WHERE winner_id IS NOT NULL AND (?1 IS NULL OR rater = ?1) ORDER BY id`,
    )
      .bind(rater)
      .all<{ rater: string; winner_id: number; loser_id: number }>()
  ).results.filter((r) => index.has(r.winner_id) && index.has(r.loser_id))
    // A battle that crossed groups (before a gender fix) says nothing within one.
    .filter((r) => G[index.get(r.winner_id)!] === G[index.get(r.loser_id)!]);
  const picks: (Pick & { rater: string })[] = rows.map((r) => ({
    w: index.get(r.winner_id)!, l: index.get(r.loser_id)!, rater: r.rater,
  }));
  return { peaks, peakGroup, faces, index, Z, G, picks };
}

export function hillsFrom(p: Params) {
  const h = new Hills(Array.from({ length: p.K }, (_, k) => p.q.slice(k * p.r, (k + 1) * p.r)), p.pg ?? []);
  h.A = Float64Array.from(p.A);
  return h;
}

// One rater's own model, refitted only when they've made new picks.
export async function raterModel(env: Env, rater: string) {
  const data = await loadData(env, rater);
  if (!data) return null;
  const n = data.picks.length;
  if (n < MIN_PICKS) return { data, hills: null, n };
  const cached = await env.DB.prepare("SELECT n_picks, params FROM rater_models WHERE rater = ?1")
    .bind(rater)
    .first<{ n_picks: number; params: string }>();
  if (cached && cached.n_picks === n) return { data, hills: hillsFrom(JSON.parse(cached.params)), n };
  const hills = new Hills(data.peaks, data.peakGroup).fit(data.Z, data.picks, data.G);
  await env.DB.prepare(
    `INSERT INTO rater_models (rater, n_picks, params, trained_at) VALUES (?1, ?2, ?3, datetime('now'))
     ON CONFLICT (rater) DO UPDATE SET n_picks = ?2, params = ?3, trained_at = datetime('now')`,
  )
    .bind(rater, n, JSON.stringify(hills))
    .run();
  return { data, hills, n };
}

// "One hill" comparison model: a single peak per group (cewek, cowok),
// starting at the group's average face. Same picks, same training.
export function oneHill(Z: number[][], G: (string | null)[]) {
  const groups = [...new Set(G)];
  const peaks = groups.map((g) => {
    const rows = Z.filter((_, i) => G[i] === g);
    return rows[0].map((_, d) => rows.reduce((s, z) => s + z[d], 0) / rows.length);
  });
  return new Hills(peaks, groups);
}

export async function retrain(env: Env) {
  const data = await loadData(env);
  if (!data) return { status: "no taste space loaded yet" };
  const { peaks, peakGroup, faces, Z, G, picks } = data;
  const fresh = () => new Hills(peaks, peakGroup);
  if (picks.length < MIN_PICKS) return { status: `waiting for ${MIN_PICKS} picks`, picks: picks.length };

  // Held-out check: hills vs the straight arrow on picks they didn't see.
  const rand = rng(picks.length * 7919);
  const { by, train, test } = split(picks, faces.length, rand);
  let heldoutHills: number | null = null, heldoutLine: number | null = null, heldoutOne: number | null = null;
  if (train.length >= MIN_PICKS && test.length >= 3) {
    const h = fresh().fit(Z, train, G);
    const a = new Arrow(Z[0].length).fit(Z, train);
    heldoutHills = agreement(Z.map((z, i) => h.score(z, G[i])), test);
    heldoutLine = agreement(Z.map((z) => a.score(z)), test);
    const o = oneHill(Z, G).fit(Z, train, G);
    heldoutOne = agreement(Z.map((z, i) => o.score(z, G[i])), test);
  }
  const one = oneHill(Z, G).fit(Z, picks, G);

  // The ranker: hills on every pick from every rater. Std = spread of 3 bootstrap fits.
  const main = fresh().fit(Z, picks, G);
  const scores = Z.map((z, i) => main.score(z, G[i]));
  const hill = Z.map((z, i) => main.hill(z, G[i]));
  const runs = [0, 1, 2].map(() => {
    const boot = picks.map(() => picks[Math.floor(rand() * picks.length)]);
    const b = fresh().fit(Z, boot, G, 150);
    const s = Z.map((z, i) => b.score(z, G[i]));
    // Centre within each group: scores only compare inside a group.
    const mean = new Map<string | null, number>();
    for (const g of new Set(G)) {
      const v = s.filter((_, i) => G[i] === g);
      mean.set(g, v.reduce((x, y) => x + y, 0) / v.length);
    }
    return s.map((x, i) => x - mean.get(G[i])!);
  });
  const std = scores.map((_, i) => {
    const v = runs.map((r) => r[i]);
    const m = v.reduce((x, y) => x + y, 0) / v.length;
    return Math.sqrt(v.reduce((x, y) => x + (y - m) ** 2, 0) / v.length);
  });

  const now = new Date().toISOString();
  const nPairs = Math.min(500, Math.max(20, faces.length * 3));
  const pairs = groupedPairs(scores, std, nPairs, rand, hill, G);
  const stmts: D1PreparedStatement[] = faces.map((f, i) =>
    env.DB.prepare(
      "UPDATE applicants SET score = ?1, score_std = ?2, look = ?3, model = 'hills', updated_at = ?4 WHERE id = ?5",
    ).bind(scores[i], std[i], hill[i], now, f.id),
  );
  // Every unanswered pair goes, including ones on screen now: they were chosen
  // by the old model (and may predate a gender fix). Their picks still count.
  stmts.push(env.DB.prepare("DELETE FROM pair_queue WHERE done_at IS NULL"));
  for (const p of pairs) {
    stmts.push(
      env.DB.prepare("INSERT INTO pair_queue (left_id, right_id, kind, batch) VALUES (?1, ?2, ?3, ?4)").bind(
        faces[p.a].id, faces[p.b].id, p.kind, now,
      ),
    );
  }
  stmts.push(
    env.DB.prepare(
      `UPDATE model SET params = ?1, active = 'hills', heldout_hills = ?2, heldout_line = ?3, n_test = ?4,
       split_by = ?5, n_picks = ?6, trained_at = ?7, params_one = ?8, heldout_one = ?9 WHERE id = 1`,
    ).bind(JSON.stringify(main), heldoutHills, heldoutLine, test.length, by, picks.length, now, JSON.stringify(one), heldoutOne),
  );
  await runBatches(env.DB, stmts);
  return {
    status: "trained", picks: picks.length, heldout_hills: heldoutHills, heldout_one: heldoutOne,
    heldout_line: heldoutLine, n_test: test.length,
  };
}

// How much the raters agree. For each rater: how often the team ranking
// agrees with their picks, and how often each other rater's personal model
// predicts their picks. Where two raters judged the very same pair, also the
// plain share of matching answers.
export async function raterAgreement(env: Env) {
  const all = await loadData(env);
  if (!all) return { raters: [], cross: [], same_pair: [] };
  const team = await env.DB.prepare("SELECT params FROM model WHERE id = 1").first<{ params: string | null }>();
  const teamModel = team?.params ? hillsFrom(JSON.parse(team.params)) : null;
  const teamScores = teamModel ? all.Z.map((z, i) => teamModel.score(z, all.G[i])) : null;
  const names = [...new Set(all.picks.map((p) => p.rater))].sort();
  const byRater = new Map(names.map((n) => [n, all.picks.filter((p) => p.rater === n)]));
  const models = new Map<string, number[] | null>();
  for (const n of names) {
    const m = await raterModel(env, n);
    models.set(n, m?.hills ? all.Z.map((z, i) => m.hills!.score(z, all.G[i])) : null);
  }
  const raters = names.map((n) => ({
    name: n,
    picks: byRater.get(n)!.length,
    team: teamScores ? agreement(teamScores, byRater.get(n)!) : null,
    own_model: models.get(n) !== null,
  }));
  const cross = names.flatMap((a) =>
    names.filter((b) => b !== a && models.get(a)).map((b) => ({
      model: a, picks_of: b, agree: agreement(models.get(a)!, byRater.get(b)!), n: byRater.get(b)!.length,
    })),
  );
  // Same unordered pair judged by two raters.
  const key = (p: Pick) => (p.w < p.l ? `${p.w}-${p.l}` : `${p.l}-${p.w}`);
  const same_pair = [];
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const mine = new Map(byRater.get(names[i])!.map((p) => [key(p), p.w]));
      let n = 0, match = 0;
      for (const p of byRater.get(names[j])!) {
        const w = mine.get(key(p));
        if (w === undefined) continue;
        n++;
        if (w === p.w) match++;
      }
      if (n) same_pair.push({ a: names[i], b: names[j], n, agree: match / n });
    }
  }
  return { raters, cross, same_pair };
}
