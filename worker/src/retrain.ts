// Retrain from every pick in D1, rescore everyone, refill the pair queue.
// Runs automatically every RETRAIN_EVERY picks and on demand. The straight
// arrow ranks and picks the battles (it won on held-out picks, 2026-10-22);
// the hills and one-hill models are refitted alongside as comparisons, and
// the hills still name each person's cluster (look).
// The same fit also runs on one rater's picks alone (their personal taste);
// those models are cached in rater_models and never touch the shared scores.

import { Arrow, Below, Hills, Pick, agreement, groupedPairs, rng, split } from "./hills";

export const RETRAIN_EVERY = 10;
export const MIN_PICKS = 5;
// A pick that tells the model something: a winner, "Sama saja" (tie) or
// "Keduanya kurang menarik" (neither). Skips don't.
export const ANSWERED = "(winner_id IS NOT NULL OR tie = 1 OR neither = 1)";
// About 1 in REPEAT_EVERY battles re-asks a pair the rater already answered.
// Those answers measure how consistent people are, so they stay out of training.
export const REPEAT_EVERY = 20;
// Which model's straight line ranks everyone and picks the battles: "m1" is
// Model 1 (the main taste space), anything else an alt_models key.
// 2026-10-27: Model 6 (Muajir: "can we try model 6"). Offline, 30 held-out
// splits at 552 picks: 74.8% vs Model 1's 64.0%, ahead in all 30.
export const RANKER: string = "m6";

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
    await env.DB.prepare("SELECT id, z, gender FROM applicants WHERE z IS NOT NULL AND noise IS NULL AND test = 0 ORDER BY id")
      .all<{ id: number; z: string; gender: string | null }>()
  ).results;
  const index = new Map(faces.map((f, i) => [f.id, i]));
  const Z = faces.map((f) => JSON.parse(f.z) as number[]);
  const G = faces.map((f) => f.gender);
  const rows = (
    await env.DB.prepare(
      `SELECT rater, tie, COALESCE(winner_id, left_id) AS winner_id,
         CASE WHEN winner_id = right_id THEN left_id ELSE right_id END AS loser_id
       FROM picks WHERE (winner_id IS NOT NULL OR tie = 1) AND kind != 'repeat' AND (?1 IS NULL OR rater = ?1) ORDER BY id`,
    )
      .bind(rater)
      .all<{ rater: string; tie: number; winner_id: number; loser_id: number }>()
  ).results.filter((r) => index.has(r.winner_id) && index.has(r.loser_id))
    // A battle that crossed groups (before a gender fix) says nothing within one.
    .filter((r) => G[index.get(r.winner_id)!] === G[index.get(r.loser_id)!]);
  const picks: (Pick & { rater: string })[] = rows.map((r) => ({
    w: index.get(r.winner_id)!, l: index.get(r.loser_id)!, y: r.tie ? 0.5 : 1, rater: r.rater,
  }));
  // "Keduanya kurang menarik": both faces fell below their group's "no" line. Only the
  // straight line learns from these; groups are numbered for its tau.
  const groupNo = [...new Set(G)];
  const neither = (
    await env.DB.prepare(
      `SELECT left_id, right_id FROM picks WHERE neither = 1 AND kind != 'repeat' AND (?1 IS NULL OR rater = ?1) ORDER BY id`,
    ).bind(rater).all<{ left_id: number; right_id: number }>()
  ).results.filter((r) => index.has(r.left_id) && index.has(r.right_id))
    .filter((r) => G[index.get(r.left_id)!] === G[index.get(r.right_id)!]);
  const below: Below[] = neither.flatMap((r) => [r.left_id, r.right_id].map((id) => {
    const i = index.get(id)!;
    return { i, g: groupNo.indexOf(G[i]) };
  }));
  // Answers the scores learned from, counted the way maybeRetrain counts them.
  const used = picks.length + neither.length;
  return { peaks, peakGroup, faces, index, Z, G, picks, below, groupNo, used };
}

export function arrowFrom(p: { w: number[]; tau?: number[] }) {
  const a = new Arrow(p.w.length);
  a.w = Float64Array.from(p.w);
  if (p.tau) a.tau = Float64Array.from(p.tau);
  return a;
}
const arrowJson = (a: Arrow, groups: (string | null)[] = []) =>
  JSON.stringify({ w: Array.from(a.w), tau: Array.from(a.tau), groups });

export function hillsFrom(p: Params) {
  const h = new Hills(Array.from({ length: p.K }, (_, k) => p.q.slice(k * p.r, (k + 1) * p.r)), p.pg ?? []);
  h.A = Float64Array.from(p.A);
  return h;
}

// The same data cut down to the faces `keep` accepts, picks renumbered.
function only(data: NonNullable<Awaited<ReturnType<typeof loadData>>>, keep: (f: { id: number }) => boolean) {
  const old = data.faces.map((f, i) => i).filter((i) => keep(data.faces[i]));
  const to = new Map(old.map((i, j) => [i, j]));
  const picks = data.picks.filter((p) => to.has(p.w) && to.has(p.l)).map((p) => ({ ...p, w: to.get(p.w)!, l: to.get(p.l)! }));
  const below = data.below.filter((b) => to.has(b.i)).map((b) => ({ ...b, i: to.get(b.i)! }));
  const faces = old.map((i) => data.faces[i]);
  return { ...data, faces, index: new Map(faces.map((f, j) => [f.id, j])), Z: old.map((i) => data.Z[i]), G: old.map((i) => data.G[i]),
    picks, below, used: picks.length + below.length / 2 };
}

// One rater's own model, on any model and shape the scores page shows
// (2026-10-28, Muajir: "Taste of" with Model 1..N and straight line / many /
// one cluster). Cached per rater and mode, refitted only when they've made new
// picks. "hills" keeps its old cache key, the plain name; login collapses
// whitespace, so a name never holds a tab and `name<TAB>mode` can't clash.
export async function raterModel(env: Env, rater: string, mode = "hills") {
  const loaded = await loadData(env, rater);
  if (!loaded) return null;
  let data = loaded;
  // Same modes as the team view: "line" = the ranker's straight line, "1:line" = Model 1's.
  const [key, variant] = mode === "line" ? [RANKER === "m1" ? "1" : RANKER, "line"]
    : ["hills", "one"].includes(mode) ? ["1", mode] : [mode.split(":")[0], mode.split(":")[1] ?? "hills"];
  let Z = data.Z, peaks = data.peaks, pg = data.peakGroup;
  let teamHills: string | null, zs: Map<number, number[]> | undefined, init: { peaks: number[][]; peak_gender?: string[] } | undefined;
  if (key === "1") {
    teamHills = (await env.DB.prepare("SELECT params FROM model WHERE id = 1").first<{ params: string | null }>())?.params ?? null;
  } else {
    const alt = await loadAlternate(env, key);
    if (!alt) return null;
    if (!alt.zs.size) {
      // A fixed rule: nothing to learn, so everyone keeps the rule's own score.
      const fixed = alt.fixed ?? {};
      return { data, n: data.picks.length, pg: [], params: null, zs: undefined, init: undefined, fitted: true, rule: true,
        score: (i: number): number | null => fixed[data.faces[i].id] ?? null, hill: (_i: number): number | null => null };
    }
    zs = alt.zs; init = alt.init;
    // People added after this model was computed have no place in it: they sit
    // out (no score), and so do their picks.
    if (data.faces.some((f) => !alt.zs.has(f.id))) data = only(data, (f) => alt.zs.has(f.id));
    Z = data.faces.map((f) => alt.zs.get(f.id)!);
    peaks = alt.init.peaks; pg = alt.init.peak_gender ?? [];
    teamHills = alt.params;
  }
  // The straight line also learns from "keduanya kurang menarik"; the hills don't.
  const n = variant === "line" ? data.used : data.picks.length;
  const team = teamHills ? hillsFrom(JSON.parse(teamHills)) : null;
  const none = { data, n, pg: team?.pg ?? pg, params: teamHills, zs, init, fitted: false, rule: false,
    score: (_i: number): number | null => null, hill: (_i: number): number | null => null };
  if (data.picks.length < MIN_PICKS) return none;
  const cacheKey = mode === "hills" ? rater : `${rater}\t${mode}`;
  const cached = await env.DB.prepare("SELECT n_picks, params FROM rater_models WHERE rater = ?1")
    .bind(cacheKey)
    .first<{ n_picks: number; params: string }>();
  let params = cached && cached.n_picks === n ? cached.params : null;
  if (!params) {
    params = variant === "line"
      ? arrowJson(new Arrow(Z[0].length, data.groupNo.length).fit(Z, data.picks, 300, 0.05, 1e-2, data.below))
      : JSON.stringify((variant === "one" ? oneHill(Z, data.G) : new Hills(peaks, pg)).fit(Z, data.picks, data.G));
    await env.DB.prepare(
      `INSERT INTO rater_models (rater, n_picks, params, trained_at) VALUES (?1, ?2, ?3, datetime('now'))
       ON CONFLICT (rater) DO UPDATE SET n_picks = ?2, params = ?3, trained_at = datetime('now')`,
    )
      .bind(cacheKey, n, params)
      .run();
  }
  if (variant === "line") {
    // Clusters on a line come from the team's hills of the same model, as on the team view.
    const a = arrowFrom(JSON.parse(params));
    return { ...none, fitted: true, score: (i: number) => a.score(Z[i]), hill: (i: number) => team ? team.hill(Z[i], data.G[i]) : null };
  }
  const h = hillsFrom(JSON.parse(params));
  return { ...none, pg: h.pg, params, fitted: true, score: (i: number) => h.score(Z[i], data.G[i]), hill: (i: number) => h.hill(Z[i], data.G[i]) };
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

// Comparison models on a second embedding ("Model 2"): same people, same
// picks and the same held-out split, each in its own taste space. They only
// feed the scores page; battles keep using the main model.
export async function loadAlternate(env: Env, key: string) {
  const row = await env.DB.prepare("SELECT init, params, params_one, params_line, fixed FROM alt_models WHERE key = ?1")
    .bind(key).first<{ init: string; params: string | null; params_one: string | null; params_line: string | null; fixed: string | null }>();
  if (!row) return null;
  const zs = new Map(
    (await env.DB.prepare("SELECT applicant_id AS id, z FROM alt_z WHERE key = ?1").bind(key).all<{ id: number; z: string }>())
      .results.map((r) => [r.id, JSON.parse(r.z) as number[]]),
  );
  const init = JSON.parse(row.init) as { peaks: number[][]; peak_gender?: string[] };
  const fixed = row.fixed ? (JSON.parse(row.fixed) as Record<string, number>) : null;
  return { init, params: row.params, paramsOne: row.params_one, paramsLine: row.params_line, fixed, zs };
}

async function fitAlternates(
  env: Env, faces: { id: number }[], G: (string | null)[], picks: Pick[], train: Pick[], test: Pick[],
  below: Below[], belowTrain: Below[], nGroups: number, used: number,
) {
  const keys = (await env.DB.prepare("SELECT key FROM alt_models ORDER BY key").all<{ key: string }>()).results;
  const stmts: D1PreparedStatement[] = [];
  for (const { key } of keys) {
    const alt = await loadAlternate(env, key);
    if (!alt) continue;
    let heldout: number | null = null, heldoutOne: number | null = null, heldoutLine: number | null = null;
    let heldoutFixed: number | null = null;
    if (test.length >= 3 && alt.fixed) {
      // A never-trained score (missing people count as average, 0).
      heldoutFixed = agreement(faces.map((f) => alt.fixed![f.id] ?? 0), test);
    }
    if (faces.some((f) => !alt.zs.has(f.id))) {
      // No taste space (a fixed rule only), or it doesn't cover everyone.
      stmts.push(env.DB.prepare(
        "UPDATE alt_models SET heldout_fixed = ?1, n_picks = ?2, trained_at = datetime('now') WHERE key = ?3",
      ).bind(heldoutFixed, used, key));
      continue;
    }
    const Z = faces.map((f) => alt.zs.get(f.id)!);
    const fresh = () => new Hills(alt.init.peaks, alt.init.peak_gender ?? []);
    if (train.length >= MIN_PICKS && test.length >= 3) {
      const h = fresh().fit(Z, train, G);
      heldout = agreement(Z.map((z, i) => h.score(z, G[i])), test);
      const o = oneHill(Z, G).fit(Z, train, G);
      heldoutOne = agreement(Z.map((z, i) => o.score(z, G[i])), test);
      const a = new Arrow(Z[0].length, nGroups).fit(Z, train, 300, 0.05, 1e-2, belowTrain);
      heldoutLine = agreement(Z.map((z) => a.score(z)), test);
    }
    const main = fresh().fit(Z, picks, G);
    const one = oneHill(Z, G).fit(Z, picks, G);
    const line = new Arrow(Z[0].length, nGroups).fit(Z, picks, 300, 0.05, 1e-2, below);
    stmts.push(
      env.DB.prepare(
        `UPDATE alt_models SET params = ?1, heldout = ?2, params_one = ?3, heldout_one = ?4, heldout_fixed = ?5,
           n_picks = ?6, trained_at = datetime('now'), params_line = ?7, heldout_line = ?8 WHERE key = ?9`,
      ).bind(JSON.stringify(main), heldout, JSON.stringify(one), heldoutOne, heldoutFixed, used,
        arrowJson(line), heldoutLine, key),
    );
  }
  return stmts;
}

// The ranker's taste space for these faces, or null when the ranker is
// Model 1 or its model doesn't cover every face.
async function rankerZ(env: Env, faces: { id: number }[]) {
  if (RANKER === "m1") return null;
  const alt = await loadAlternate(env, RANKER);
  if (!alt || faces.some((f) => !alt.zs.has(f.id))) return null;
  return faces.map((f) => alt.zs.get(f.id)!);
}

// The ranker's straight line and taste space, as last trained (for scoring
// test faces and the rater table).
export async function rankerLine(env: Env) {
  if (RANKER !== "m1") {
    const alt = await loadAlternate(env, RANKER);
    if (alt?.paramsLine && alt.zs.size) return { line: arrowFrom(JSON.parse(alt.paramsLine)), zs: alt.zs };
  }
  const m = await env.DB.prepare("SELECT params_line FROM model WHERE id = 1").first<{ params_line: string | null }>();
  if (!m?.params_line) return null;
  const zs = new Map(
    (await env.DB.prepare("SELECT id, z FROM applicants WHERE z IS NOT NULL").all<{ id: number; z: string }>())
      .results.map((r) => [r.id, JSON.parse(r.z) as number[]]),
  );
  return { line: arrowFrom(JSON.parse(m.params_line)), zs };
}

export async function retrain(env: Env) {
  const data = await loadData(env);
  if (!data) return { status: "no taste space loaded yet" };
  const { peaks, peakGroup, faces, Z, G, picks, below, groupNo, used } = data;
  const nG = groupNo.length;
  const fresh = () => new Hills(peaks, peakGroup);
  if (picks.length < MIN_PICKS) return { status: `waiting for ${MIN_PICKS} picks`, picks: picks.length };

  // Held-out check: hills vs the straight arrow on picks they didn't see.
  const rand = rng(picks.length * 7919);
  const { by, held, train, test } = split(picks, faces.length, rand);
  const belowTrain = held ? below.filter((b) => !held[b.i]) : below;
  let heldoutHills: number | null = null, heldoutLine: number | null = null, heldoutOne: number | null = null;
  if (train.length >= MIN_PICKS && test.length >= 3) {
    const h = fresh().fit(Z, train, G);
    const a = new Arrow(Z[0].length, nG).fit(Z, train, 300, 0.05, 1e-2, belowTrain);
    heldoutHills = agreement(Z.map((z, i) => h.score(z, G[i])), test);
    heldoutLine = agreement(Z.map((z) => a.score(z)), test);
    const o = oneHill(Z, G).fit(Z, train, G);
    heldoutOne = agreement(Z.map((z, i) => o.score(z, G[i])), test);
  }
  const one = oneHill(Z, G).fit(Z, picks, G);
  const main = fresh().fit(Z, picks, G);
  // Model 1's own line, kept as a comparison when another model ranks.
  const line = new Arrow(Z[0].length, nG).fit(Z, picks, 300, 0.05, 1e-2, below);

  // The ranker: the straight arrow on every pick from every rater, in the
  // ranker's taste space (Model 1's if that model doesn't cover everyone).
  // Std = spread of 3 bootstrap fits. Model 1's hills still name each cluster.
  const Zr = await rankerZ(env, faces) ?? Z;
  const ranker = Zr === Z ? line : new Arrow(Zr[0].length, nG).fit(Zr, picks, 300, 0.05, 1e-2, below);
  const scores = Zr.map((z) => ranker.score(z));
  const hill = Z.map((z, i) => main.hill(z, G[i]));
  const runs = [0, 1, 2].map(() => {
    const boot = picks.map(() => picks[Math.floor(rand() * picks.length)]);
    const b = new Arrow(Zr[0].length, nG).fit(Zr, boot, 150, 0.05, 1e-2, below);
    const s = Zr.map((z) => b.score(z));
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
      "UPDATE applicants SET score = ?1, score_std = ?2, look = ?3, model = ?4, updated_at = ?5 WHERE id = ?6",
    ).bind(scores[i], std[i], hill[i], Zr === Z ? "line" : `${RANKER}:line`, now, f.id),
  );
  // Every unanswered pair goes (repeats too: the next one is drawn when served), including ones on screen now: they were chosen
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
      `UPDATE model SET params = ?1, active = 'line', heldout_hills = ?2, heldout_line = ?3, n_test = ?4,
       split_by = ?5, n_picks = ?6, trained_at = ?7, params_one = ?8, heldout_one = ?9, params_line = ?10 WHERE id = 1`,
    ).bind(JSON.stringify(main), heldoutHills, heldoutLine, test.length, by, used, now, JSON.stringify(one), heldoutOne,
      arrowJson(line, groupNo)),
  );
  stmts.push(...(await fitAlternates(env, faces, G, picks, train, test, below, belowTrain, nG, used)));
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
  const team = await rankerLine(env);
  const teamScores = team ? all.faces.map((f) => { const z = team.zs.get(f.id); return z ? team.line.score(z) : 0; }) : null;
  // What each rater answered, skips and "neither" included, and how fast
  // (2026-10-27: one rater answered "neither" 1,363 times in 34 minutes,
  // ~1 s each, invisible in a table that only counted winners).
  const { results: answered } = await env.DB.prepare(
    `SELECT rater, SUM(winner_id IS NOT NULL) AS winners, SUM(tie) AS ties, SUM(neither) AS neither,
       SUM(winner_id IS NULL AND tie = 0 AND neither = 0) AS skips FROM picks GROUP BY rater`,
  ).all<{ rater: string; winners: number; ties: number; neither: number; skips: number }>();
  const { results: times } = await env.DB.prepare(
    `SELECT rater, ms FROM picks WHERE ms IS NOT NULL AND (winner_id IS NOT NULL OR tie = 1 OR neither = 1) ORDER BY rater, ms`,
  ).all<{ rater: string; ms: number }>();
  const median = (n: string) => {
    const ms = times.filter((t) => t.rater === n).map((t) => t.ms);
    return ms.length ? ms[Math.floor(ms.length / 2)] : null;
  };
  const counts = new Map(answered.map(({ rater, ...c }) => [rater, c]));
  const names = [...new Set([...all.picks.map((p) => p.rater), ...answered.map((a) => a.rater)])].sort();
  const byRater = new Map(names.map((n) => [n, all.picks.filter((p) => p.rater === n)]));
  const models = new Map<string, number[] | null>();
  for (const n of names) {
    if (!byRater.get(n)!.length) { models.set(n, null); continue; } // only skips or "neither"
    const m = await raterModel(env, n);
    models.set(n, m?.fitted ? all.Z.map((_, i) => m.score(i)!) : null);
  }
  const raters = names.map((n) => ({
    name: n,
    picks: byRater.get(n)!.length,
    team: teamScores ? agreement(teamScores, byRater.get(n)!) : null,
    own_model: models.get(n) !== null,
    answers: counts.get(n) ?? { winners: 0, ties: 0, neither: 0, skips: 0 },
    median_ms: median(n),
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
      const decisive = (n: string) => byRater.get(n)!.filter((p) => p.y === 1);
      const mine = new Map(decisive(names[i]).map((p) => [key(p), p.w]));
      let n = 0, match = 0;
      for (const p of decisive(names[j])) {
        const w = mine.get(key(p));
        if (w === undefined) continue;
        n++;
        if (w === p.w) match++;
      }
      if (n) same_pair.push({ a: names[i], b: names[j], n, agree: match / n });
    }
  }
  return { raters, cross, same_pair, repeat: await repeatAgreement(env) };
}

// The same rater on the same pair twice (a repeat battle against their
// latest earlier answer). How often they pick the same face again is the
// ceiling for any model: nobody can predict them better than they agree
// with themselves. Only answers with a winner both times count.
async function repeatAgreement(env: Env) {
  const { results } = await env.DB.prepare(
    `SELECT r.rater, r.winner_id = o.winner_id AS same FROM picks r JOIN picks o ON o.id = (
       SELECT MAX(x.id) FROM picks x WHERE x.rater = r.rater AND x.kind != 'repeat' AND x.id < r.id
         AND ((x.left_id = r.left_id AND x.right_id = r.right_id) OR (x.left_id = r.right_id AND x.right_id = r.left_id)))
     WHERE r.kind = 'repeat' AND r.winner_id IS NOT NULL AND o.winner_id IS NOT NULL`,
  ).all<{ rater: string; same: number }>();
  const names = [...new Set(results.map((r) => r.rater))].sort();
  const sum = (rows: { same: number }[]) => ({ n: rows.length, same: rows.length ? rows.filter((r) => r.same).length / rows.length : null });
  return { ...sum(results), raters: names.map((name) => ({ name, ...sum(results.filter((r) => r.rater === name)) })) };
}
