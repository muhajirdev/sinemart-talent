// Sinemart A/B picking app: serves pairs of applicant crops, records picks,
// retrains the models as picks come in, and reports scores.
// Static UI lives in ../public; everything under /api and /img lands here.

import { territory } from "./map";
import {
  ANSWERED, MIN_PICKS, RANKER, REPEAT_EVERY, RETRAIN_EVERY, arrowFrom, hillsFrom, loadAlternate, rankerLine, raterAgreement, raterModel, retrain, withoutRaters,
} from "./retrain";

export interface Env {
  DB: D1Database;
  IMAGES: R2Bucket;
  ADMIN_TOKEN: string;
}

type Rater = { token: string; name: string };

// The app is open to anyone, but applicant photos must stay out of search engines.
const NOINDEX = "noindex, nofollow, noimageindex";

const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store", "x-robots-tag": NOINDEX },
  });

async function rater(req: Request, env: Env): Promise<Rater | null> {
  const url = new URL(req.url);
  const token = req.headers.get("x-rater-token") ?? url.searchParams.get("t");
  if (!token) return null;
  return env.DB.prepare("SELECT token, name FROM raters WHERE token = ?1").bind(token).first<Rater>();
}

// Name-only sign-in, open to anyone (Muajir, 2026-10-12). The same name (any
// case) is the same rater, so picks stay traceable to a person across devices.
async function login(req: Request, env: Env) {
  const body = (await req.json().catch(() => ({}))) as { name?: string };
  const name = String(body.name ?? "").replace(/\s+/g, " ").trim().slice(0, 40);
  if (name.length < 2) return json({ error: "type your name" }, 400);
  const known = await env.DB.prepare("SELECT token, name FROM raters WHERE lower(name) = lower(?1)")
    .bind(name)
    .first<Rater>();
  if (known) return json(known);
  const token = [...crypto.getRandomValues(new Uint8Array(18))].map((b) => b.toString(16).padStart(2, "0")).join("");
  await env.DB.prepare("INSERT INTO raters (token, name) VALUES (?1, ?2)").bind(token, name).run();
  return json({ token, name });
}

// Each side of a battle: the person, plus how many whole photos they have.
async function withPhotos(env: Env, pair: { queue_id: number | null; left_id: number; right_id: number }) {
  const { results } = await env.DB.prepare(
    "SELECT applicant_id AS id, kind FROM photos WHERE applicant_id IN (?1, ?2) ORDER BY applicant_id, n",
  ).bind(pair.left_id, pair.right_id).all<{ id: number; kind: string }>();
  const kinds = (id: number) => results.filter((r) => r.id === id).map((r) => r.kind);
  const [lp, rp] = await Promise.all([facePoints(env, pair.left_id), facePoints(env, pair.right_id)]);
  const g = new Map((await env.DB.prepare("SELECT id, gender FROM applicants WHERE id IN (?1, ?2)")
    .bind(pair.left_id, pair.right_id).all<{ id: number; gender: string | null }>()).results.map((r) => [r.id, r.gender]));
  return {
    ...pair, left_photos: kinds(pair.left_id), right_photos: kinds(pair.right_id), left_points: lp, right_points: rp,
    left_gender: g.get(pair.left_id) ?? null, right_gender: g.get(pair.right_id) ?? null,
  };
}

// Face points on a person's main face crop (what /img/<id> shows), or null.
async function facePoints(env: Env, id: number) {
  const r = await env.DB.prepare(
    "SELECT f.points, f.front, f.yaw, f.pitch FROM face_points f JOIN applicants a ON a.image_key = f.image_key WHERE a.id = ?1",
  ).bind(id).first<{ points: string; front: string | null; yaw: number; pitch: number }>();
  return r ? { points: JSON.parse(r.points), front: r.front ? JSON.parse(r.front) : null, yaw: r.yaw, pitch: r.pitch } : null;
}

// About 1 in REPEAT_EVERY battles: a pair this rater answered at least 20
// of their picks ago, sides swapped, never repeated before. It goes in the
// queue already served to them, so the pick is recorded as kind 'repeat'.
async function repeatPair(env: Env, who: Rater) {
  if (Math.random() * REPEAT_EVERY >= 1) return null;
  const old = await env.DB.prepare(
    `SELECT p.left_id, p.right_id FROM picks p
     WHERE p.rater = ?1 AND p.winner_id IS NOT NULL AND p.kind != 'repeat'
       AND p.id < (SELECT id FROM picks WHERE rater = ?1 ORDER BY id DESC LIMIT 1 OFFSET 19)
       AND NOT EXISTS (SELECT 1 FROM picks r WHERE r.rater = ?1 AND r.kind = 'repeat'
         AND ((r.left_id = p.left_id AND r.right_id = p.right_id) OR (r.left_id = p.right_id AND r.right_id = p.left_id)))
     ORDER BY RANDOM() LIMIT 1`,
  ).bind(who.name).first<{ left_id: number; right_id: number }>();
  return old ? queueRepeat(env, who, old.right_id, old.left_id) : null;
}

async function queueRepeat(env: Env, who: Rater, left_id: number, right_id: number) {
  const q = await env.DB.prepare(
    `INSERT INTO pair_queue (left_id, right_id, kind, batch, served_to, served_at)
     VALUES (?1, ?2, 'repeat', 'repeat', ?3, datetime('now')) RETURNING id`,
  ).bind(left_id, right_id, who.name).first<{ id: number }>();
  return { queue_id: q!.id, left_id, right_id };
}

async function nextPair(env: Env, who: Rater) {
  const repeat = await repeatPair(env, who);
  if (repeat) return repeat;
  // Claim the oldest open pair; a pair served but unanswered for 10 minutes
  // goes back into the pool. Repeats belong to the rater they were made for.
  const row = await env.DB.prepare(
    `UPDATE pair_queue SET served_to = ?1, served_at = datetime('now')
     WHERE id = (
       SELECT id FROM pair_queue
       WHERE done_at IS NULL AND kind != 'repeat' AND (served_at IS NULL OR served_at < datetime('now', '-10 minutes'))
       ORDER BY id LIMIT 1)
     RETURNING id, left_id, right_id`,
  )
    .bind(who.name)
    .first<{ id: number; left_id: number; right_id: number }>();
  if (row) return { queue_id: row.id, left_id: row.left_id, right_id: row.right_id };

  // Queue empty (cold start, or the trainer hasn't refilled it): random pair.
  // Same group only: cewek vs cewek, cowok vs cowok.
  const a = await env.DB.prepare("SELECT id, gender FROM applicants WHERE noise IS NULL AND test = 0 ORDER BY RANDOM() LIMIT 1").first<{ id: number; gender: string | null }>();
  if (!a) return null;
  const b = await env.DB.prepare(
    "SELECT id FROM applicants WHERE id != ?1 AND gender IS ?2 AND noise IS NULL AND test = 0 ORDER BY RANDOM() LIMIT 1",
  ).bind(a.id, a.gender).first<{ id: number }>();
  if (!b) return null;
  return { queue_id: null, left_id: a.id, right_id: b.id };
}

async function recordPick(req: Request, env: Env, who: Rater, ctx: ExecutionContext) {
  const body = (await req.json()) as {
    queue_id: number | null;
    left_id: number;
    right_id: number;
    winner_id: number | null;
    tie?: boolean; // "Sama saja": both about equal
    neither?: boolean; // "Keduanya kurang menarik": both are a clear no
    ms?: number;
  };
  const { queue_id, left_id, right_id, winner_id } = body;
  const neither = body.neither === true && winner_id === null ? 1 : 0;
  const tie = body.tie === true && winner_id === null && !neither ? 1 : 0;
  if (!Number.isInteger(left_id) || !Number.isInteger(right_id) || left_id === right_id) {
    return json({ error: "bad pair" }, 400);
  }
  if (winner_id !== null && winner_id !== left_id && winner_id !== right_id) {
    return json({ error: "winner must be left, right or null" }, 400);
  }
  let kind = "fallback";
  if (queue_id !== null) {
    const q = await env.DB.prepare("SELECT kind, left_id, right_id FROM pair_queue WHERE id = ?1")
      .bind(queue_id)
      .first<{ kind: string; left_id: number; right_id: number }>();
    if (q && (q.left_id !== left_id || q.right_id !== right_id)) return json({ error: "unknown pair" }, 400);
    kind = q?.kind ?? "fallback"; // a retrain may have cleared the queue while this pair was on screen
  }
  const ms = Number.isFinite(body.ms) ? Math.round(body.ms!) : null;
  const stmts = [
    env.DB.prepare(
      "INSERT INTO picks (rater, left_id, right_id, winner_id, tie, kind, ms, neither) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    ).bind(who.name, left_id, right_id, winner_id, tie, kind, ms, neither),
  ];
  if (queue_id !== null) {
    stmts.push(env.DB.prepare("UPDATE pair_queue SET done_at = datetime('now') WHERE id = ?1").bind(queue_id));
  }
  await env.DB.batch(stmts);
  return json({ ok: true });
}

// Take back this rater's last answer (from the last 30 minutes) and show the
// same pair again. Skips can be taken back too.
async function undoPick(env: Env, who: Rater) {
  const last = await env.DB.prepare(
    `SELECT id, left_id, right_id, winner_id, tie, neither, kind FROM picks
     WHERE rater = ?1 AND created_at > datetime('now', '-30 minutes') ORDER BY id DESC LIMIT 1`,
  ).bind(who.name).first<{
    id: number; left_id: number; right_id: number; winner_id: number | null; tie: number; neither: number; kind: string;
  }>();
  if (!last) return json({ error: "nothing to undo" }, 404);
  await env.DB.prepare("DELETE FROM picks WHERE id = ?1").bind(last.id).run();
  // A repeat stays a repeat, so the new answer doesn't count twice in training.
  const again = last.kind === "repeat"
    ? await queueRepeat(env, who, last.left_id, last.right_id)
    : { queue_id: null, left_id: last.left_id, right_id: last.right_id };
  const pair = await withPhotos(env, again);
  return json({ ...pair, was_pick: last.winner_id !== null || last.tie === 1 || last.neither === 1 });
}

// A change that isn't a pick (a gender fix, a restored person): make the next
// cron run retrain, by marking the model as trained on no picks.
async function retrainSoon(env: Env) {
  await env.DB.prepare("UPDATE model SET n_picks = 0 WHERE id = 1").run();
}

// Runs from the every-minute cron, not after each pick: with 840 people and
// eight models a retrain outlasts the ~30 s a request may keep working after
// its response, so it was cut off and the scores froze. Counts picks the way
// retrain does (no repeats, same group, both people scored and not marked
// bad), so n_picks and this count can actually meet.
async function maybeRetrain(env: Env) {
  const row = await env.DB.prepare(
    `SELECT (SELECT COUNT(*) FROM picks p JOIN applicants a ON a.id = p.left_id JOIN applicants b ON b.id = p.right_id
             WHERE (p.winner_id IS NOT NULL OR p.tie = 1 OR p.neither = 1) AND p.kind != 'repeat' AND a.gender IS b.gender
               AND a.z IS NOT NULL AND b.z IS NOT NULL AND a.noise IS NULL AND b.noise IS NULL AND a.test = 0 AND b.test = 0) AS picks,
            (SELECT n_picks FROM model WHERE id = 1) AS trained`,
  ).first<{ picks: number; trained: number | null }>();
  if (row && row.trained !== null && row.picks - row.trained >= RETRAIN_EVERY) await retrain(env);
  else if (row && row.trained === 0 && row.picks >= 5) await retrain(env);
}

// A comparison model reloaded by the pipeline comes back untrained (and so
// does the straight line right after it became the ranker): fit them before
// the scores page shows blanks.
async function retrainIfStale(env: Env) {
  const r = await env.DB.prepare(
    `SELECT (SELECT n_picks >= ?1 AND params_line IS NULL FROM model WHERE id = 1)
       OR EXISTS (SELECT 1 FROM alt_models m WHERE m.params IS NULL AND EXISTS (SELECT 1 FROM alt_z z WHERE z.key = m.key)) AS stale`,
  ).bind(MIN_PICKS).first<{ stale: number }>();
  if (r?.stale) await retrain(env);
}

// Face-geometry numbers for the picking page's "Golden ratio" toggle: each rule
// as a percentile within the person's gender (100 = closest to its ideal).
async function geometryPercentiles(env: Env) {
  const { results: rows } = await env.DB.prepare(
    `SELECT g.applicant_id AS id, a.gender, g.phi_dev, g.asymmetry, g.atypicality, g.pallett_dev
     FROM geometry g JOIN applicants a ON a.id = g.applicant_id`,
  ).all<Record<string, number | string | null>>();
  const out: Record<number, Record<string, number>> = {};
  for (const r of rows) out[r.id as number] = {};
  const keys = [["phi_dev", "phi"], ["asymmetry", "sym"], ["atypicality", "avg"], ["pallett_dev", "pal"]];
  for (const g of new Set(rows.map((r) => r.gender))) {
    const grp = rows.filter((r) => r.gender === g);
    for (const [k, short] of keys) {
      const have = grp.filter((r) => r[k] != null).sort((a, b) => (b[k] as number) - (a[k] as number)); // worst first
      have.forEach((r, i) => (out[r.id as number][short] = have.length > 1 ? Math.round((i / (have.length - 1)) * 100) : 50));
    }
  }
  return out;
}

async function results(env: Env, who: string | null, mode: string | null = null, without: string[] = []) {
  if (who) return raterResults(env, who, mode);
  const model = await env.DB.prepare(
    `SELECT active, heldout_hills, heldout_one, heldout_line, n_test, split_by, n_picks, trained_at, init, params, params_one
     FROM model WHERE id = 1`,
  ).first<Record<string, unknown>>();
  const oneParams = model?.params_one as string | null, hillsParams = model?.params as string | null;
  const alternates = (await env.DB.prepare(`SELECT key, heldout, heldout_one, heldout_line, heldout_fixed, fixed IS NOT NULL AS has_fixed,
       EXISTS (SELECT 1 FROM alt_z z WHERE z.key = m.key) AS has_z, n_picks FROM alt_models m ORDER BY key`).all()).results;
  if (model) { delete model.params_one; delete model.params; }
  const picks = await env.DB.prepare(
    `SELECT COUNT(*) AS decided, SUM(tie) AS ties, SUM(neither) AS neithers, (SELECT COUNT(*) FROM picks) AS total, COUNT(DISTINCT rater) AS raters
     FROM picks WHERE ${ANSWERED}`,
  ).first<{ decided: number; total: number; raters: number }>();
  const { results: faces } = await env.DB.prepare(
    `SELECT a.id, a.score, a.score_std, a.look, a.gender, a.gender_src,
       (SELECT COUNT(*) FROM picks p WHERE p.winner_id = a.id) AS wins,
       (SELECT COUNT(*) FROM picks p WHERE (p.winner_id IS NOT NULL OR p.tie = 1 OR p.neither = 1) AND (p.left_id = a.id OR p.right_id = a.id)) AS games,
       (SELECT COUNT(*) FROM photos ph WHERE ph.applicant_id = a.id) AS n_photos,
       g.phi_dev, g.pallett_dev, g.asymmetry, g.atypicality, a.source, a.test, CASE WHEN a.test = 1 THEN a.z END AS z
     FROM applicants a LEFT JOIN geometry g ON g.applicant_id = a.id WHERE a.noise IS NULL ORDER BY a.score DESC NULLS LAST, a.id`,
  ).all<Record<string, unknown>>();
  // Retrains never touch test faces, so their stored score is from load
  // time: score them with the ranker's current straight line instead.
  const rl = await rankerLine(env);
  for (const f of faces) {
    if (f.test) { const z = rl?.zs.get(f.id as number); f.score = rl && z ? rl.line.score(z) : null; }
    delete f.z;
  }
  let peakGender: (string | null)[] | null = null;
  let preview: unknown = null;
  if (mode === "1:line") {
    // Model 1's own straight line, a comparison once another model ranks.
    const p = (await env.DB.prepare("SELECT params_line FROM model WHERE id = 1").first<{ params_line: string | null }>())?.params_line;
    const line1 = p ? arrowFrom(JSON.parse(p)) : null;
    const zs = new Map(
      (await env.DB.prepare("SELECT id, z FROM applicants WHERE z IS NOT NULL").all<{ id: number; z: string }>())
        .results.map((r) => [r.id, JSON.parse(r.z) as number[]]),
    );
    for (const f of faces) {
      const z = zs.get(f.id as number);
      f.score = line1 && z ? line1.score(z) : null;
      f.score_std = null;
    }
    faces.sort((a, b) => ((b.score as number) ?? -Infinity) - ((a.score as number) ?? -Infinity));
  } else if (mode && mode !== "one" && mode !== "hills" && mode !== "line") {
    // Model 2 (another embedding), trained on the same picks. "m2" is that
    // embedding with many hills, "m2:one" one hill per group, "m2:line" the
    // straight arrow (clusters still from its hills).
    const [key, variant] = mode.split(":");
    const alt = await loadAlternate(env, key);
    const params = variant === "one" ? alt?.paramsOne : alt?.params;
    const h = params ? hillsFrom(JSON.parse(params)) : null;
    const line = variant === "line" && alt?.paramsLine ? arrowFrom(JSON.parse(alt.paramsLine)) : null;
    peakGender = h ? h.pg : alt ? (alt.init.peak_gender ?? []) : [];
    if (alt && !alt.zs.size && alt.fixed) {
      // A fixed rule ("Golden ratio"): no clusters, ranked by its own score.
      for (const f of faces) { f.score = alt.fixed[f.id as number] ?? null; f.look = null; f.score_std = null; }
    } else for (const f of faces) {
      const z = alt?.zs.get(f.id as number), g = f.gender as string | null;
      f.score = !z ? null : variant === "line" ? (line ? line.score(z) : null) : h ? h.score(z, g) : null;
      f.look = h && z ? h.hill(z, g) : null;
      f.score_std = null;
    }
    faces.sort((a, b) => ((b.score as number) ?? -Infinity) - ((a.score as number) ?? -Infinity));
  } else if (!mode || mode === "line") {
    faces.sort((a, b) => ((b.score as number) ?? -Infinity) - ((a.score as number) ?? -Infinity));
  }
  if (mode === "one" || mode === "hills") {
    // Same page, ranked by Model 1's many-hills or one-hill-per-group model.
    const p = mode === "one" ? oneParams : hillsParams;
    const one = p ? hillsFrom(JSON.parse(p)) : null;
    peakGender = one ? one.pg : [];
    const zs = new Map(
      (await env.DB.prepare("SELECT id, z FROM applicants WHERE z IS NOT NULL").all<{ id: number; z: string }>())
        .results.map((r) => [r.id, JSON.parse(r.z) as number[]]),
    );
    for (const f of faces) {
      const z = zs.get(f.id as number), g = f.gender as string | null;
      f.score = one && z ? one.score(z, g) : null;
      f.look = one && z ? one.hill(z, g) : null;
      f.score_std = null;
    }
    faces.sort((a, b) => ((b.score as number) ?? -Infinity) - ((a.score as number) ?? -Infinity));
  }
  if (without.length && (!mode || mode === "line" || mode === `${RANKER}:line`)) {
    // Preview of the ranker with some raters left out (the Lab's rater
    // checklist). Clusters stay as they are; only the scores change.
    const w = await withoutRaters(env, without);
    for (const f of faces) {
      const z = w?.zs.get(f.id as number);
      f.score = w?.line && z ? w.line.score(z) : null;
      f.score_std = null;
    }
    faces.sort((a, b) => ((b.score as number) ?? -Infinity) - ((a.score as number) ?? -Infinity));
    preview = w?.summary ?? null;
  }
  const elo = await eloRatings(env, { without });
  for (const f of faces) f.elo = elo.get(f.id as number) ?? null;
  const init = model?.init ? JSON.parse(model.init as string) : null;
  if (model) delete model.init;
  return json({
    model: {
      ...model, mode: mode ?? "line", ranker: RANKER, hills: init?.peaks?.length ?? 0, alternates,
      peak_gender: peakGender ?? init?.peak_gender ?? [],
      retrain_every: RETRAIN_EVERY, without: preview,
    },
    picks,
    faces,
  });
}

// The same page, scored by one rater's personal model.
async function raterResults(env: Env, who: string, mode: string | null = null) {
  const m = await raterModel(env, who, mode ?? "line");
  if (!m) return json({ error: "no taste space loaded yet" }, 404);
  const { data, n, fitted } = m;
  const counts = await env.DB.prepare(
    `SELECT SUM(${ANSWERED}) AS decided, COUNT(*) AS total FROM picks WHERE rater = ?1`,
  ).bind(who).first<{ decided: number; total: number }>();
  const photoCount = new Map(
    (await env.DB.prepare("SELECT applicant_id AS id, COUNT(*) AS n FROM photos GROUP BY applicant_id").all<{ id: number; n: number }>())
      .results.map((r) => [r.id, r.n]),
  );
  const geo = new Map(
    (await env.DB.prepare("SELECT applicant_id AS id, phi_dev, pallett_dev, asymmetry, atypicality FROM geometry").all<Record<string, number>>())
      .results.map(({ id, ...g }) => [id, g]),
  );
  const wins = new Map<number, number>(), games = new Map<number, number>();
  for (const p of data.picks) {
    const [w, l] = [data.faces[p.w].id, data.faces[p.l].id];
    if (p.y === 1) wins.set(w, (wins.get(w) ?? 0) + 1);
    games.set(w, (games.get(w) ?? 0) + 1);
    games.set(l, (games.get(l) ?? 0) + 1);
  }
  const elo = await eloRatings(env, { only: who });
  const faces = data.faces
    .map((f, i) => ({
      id: f.id,
      score: m.score(i),
      elo: elo.get(f.id) ?? null,
      score_std: null,
      look: m.hill(i),
      gender: data.G[i],
      gender_src: null,
      wins: wins.get(f.id) ?? 0,
      games: games.get(f.id) ?? 0,
      n_photos: photoCount.get(f.id) ?? 0,
      ...geo.get(f.id),
    }))
    .sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity) || a.id - b.id);
  // The model switch needs the list of models, same as the team view.
  const alternates = (await env.DB.prepare(`SELECT key, fixed IS NOT NULL AS has_fixed,
       EXISTS (SELECT 1 FROM alt_z z WHERE z.key = m.key) AS has_z FROM alt_models m ORDER BY key`).all()).results;
  return json({
    rater: who,
    model: {
      mode: mode ?? "line", ranker: RANKER, alternates,
      active: fitted ? "rater" : null, n_picks: fitted ? n : 0, trained_at: fitted ? new Date().toISOString() : null,
      hills: data.peaks.length, peak_gender: m.pg, min_picks: MIN_PICKS, retrain_every: RETRAIN_EVERY,
    },
    picks: { decided: counts?.decided ?? 0, total: counts?.total ?? 0, raters: 1 },
    faces,
    params: m.params,
  });
}

// Fix a wrong cewek/cowok guess by hand. It moves the person to the other
// group's hills and battles, so every model refits.
async function setGender(req: Request, env: Env, ctx: ExecutionContext) {
  const { id, gender } = (await req.json().catch(() => ({}))) as { id?: number; gender?: string };
  if (!Number.isInteger(id) || (gender !== "f" && gender !== "m")) return json({ error: "id and gender f|m" }, 400);
  const r = await env.DB.prepare("UPDATE applicants SET gender = ?1, gender_src = 'manual' WHERE id = ?2").bind(gender, id).run();
  if (!r.meta.changes) return json({ error: "unknown applicant" }, 404);
  await env.DB.prepare("DELETE FROM rater_models").run();
  await retrainSoon(env);
  return json({ ok: true });
}

// Mark a person's photos as bad data (or restore them). Marked people drop
// out of battles, training and the leaderboard; their picks stay stored.
async function setNoise(req: Request, env: Env, who: Rater, ctx: ExecutionContext) {
  const { id, noise } = (await req.json().catch(() => ({}))) as { id?: number; noise?: boolean };
  if (!Number.isInteger(id) || typeof noise !== "boolean") return json({ error: "id and noise true|false" }, 400);
  const r = await env.DB.prepare("UPDATE applicants SET noise = ?1 WHERE id = ?2").bind(noise ? who.name : null, id).run();
  if (!r.meta.changes) return json({ error: "unknown applicant" }, 404);
  if (noise) {
    await env.DB.prepare("DELETE FROM pair_queue WHERE done_at IS NULL AND (left_id = ?1 OR right_id = ?1)").bind(id).run();
  } else await retrainSoon(env);
  return json({ ok: true });
}

// How the ranking compares with an outside human screening, in aggregate
// only: the chance a passed person outranks a not-passed one of the same
// group (0.5 = no better than chance), and how many passed land in the top 50.
async function outsideCheck(env: Env) {
  const { results } = await env.DB.prepare(
    `SELECT l.source, l.label, a.gender, a.score FROM outside_labels l JOIN applicants a ON a.id = l.applicant_id
     WHERE a.score IS NOT NULL AND a.noise IS NULL AND a.test = 0`,
  ).all<{ source: string; label: number; gender: string; score: number }>();
  const out = [];
  for (const source of [...new Set(results.map((r) => r.source))]) {
    let wins = 0, pairs = 0, passed = 0, inTop = 0, people = 0, expected = 0;
    for (const g of ["f", "m"]) {
      const rows = results.filter((r) => r.source === source && r.gender === g).sort((a, b) => b.score - a.score);
      const pos = rows.filter((r) => r.label === 1), neg = rows.filter((r) => r.label !== 1);
      for (const p of pos) for (const q of neg) { pairs++; wins += p.score > q.score ? 1 : p.score === q.score ? 0.5 : 0; }
      passed += pos.length;
      people += rows.length;
      inTop += rows.slice(0, 50).filter((r) => r.label === 1).length;
      if (rows.length) expected += (pos.length * Math.min(50, rows.length)) / rows.length; // by chance
    }
    out.push({ source, people, passed, auc: pairs ? wins / pairs : null, passed_in_top50: inTop, expected_in_top50: expected });
  }
  return out;
}

// 2026-10-28 (Muajir: "an elo please (not model prediction), just based on
// their lost / win"). Plain Elo over every answered battle in order: start
// 1500, K = 32, "sama saja" a draw. Skips, "neither" and repeat battles
// don't count. Optionally one rater's picks only, or without some raters.
const ELO_START = 1500, ELO_K = 32;
async function eloRatings(env: Env, opts: { only?: string; without?: string[] } = {}) {
  const { results } = await env.DB.prepare(
    `SELECT rater, left_id, right_id, winner_id FROM picks
     WHERE (winner_id IS NOT NULL OR tie = 1) AND kind != 'repeat' AND (?1 IS NULL OR rater = ?1) ORDER BY id`,
  ).bind(opts.only ?? null).all<{ rater: string; left_id: number; right_id: number; winner_id: number | null }>();
  const drop = new Set(opts.without ?? []);
  const elo = new Map<number, number>();
  for (const p of results) {
    if (drop.has(p.rater)) continue;
    const a = elo.get(p.left_id) ?? ELO_START, b = elo.get(p.right_id) ?? ELO_START;
    const expect = 1 / (1 + 10 ** ((b - a) / 400));
    const got = p.winner_id == null ? 0.5 : p.winner_id === p.left_id ? 1 : 0;
    elo.set(p.left_id, a + ELO_K * (got - expect));
    elo.set(p.right_id, b - ELO_K * (got - expect));
  }
  return elo;
}

// Every battle one person was in, newest first: who they faced, how it went
// and who judged it (skips included, so a much-skipped face stands out).
async function battleHistory(env: Env, id: number) {
  const { results } = await env.DB.prepare(
    `SELECT CASE WHEN left_id = ?1 THEN right_id ELSE left_id END AS other,
       CASE WHEN winner_id = ?1 THEN 'won' WHEN winner_id IS NOT NULL THEN 'lost' WHEN tie = 1 THEN 'tie' WHEN neither = 1 THEN 'neither' ELSE 'skip' END AS result,
       rater, kind = 'repeat' AS repeat, created_at
     FROM picks WHERE left_id = ?1 OR right_id = ?1 ORDER BY id DESC LIMIT 200`,
  ).bind(id).all();
  return results;
}

// 2026-10-28 (Muajir): one rater's pick history, newest first, for the side
// panel on the Lab page: both faces, what they answered and how fast.
async function raterHistory(env: Env, name: string) {
  const [{ results }, counts] = await Promise.all([
    env.DB.prepare(
      `SELECT left_id, right_id, winner_id, tie, neither, kind = 'repeat' AS repeat, ms, created_at
       FROM picks WHERE rater = ?1 ORDER BY id DESC LIMIT 200`,
    ).bind(name).all(),
    env.DB.prepare(
      `SELECT COUNT(*) AS total, SUM(winner_id IS NOT NULL) AS picked, SUM(tie) AS ties, SUM(neither) AS neither
       FROM picks WHERE rater = ?1`,
    ).bind(name).first(),
  ]);
  return { counts, rows: results };
}

async function exportPicks(env: Env) {
  // For the trainer: one line per decided pick (skips dropped).
  const { results } = await env.DB.prepare(
    `SELECT winner_id, CASE WHEN winner_id = left_id THEN right_id ELSE left_id END AS loser_id, rater, kind, created_at
     FROM picks WHERE winner_id IS NOT NULL ORDER BY id`,
  ).all<Record<string, string | number>>();
  const cols = ["winner_id", "loser_id", "rater", "kind", "created_at"];
  const lines = [cols.join(","), ...results.map((r) => cols.map((c) => r[c]).join(","))];
  return new Response(lines.join("\n") + "\n", { headers: { "content-type": "text/csv" } });
}

export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);

    if (url.pathname === "/api/export.csv") {
      if (!env.ADMIN_TOKEN || url.searchParams.get("admin") !== env.ADMIN_TOKEN) return json({ error: "forbidden" }, 403);
      return exportPicks(env);
    }

    if (url.pathname === "/api/login" && req.method === "POST") return login(req, env);

    const who = await rater(req, env);
    if (!who) return json({ error: "sign in first" }, 401);
    const only = url.searchParams.get("rater"); // results/map for one rater's taste

    if (url.pathname === "/api/geometry") return json(await geometryPercentiles(env));
    const bh = url.pathname.match(/^\/api\/battles\/(\d+)$/);
    if (bh) return json(await battleHistory(env, Number(bh[1])));
    if (url.pathname === "/api/rater-picks") return json(await raterHistory(env, url.searchParams.get("name") ?? ""));
    const fp = url.pathname.match(/^\/api\/points\/(\d+)$/);
    if (fp) return json(await facePoints(env, Number(fp[1])));
    if (url.pathname === "/api/me") {
      const mine = await env.DB.prepare(`SELECT COUNT(*) AS n FROM picks WHERE rater = ?1 AND ${ANSWERED}`).bind(who.name).first<{ n: number }>();
      return json({ name: who.name, picks: mine?.n ?? 0 });
    }
    if (url.pathname === "/api/pair" && req.method === "GET") {
      const pair = await nextPair(env, who);
      return pair ? json(await withPhotos(env, pair)) : json({ error: "no applicants loaded" }, 404);
    }
    if (url.pathname === "/api/pick" && req.method === "POST") return recordPick(req, env, who, ctx);
    if (url.pathname === "/api/undo" && req.method === "POST") return undoPick(env, who);
    // Default "line" (the ranker); "hills" / "one" = Model 1's cluster models; "m2", "m2:one", "m2:line" = Model 2...
    const mode = url.searchParams.get("mode");
    if (url.pathname === "/api/results" && req.method === "GET") {
      await retrainIfStale(env);
      const without = (url.searchParams.get("without") ?? "").split(",").filter(Boolean);
      return results(env, only, mode, without);
    }
    if (url.pathname === "/api/map" && req.method === "GET") {
      const gender = url.searchParams.get("gender");
      if (only) {
        // One rater's taste: their own scores, on the chosen model's map.
        const m = await raterModel(env, only, mode ?? "line");
        if (!m || m.rule) return json({ faces: [], peaks: [], grid: null }); // a fixed rule has no map
        const faces = new Map(m.data.faces.map((f, i) => [f.id, { score: m.score(i), hill: m.hill(i) }]));
        return json(await territory(env, { params: m.params, faces, zs: m.zs, init: m.init }, gender));
      }
      if (mode && mode !== "line") {
        const r = (await (await results(env, null, mode)).json()) as { faces: { id: number; score: number | null; look: number | null }[] };
        const faces = new Map(r.faces.map((f) => [f.id, { score: f.score, hill: f.look }]));
        if (mode === "one" || mode === "hills" || mode === "1:line") {
          // Model 1's line shows Model 1's many-hills clusters, like the ranker does.
          const m = await env.DB.prepare("SELECT params, params_one FROM model WHERE id = 1")
            .first<{ params: string | null; params_one: string | null }>();
          return json(await territory(env, { params: (mode === "one" ? m?.params_one : m?.params) ?? null, faces }, gender));
        }
        const [key, variant] = mode.split(":");
        const alt = await loadAlternate(env, key);
        if (!alt || !alt.zs.size) return json({ faces: [], peaks: [], grid: null }); // a fixed rule has no map
        const params = variant === "one" ? alt.paramsOne : alt.params; // the line's map shows its hills' clusters
        return json(await territory(env, { params, faces, zs: alt.zs, init: alt.init }, gender));
      }
      return json(await territory(env, undefined, gender));
    }
    if (url.pathname === "/api/gender" && req.method === "POST") return setGender(req, env, ctx);
    if (url.pathname === "/api/noise" && req.method === "POST") return setNoise(req, env, who, ctx);
    if (url.pathname === "/api/noise" && req.method === "GET") {
      return json((await env.DB.prepare("SELECT id, gender, noise AS by FROM applicants WHERE noise IS NOT NULL ORDER BY id").all()).results);
    }
    if (url.pathname === "/api/check" && req.method === "GET") return json(await outsideCheck(env));
    if (url.pathname === "/api/raters" && req.method === "GET") return json(await raterAgreement(env));
    if (url.pathname === "/api/retrain" && req.method === "POST") return json(await retrain(env));
    if (url.pathname === "/api/reset" && req.method === "POST") {
      // Start testing fresh: forget every pick and everything learned from them.
      await env.DB.batch([
        env.DB.prepare("DELETE FROM picks"),
        env.DB.prepare("DELETE FROM pair_queue"),
        env.DB.prepare("DELETE FROM rater_models"),
        env.DB.prepare("UPDATE applicants SET score = NULL, score_std = NULL, model = NULL, updated_at = NULL"),
        env.DB.prepare(
          `UPDATE model SET params = NULL, active = NULL, heldout_hills = NULL, heldout_line = NULL, params_line = NULL,
           params_one = NULL, heldout_one = NULL, n_test = NULL, split_by = NULL, n_picks = 0, trained_at = NULL WHERE id = 1`,
        ),
        env.DB.prepare(
          "UPDATE alt_models SET params = NULL, heldout = NULL, params_one = NULL, heldout_one = NULL, params_line = NULL, heldout_line = NULL, heldout_fixed = NULL, n_picks = 0, trained_at = NULL",
        ),
      ]);
      return json({ ok: true });
    }

    // /img/<id> = the person's main face crop; /img/<id>/<n> = their n-th whole photo.
    const img = url.pathname.match(/^\/img\/(\d+)(?:\/(\d+))?$/);
    if (img) {
      const row = img[2] === undefined
        ? await env.DB.prepare("SELECT image_key FROM applicants WHERE id = ?1").bind(Number(img[1])).first<{ image_key: string }>()
        : await env.DB.prepare("SELECT image_key FROM photos WHERE applicant_id = ?1 AND n = ?2")
            .bind(Number(img[1]), Number(img[2])).first<{ image_key: string }>();
      const obj = row ? await env.IMAGES.get(row.image_key) : null;
      if (!obj) return new Response("not found", { status: 404 });
      return new Response(obj.body, {
        headers: { "content-type": "image/jpeg", "cache-control": "private, max-age=86400", "x-robots-tag": NOINDEX },
      });
    }
    return json({ error: "not found" }, 404);
  },
  async scheduled(_event: ScheduledController, env: Env, ctx: ExecutionContext) {
    ctx.waitUntil(maybeRetrain(env));
  },
} satisfies ExportedHandler<Env>;
