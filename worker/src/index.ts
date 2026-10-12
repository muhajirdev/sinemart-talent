// Sinemart A/B picking app: serves pairs of applicant crops, records picks,
// retrains the hills model as picks come in, and reports scores.
// Static UI lives in ../public; everything under /api and /img lands here.

import { territory } from "./map";
import { MIN_PICKS, RETRAIN_EVERY, raterAgreement, raterModel, retrain } from "./retrain";

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
  return { ...pair, left_photos: kinds(pair.left_id), right_photos: kinds(pair.right_id) };
}

async function nextPair(env: Env, who: Rater) {
  // Claim the oldest open pair; a pair served but unanswered for 10 minutes
  // goes back into the pool.
  const row = await env.DB.prepare(
    `UPDATE pair_queue SET served_to = ?1, served_at = datetime('now')
     WHERE id = (
       SELECT id FROM pair_queue
       WHERE done_at IS NULL AND (served_at IS NULL OR served_at < datetime('now', '-10 minutes'))
       ORDER BY id LIMIT 1)
     RETURNING id, left_id, right_id`,
  )
    .bind(who.name)
    .first<{ id: number; left_id: number; right_id: number }>();
  if (row) return { queue_id: row.id, left_id: row.left_id, right_id: row.right_id };

  // Queue empty (cold start, or the trainer hasn't refilled it): random pair.
  // Same group only: cewek vs cewek, cowok vs cowok.
  const a = await env.DB.prepare("SELECT id, gender FROM applicants ORDER BY RANDOM() LIMIT 1").first<{ id: number; gender: string | null }>();
  if (!a) return null;
  const b = await env.DB.prepare(
    "SELECT id FROM applicants WHERE id != ?1 AND gender IS ?2 ORDER BY RANDOM() LIMIT 1",
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
    ms?: number;
  };
  const { queue_id, left_id, right_id, winner_id } = body;
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
      "INSERT INTO picks (rater, left_id, right_id, winner_id, kind, ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    ).bind(who.name, left_id, right_id, winner_id, kind, ms),
  ];
  if (queue_id !== null) {
    stmts.push(env.DB.prepare("UPDATE pair_queue SET done_at = datetime('now') WHERE id = ?1").bind(queue_id));
  }
  await env.DB.batch(stmts);
  if (winner_id !== null) ctx.waitUntil(maybeRetrain(env));
  return json({ ok: true });
}

async function maybeRetrain(env: Env) {
  const row = await env.DB.prepare(
    `SELECT (SELECT COUNT(*) FROM picks WHERE winner_id IS NOT NULL) AS picks,
            (SELECT n_picks FROM model WHERE id = 1) AS trained`,
  ).first<{ picks: number; trained: number | null }>();
  if (row && row.trained !== null && row.picks - row.trained >= RETRAIN_EVERY) await retrain(env);
  else if (row && row.trained === 0 && row.picks >= 5) await retrain(env);
}

async function results(env: Env, who: string | null) {
  if (who) return raterResults(env, who);
  const model = await env.DB.prepare(
    "SELECT active, heldout_hills, heldout_line, n_test, split_by, n_picks, trained_at, init FROM model WHERE id = 1",
  ).first<Record<string, unknown>>();
  const picks = await env.DB.prepare(
    "SELECT COUNT(*) AS decided, (SELECT COUNT(*) FROM picks) AS total, COUNT(DISTINCT rater) AS raters FROM picks WHERE winner_id IS NOT NULL",
  ).first<{ decided: number; total: number; raters: number }>();
  const { results: faces } = await env.DB.prepare(
    `SELECT a.id, a.score, a.score_std, a.look, a.gender, a.gender_src,
       (SELECT COUNT(*) FROM picks p WHERE p.winner_id = a.id) AS wins,
       (SELECT COUNT(*) FROM picks p WHERE p.winner_id IS NOT NULL AND (p.left_id = a.id OR p.right_id = a.id)) AS games,
       (SELECT COUNT(*) FROM photos ph WHERE ph.applicant_id = a.id) AS n_photos
     FROM applicants a ORDER BY a.score DESC NULLS LAST, a.id`,
  ).all();
  const init = model?.init ? JSON.parse(model.init as string) : null;
  if (model) delete model.init;
  return json({
    model: {
      ...model, embedding: init?.model ?? null, hills: init?.peaks?.length ?? 0, peak_gender: init?.peak_gender ?? [],
      retrain_every: RETRAIN_EVERY,
    },
    picks,
    faces,
  });
}

// The same page, scored by one rater's personal model.
async function raterResults(env: Env, who: string) {
  const m = await raterModel(env, who);
  if (!m) return json({ error: "no taste space loaded yet" }, 404);
  const { data, hills, n } = m;
  const counts = await env.DB.prepare(
    "SELECT COUNT(winner_id) AS decided, COUNT(*) AS total FROM picks WHERE rater = ?1",
  ).bind(who).first<{ decided: number; total: number }>();
  const photoCount = new Map(
    (await env.DB.prepare("SELECT applicant_id AS id, COUNT(*) AS n FROM photos GROUP BY applicant_id").all<{ id: number; n: number }>())
      .results.map((r) => [r.id, r.n]),
  );
  const wins = new Map<number, number>(), games = new Map<number, number>();
  for (const p of data.picks) {
    const [w, l] = [data.faces[p.w].id, data.faces[p.l].id];
    wins.set(w, (wins.get(w) ?? 0) + 1);
    games.set(w, (games.get(w) ?? 0) + 1);
    games.set(l, (games.get(l) ?? 0) + 1);
  }
  const faces = data.faces
    .map((f, i) => ({
      id: f.id,
      score: hills ? hills.score(data.Z[i], data.G[i]) : null,
      score_std: null,
      look: hills ? hills.hill(data.Z[i], data.G[i]) : null,
      gender: data.G[i],
      gender_src: null,
      wins: wins.get(f.id) ?? 0,
      games: games.get(f.id) ?? 0,
      n_photos: photoCount.get(f.id) ?? 0,
    }))
    .sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity) || a.id - b.id);
  return json({
    rater: who,
    model: {
      active: hills ? "hills" : null, n_picks: hills ? n : 0, trained_at: hills ? new Date().toISOString() : null,
      hills: data.peaks.length, peak_gender: data.peakGroup, min_picks: MIN_PICKS, retrain_every: RETRAIN_EVERY,
    },
    picks: { decided: counts?.decided ?? 0, total: counts?.total ?? 0, raters: 1 },
    faces,
    params: hills ? JSON.stringify(hills) : null,
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
  ctx.waitUntil(retrain(env));
  return json({ ok: true });
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

    if (url.pathname === "/api/me") {
      const mine = await env.DB.prepare("SELECT COUNT(*) AS n FROM picks WHERE rater = ?1 AND winner_id IS NOT NULL").bind(who.name).first<{ n: number }>();
      return json({ name: who.name, picks: mine?.n ?? 0 });
    }
    if (url.pathname === "/api/pair" && req.method === "GET") {
      const pair = await nextPair(env, who);
      return pair ? json(await withPhotos(env, pair)) : json({ error: "no applicants loaded" }, 404);
    }
    if (url.pathname === "/api/pick" && req.method === "POST") return recordPick(req, env, who, ctx);
    if (url.pathname === "/api/results" && req.method === "GET") return results(env, only);
    if (url.pathname === "/api/map" && req.method === "GET") {
      const gender = url.searchParams.get("gender");
      if (!only) return json(await territory(env, undefined, gender));
      const r = (await (await raterResults(env, only)).json()) as {
        params: string | null; faces: { id: number; score: number | null; look: number | null }[];
      };
      if (!r.faces) return json({ faces: [], peaks: [], grid: null });
      return json(await territory(env, { params: r.params, faces: new Map(r.faces.map((f) => [f.id, { score: f.score, hill: f.look }])) }, gender));
    }
    if (url.pathname === "/api/gender" && req.method === "POST") return setGender(req, env, ctx);
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
          `UPDATE model SET params = NULL, active = NULL, heldout_hills = NULL, heldout_line = NULL,
           n_test = NULL, split_by = NULL, n_picks = 0, trained_at = NULL WHERE id = 1`,
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
} satisfies ExportedHandler<Env>;
