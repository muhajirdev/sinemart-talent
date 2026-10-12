// Sinemart A/B picking app: serves pairs of applicant crops, records picks,
// retrains the hills model as picks come in, and reports scores.
// Static UI lives in ../public; everything under /api and /img lands here.

import { territory } from "./map";
import { RETRAIN_EVERY, retrain } from "./retrain";

export interface Env {
  DB: D1Database;
  IMAGES: R2Bucket;
  ADMIN_TOKEN: string;
}

type Rater = { token: string; name: string };

const json = (data: unknown, status = 200) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });

async function rater(req: Request, env: Env): Promise<Rater | null> {
  const url = new URL(req.url);
  const token = req.headers.get("x-rater-token") ?? url.searchParams.get("t");
  if (!token) return null;
  return env.DB.prepare("SELECT token, name FROM raters WHERE token = ?1").bind(token).first<Rater>();
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
  const { results } = await env.DB.prepare("SELECT id FROM applicants ORDER BY RANDOM() LIMIT 2").all<{ id: number }>();
  if (results.length < 2) return null;
  return { queue_id: null, left_id: results[0].id, right_id: results[1].id };
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
    if (!q || q.left_id !== left_id || q.right_id !== right_id) return json({ error: "unknown pair" }, 400);
    kind = q.kind;
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

async function results(env: Env) {
  const model = await env.DB.prepare(
    "SELECT active, heldout_hills, heldout_line, n_test, split_by, n_picks, trained_at, init FROM model WHERE id = 1",
  ).first<Record<string, unknown>>();
  const picks = await env.DB.prepare(
    "SELECT COUNT(*) AS decided, (SELECT COUNT(*) FROM picks) AS total, COUNT(DISTINCT rater) AS raters FROM picks WHERE winner_id IS NOT NULL",
  ).first<{ decided: number; total: number; raters: number }>();
  const { results: faces } = await env.DB.prepare(
    `SELECT a.id, a.score, a.score_std, a.look,
       (SELECT COUNT(*) FROM picks p WHERE p.winner_id = a.id) AS wins,
       (SELECT COUNT(*) FROM picks p WHERE p.winner_id IS NOT NULL AND (p.left_id = a.id OR p.right_id = a.id)) AS games
     FROM applicants a ORDER BY a.score DESC NULLS LAST, a.id`,
  ).all();
  const init = model?.init ? JSON.parse(model.init as string) : null;
  if (model) delete model.init;
  return json({
    model: { ...model, embedding: init?.model ?? null, hills: init?.peaks?.length ?? 0, retrain_every: RETRAIN_EVERY },
    picks,
    faces,
  });
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

    const who = await rater(req, env);
    if (!who) return json({ error: "ask for your rater link" }, 401);

    if (url.pathname === "/api/me") {
      const mine = await env.DB.prepare("SELECT COUNT(*) AS n FROM picks WHERE rater = ?1 AND winner_id IS NOT NULL").bind(who.name).first<{ n: number }>();
      return json({ name: who.name, picks: mine?.n ?? 0 });
    }
    if (url.pathname === "/api/pair" && req.method === "GET") {
      const pair = await nextPair(env, who);
      return pair ? json(pair) : json({ error: "no applicants loaded" }, 404);
    }
    if (url.pathname === "/api/pick" && req.method === "POST") return recordPick(req, env, who, ctx);
    if (url.pathname === "/api/results" && req.method === "GET") return results(env);
    if (url.pathname === "/api/map" && req.method === "GET") return json(await territory(env));
    if (url.pathname === "/api/retrain" && req.method === "POST") return json(await retrain(env));
    if (url.pathname === "/api/reset" && req.method === "POST") {
      // Start testing fresh: forget every pick and everything learned from them.
      await env.DB.batch([
        env.DB.prepare("DELETE FROM picks"),
        env.DB.prepare("DELETE FROM pair_queue"),
        env.DB.prepare("UPDATE applicants SET score = NULL, score_std = NULL, model = NULL, updated_at = NULL"),
        env.DB.prepare(
          `UPDATE model SET params = NULL, active = NULL, heldout_hills = NULL, heldout_line = NULL,
           n_test = NULL, split_by = NULL, n_picks = 0, trained_at = NULL WHERE id = 1`,
        ),
      ]);
      return json({ ok: true });
    }

    const img = url.pathname.match(/^\/img\/(\d+)$/);
    if (img) {
      const obj = await env.IMAGES.get(`crops/${img[1]}.jpg`);
      if (!obj) return new Response("not found", { status: 404 });
      return new Response(obj.body, {
        headers: { "content-type": "image/jpeg", "cache-control": "private, max-age=86400" },
      });
    }
    return json({ error: "not found" }, 404);
  },
} satisfies ExportedHandler<Env>;
