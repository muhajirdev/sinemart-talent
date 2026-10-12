// Retrain the hills model from every pick in D1, rescore everyone, refill
// the pair queue. Runs automatically every RETRAIN_EVERY picks and on demand.

import { Arrow, Hills, Pick, agreement, nextPairs, rng, split } from "./hills";

export const RETRAIN_EVERY = 10;
const MIN_PICKS = 5;

type Env = { DB: D1Database };

async function runBatches(db: D1Database, stmts: D1PreparedStatement[], size = 200) {
  for (let i = 0; i < stmts.length; i += size) await db.batch(stmts.slice(i, i + size));
}

export async function retrain(env: Env) {
  const model = await env.DB.prepare("SELECT init FROM model WHERE id = 1").first<{ init: string }>();
  if (!model?.init) return { status: "no taste space loaded yet" };
  const peaks: number[][] = JSON.parse(model.init).peaks;

  const faces = (
    await env.DB.prepare("SELECT id, z FROM applicants WHERE z IS NOT NULL ORDER BY id").all<{ id: number; z: string }>()
  ).results;
  const index = new Map(faces.map((f, i) => [f.id, i]));
  const Z = faces.map((f) => JSON.parse(f.z) as number[]);

  const rows = (
    await env.DB.prepare(
      `SELECT winner_id, CASE WHEN winner_id = left_id THEN right_id ELSE left_id END AS loser_id
       FROM picks WHERE winner_id IS NOT NULL ORDER BY id`,
    ).all<{ winner_id: number; loser_id: number }>()
  ).results;
  const picks: Pick[] = rows
    .filter((r) => index.has(r.winner_id) && index.has(r.loser_id))
    .map((r) => ({ w: index.get(r.winner_id)!, l: index.get(r.loser_id)! }));
  if (picks.length < MIN_PICKS) return { status: `waiting for ${MIN_PICKS} picks`, picks: picks.length };

  // Held-out check: hills vs the straight arrow on picks they didn't see.
  const rand = rng(picks.length * 7919);
  const { by, train, test } = split(picks, faces.length, rand);
  let heldoutHills: number | null = null, heldoutLine: number | null = null;
  if (train.length >= MIN_PICKS && test.length >= 3) {
    const h = new Hills(peaks).fit(Z, train);
    const a = new Arrow(Z[0].length).fit(Z, train);
    heldoutHills = agreement(Z.map((z) => h.score(z)), test);
    heldoutLine = agreement(Z.map((z) => a.score(z)), test);
  }

  // The ranker: hills on every pick. Std = spread of 3 bootstrap fits.
  const main = new Hills(peaks).fit(Z, picks);
  const scores = Z.map((z) => main.score(z));
  const hill = Z.map((z) => main.hill(z));
  const runs = [0, 1, 2].map(() => {
    const boot = picks.map(() => picks[Math.floor(rand() * picks.length)]);
    const s = Z.map((z) => new Hills(peaks).fit(Z, boot, 150).score(z));
    const mean = s.reduce((x, y) => x + y, 0) / s.length;
    return s.map((x) => x - mean);
  });
  const std = scores.map((_, i) => {
    const v = runs.map((r) => r[i]);
    const m = v.reduce((x, y) => x + y, 0) / v.length;
    return Math.sqrt(v.reduce((x, y) => x + (y - m) ** 2, 0) / v.length);
  });

  const now = new Date().toISOString();
  const nPairs = Math.min(500, Math.max(20, faces.length * 3));
  const pairs = nextPairs(scores, std, nPairs, rand);
  const stmts: D1PreparedStatement[] = faces.map((f, i) =>
    env.DB.prepare(
      "UPDATE applicants SET score = ?1, score_std = ?2, look = ?3, model = 'hills', updated_at = ?4 WHERE id = ?5",
    ).bind(scores[i], std[i], hill[i], now, f.id),
  );
  stmts.push(env.DB.prepare("DELETE FROM pair_queue WHERE served_at IS NULL"));
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
       split_by = ?5, n_picks = ?6, trained_at = ?7 WHERE id = 1`,
    ).bind(JSON.stringify(main), heldoutHills, heldoutLine, test.length, by, picks.length, now),
  );
  await runBatches(env.DB, stmts);
  return { status: "trained", picks: picks.length, heldout_hills: heldoutHills, heldout_line: heldoutLine, n_test: test.length };
}
