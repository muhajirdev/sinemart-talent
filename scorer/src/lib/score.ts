// Score photos that aren't in the app, the way the Lab scores applicants.
//
//   photo ─► container: face crop + raw numbers per model (Python, same code as the pipeline)
//         ─► taste space from R2 models/<key>/taste.json → 16 numbers
//         ─► × the model's current straight-line weights in D1 → raw score
//         ─► against the applicants of that gender: 0–100, rank, percentile
//
// Read-only: nothing here writes to D1 or R2, and photos are never stored.
import { env } from 'cloudflare:workers'
import { getContainer } from '@cloudflare/containers'

// The Lab names comparison models "Model N" by key order (m2 = Model 2, ...),
// so these labels match what the team sees there. Never name the technology.
// Model 6's line ranks the leaderboard since 2026-10-27 (RANKER in
// worker/src/retrain.ts), so it comes first.
export const MODELS = [
  { key: 'm6', label: 'Model 6', ranker: true },
  { key: 'm1', label: 'Model 1', ranker: false },
] as const
type Key = (typeof MODELS)[number]['key']

type Taste = { dim: number; rank: number; mu: number[]; P: number[][] }
type Embedded = {
  face: boolean
  ok: boolean
  reasons: string[]
  crop?: string
  p_female?: number | null
  emb?: Record<string, number[]>
}

export type PhotoResult = { face: boolean; ok: boolean; reasons: string[]; crop: string | null }
export type ModelResult = {
  key: string; label: string; ranker: boolean; score: number; rank: number; of: number; percentile: number
}
export type ScoreResult = {
  gender: 'f' | 'm'
  genderGuess: 'f' | 'm' | null
  photos: PhotoResult[]
  used: number
  lowQuality: boolean
  models: ModelResult[]
}

export async function raterName(token: string | null) {
  if (!token) return null
  const r = await env.DB.prepare('SELECT name FROM raters WHERE token = ?1').bind(token).first<{ name: string }>()
  return r?.name ?? null
}

// Sign in with a name the team already uses on the picking page; the scorer
// never creates raters.
export async function knownRater(name: string) {
  const clean = name.replace(/\s+/g, ' ').trim().slice(0, 40)
  if (clean.length < 2) return null
  return env.DB.prepare('SELECT token, name FROM raters WHERE lower(name) = lower(?1)')
    .bind(clean)
    .first<{ token: string; name: string }>()
}

const tastes = new Map<string, Taste>()
async function taste(key: Key): Promise<Taste> {
  const cached = tastes.get(key)
  if (cached) return cached
  const obj = await env.IMAGES.get(`models/${key}/taste.json`)
  if (!obj) throw new Error(`no taste space for ${key}`)
  const t = (await obj.json()) as Taste
  tastes.set(key, t)
  return t
}

const project = (t: Taste, e: number[]) => t.P.map((row) => row.reduce((s, p, i) => s + p * (e[i] - t.mu[i]), 0))
const dot = (w: number[], z: number[]) => w.reduce((s, x, i) => s + x * z[i], 0)

// The model's current straight line and everyone's taste-space numbers, as
// the retrain left them (test batches and bad photos left out).
async function liveModel(key: Key, gender: 'f' | 'm') {
  const row =
    key === 'm1'
      ? await env.DB.prepare('SELECT params_line FROM model WHERE id = 1').first<{ params_line: string | null }>()
      : await env.DB.prepare('SELECT params_line FROM alt_models WHERE key = ?1').bind(key).first<{ params_line: string | null }>()
  if (!row?.params_line) throw new Error(`${key} is not trained yet`)
  const w = (JSON.parse(row.params_line) as { w: number[] }).w
  const { results } =
    key === 'm1'
      ? await env.DB.prepare(
          'SELECT z FROM applicants WHERE z IS NOT NULL AND noise IS NULL AND test = 0 AND gender = ?1',
        ).bind(gender).all<{ z: string }>()
      : await env.DB.prepare(
          `SELECT z.z FROM alt_z z JOIN applicants a ON a.id = z.applicant_id
           WHERE z.key = ?1 AND a.noise IS NULL AND a.test = 0 AND a.gender = ?2`,
        ).bind(key, gender).all<{ z: string }>()
  return { w, scores: results.map((r) => dot(w, JSON.parse(r.z) as number[])) }
}

// One person = the average of their usable face crops, re-normalised
// (sinemart.embed.per_person).
function average(vectors: number[][]) {
  const sum = vectors[0].map((_, i) => vectors.reduce((s, v) => s + v[i], 0))
  const norm = Math.hypot(...sum)
  return sum.map((x) => x / norm)
}

// The container wakes with the models still loading: wait for it.
async function embed(photo: ArrayBuffer): Promise<Embedded> {
  const box = getContainer(env.SCORER, 'main')
  for (let attempt = 0; attempt < 90; attempt++) {
    const r = await box.fetch('http://container/embed', { method: 'POST', body: photo })
    if (r.status !== 503) {
      if (!r.ok) throw new Error(`scoring failed (${r.status})`)
      return (await r.json()) as Embedded
    }
    await new Promise((ok) => setTimeout(ok, 2000))
  }
  throw new Error('the scorer is still starting; try again in a minute')
}

export async function scorePhotos(photos: ArrayBuffer[], gender: 'f' | 'm' | 'auto'): Promise<ScoreResult> {
  const out: Embedded[] = []
  for (const p of photos) out.push(await embed(p)) // one at a time: the container scores one photo at a time
  const faces = out.filter((o) => o.face && o.emb)
  if (!faces.length) {
    return {
      gender: gender === 'auto' ? 'f' : gender, genderGuess: null, used: 0, lowQuality: false, models: [],
      photos: out.map((o) => ({ face: o.face, ok: o.ok, reasons: o.reasons, crop: o.crop ?? null })),
    }
  }
  // Like the pipeline, only crops that pass the quality gate count; if none
  // pass, score what there is and say so.
  const good = faces.filter((o) => o.ok)
  const use = good.length ? good : faces
  const pf = use.map((o) => o.p_female).filter((x): x is number => x != null)
  const genderGuess = pf.length ? (pf.reduce((a, b) => a + b, 0) / pf.length >= 0.5 ? 'f' : 'm') : null
  const g = gender === 'auto' ? (genderGuess ?? 'f') : gender
  const models: ModelResult[] = []
  for (const m of MODELS) {
    const t = await taste(m.key)
    const { w, scores } = await liveModel(m.key, g)
    const raw = dot(w, project(t, average(use.map((o) => o.emb![m.key]))))
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length
    const above = scores.filter((s) => s > raw).length
    models.push({
      key: m.key, label: m.label, ranker: m.ranker,
      score: Math.round(100 / (1 + Math.exp(-(raw - mean)))), // as the Lab: vs an average applicant
      rank: above + 1, of: scores.length,
      percentile: Math.round((scores.filter((s) => s < raw).length / scores.length) * 100),
    })
  }
  return {
    gender: g, genderGuess, used: use.length, lowQuality: !good.length, models,
    photos: out.map((o) => ({ face: o.face, ok: o.ok, reasons: o.reasons, crop: o.crop ?? null })),
  }
}
