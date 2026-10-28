import { useEffect, useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import type { ScoreResult } from '#/lib/score'

export const Route = createFileRoute('/')({ component: Scorer })

// What the quality gate's reasons mean (sinemart.crop.assess).
const REASONS: Record<string, string> = {
  no_face: 'wajah tidak ketemu',
  low_face_score: 'wajah kurang jelas',
  face_too_small: 'wajah terlalu kecil',
  blurry: 'buram',
  bad_exposure: 'terlalu gelap / terang',
  not_frontal: 'wajah tidak menghadap depan',
  multiple_faces: 'ada beberapa wajah',
  unreadable: 'file tidak bisa dibuka',
}
const GROUP = { f: 'cewek', m: 'cowok' } as const

function Scorer() {
  const [token, setToken] = useState<string | null>(null)
  const [files, setFiles] = useState<File[]>([])
  const [gender, setGender] = useState<'auto' | 'f' | 'm'>('auto')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<ScoreResult | null>(null)
  // Until React is running, a tap would submit the form the plain-HTML way
  // (a page reload): keep the buttons off until then.
  const [ready, setReady] = useState(false)

  useEffect(() => {
    setReady(true)
    try {
      setToken(localStorage.getItem('scorer-token'))
    } catch {}
  }, [])

  async function login(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    // Read the field itself: a name typed before the page finished loading
    // never reached React's state.
    const name = String(new FormData(e.currentTarget as HTMLFormElement).get('name') ?? '')
    const r = await fetch('/api/login', { method: 'POST', body: JSON.stringify({ name }) })
    const d = (await r.json()) as { token?: string; error?: string }
    if (!d.token) return setError(d.error ?? 'Gagal masuk.')
    try {
      localStorage.setItem('scorer-token', d.token)
    } catch {}
    setToken(d.token)
  }

  async function score(e: React.FormEvent) {
    e.preventDefault()
    if (!files.length || !token) return
    setBusy(true)
    setError('')
    setResult(null)
    const form = new FormData()
    for (const f of files) form.append('photo', f)
    form.append('gender', gender)
    try {
      const r = await fetch('/api/score', { method: 'POST', body: form, headers: { 'x-rater-token': token } })
      const d = (await r.json()) as ScoreResult & { error?: string }
      if (r.status === 401) {
        setToken(null)
        try { localStorage.removeItem('scorer-token') } catch {}
      }
      if (!r.ok) setError(d.error ?? 'Gagal menilai foto.')
      else setResult(d)
    } catch {
      setError('Koneksi putus. Coba lagi.')
    } finally {
      setBusy(false)
    }
  }

  if (!token) {
    return (
      <main>
        <h1>Nilai Foto</h1>
        <p className="note">Masuk dengan nama yang sama seperti di halaman pilih.</p>
        <form key="login" className="card" onSubmit={login}>
          <label htmlFor="name">Nama</label>
          <input id="name" name="name" type="text" autoComplete="name" />
          <button type="submit" disabled={!ready}>Masuk</button>
          {error && <p className="err">{error}</p>}
        </form>
      </main>
    )
  }

  return (
    <main>
      <h1>Nilai Foto</h1>
      <p className="note">
        Lihat skor seseorang yang belum ada di app, dengan model yang sama seperti di Lab. Foto tidak disimpan.
      </p>
      <form key="score" className="card" onSubmit={score}>
        <label htmlFor="photos">Foto (1–6 foto orang yang sama; wajah menghadap depan paling bagus)</label>
        <input
          id="photos" type="file" accept="image/*" multiple
          onChange={(e) => setFiles([...(e.target.files ?? [])].slice(0, 6))}
        />
        <label htmlFor="gender">Dibandingkan dengan</label>
        <select id="gender" value={gender} onChange={(e) => setGender(e.target.value as 'auto' | 'f' | 'm')}>
          <option value="auto">Otomatis</option>
          <option value="f">Cewek</option>
          <option value="m">Cowok</option>
        </select>
        <button type="submit" disabled={!ready || busy || !files.length}>{busy ? 'Menilai…' : 'Nilai'}</button>
        {busy && <p className="note">Kalau sudah lama tidak dipakai, foto pertama bisa makan waktu sampai 1 menit.</p>}
        {error && <p className="err">{error}</p>}
      </form>
      {result && <Result r={result} />}
      <p className="note">
        <button className="link" onClick={() => { try { localStorage.removeItem('scorer-token') } catch {} setToken(null) }}>
          Keluar
        </button>
      </p>
    </main>
  )
}

function Result({ r }: { r: ScoreResult }) {
  return (
    <section className="card">
      <div className="faces">
        {r.photos.map((p, i) => (
          <div key={i} className={`face ${p.ok ? 'ok' : 'bad'}`}>
            {p.crop ? <img src={`data:image/jpeg;base64,${p.crop}`} alt={`Wajah ${i + 1}`} /> : <div style={{ height: 112 }} />}
            {p.ok ? 'dipakai' : p.reasons.map((x) => REASONS[x] ?? x).join(', ')}
          </div>
        ))}
      </div>
      {!r.models.length ? (
        <p className="err">Tidak ada wajah yang bisa dinilai. Coba foto lain yang wajahnya jelas dan menghadap depan.</p>
      ) : (
        <>
          <p className="note">
            Dinilai dari {r.used} foto, dibandingkan dengan pendaftar {GROUP[r.gender]}
            {r.genderGuess && r.genderGuess !== r.gender ? ` (tebakan otomatis: ${GROUP[r.genderGuess]})` : ''}.
            {r.lowQuality ? ' Fotonya kurang bagus, jadi skornya kurang bisa dipercaya.' : ''}
          </p>
          <div className="models">
            {r.models.map((m) => (
              <div key={m.key} className="card" style={{ margin: 0, background: '#333' }}>
                <div className="note">{m.label} · garis lurus{m.ranker ? ' · peringkat utama' : ''}</div>
                <div className="score">{m.score}</div>
                <div className="bar"><i style={{ width: `${m.score}%` }} /></div>
                <div>Peringkat #{m.rank} dari {m.of} {GROUP[r.gender]}</div>
                <div className="note">Lebih tinggi dari {m.percentile}% pendaftar</div>
              </div>
            ))}
          </div>
          <p className="note">
            Skor = seberapa sering (0–100) orang ini menang melawan pendaftar rata-rata, memakai bobot model terbaru.
          </p>
        </>
      )}
    </section>
  )
}
