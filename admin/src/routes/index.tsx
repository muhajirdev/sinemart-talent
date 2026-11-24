import { useCallback, useEffect, useMemo, useState } from 'react'
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { GENDER, formRow, reasonName, sourceName, toCsv, unreadableCsv, type Casting, type Face, type Unreadable } from '#/lib/data'

// 2026-11-24 (Muajir): "allow the filtering or threshold ... so the admin can
// kinda choose to filter out at what threshold", "contrast effect, at a glance
// we can see top and bad", "default to cowok". The view lives in the URL, so a
// link opens the same source, gender and cutoff.
type Search = { source?: string; g: 'm' | 'f' | 'all'; cut: number; tab: 'lolos' | 'tersaring' | 'semua' | 'tidak-terbaca' }

export const Route = createFileRoute('/')({
  validateSearch: (s: Record<string, unknown>): Search => ({
    source: s.source == null || s.source === '' ? undefined : String(s.source), // '2026' arrives as a number
    g: s.g === 'f' || s.g === 'all' ? s.g : 'm',
    cut: Math.max(0, Math.min(100, Number(s.cut) || 0)),
    tab: s.tab === 'tersaring' || s.tab === 'semua' || s.tab === 'tidak-terbaca' ? s.tab : 'lolos',
  }),
  component: Page,
})

const TOKEN_KEY = 'casting-token'

function Page() {
  const [token, setToken] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  useEffect(() => {
    try { setToken(localStorage.getItem(TOKEN_KEY)) } catch {}
    setReady(true)
  }, [])
  if (!ready) return <div className="center muted">Memuat…</div>
  if (!token) return <Login onToken={(t) => { try { localStorage.setItem(TOKEN_KEY, t) } catch {} setToken(t) }} />
  return <Admin token={token} onLogout={() => { try { localStorage.removeItem(TOKEN_KEY) } catch {} setToken(null) }} />
}

function Login({ onToken }: { onToken: (t: string) => void }) {
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault()
    setBusy(true); setError('')
    const name = String(new FormData(e.currentTarget).get('name') ?? '')
    try {
      const r = await fetch('/api/login', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name }) })
      const d = (await r.json()) as { token?: string; error?: string }
      if (d.token) onToken(d.token)
      else setError('Ketik nama kamu dulu.')
    } catch { setError('Koneksi putus. Coba lagi.') }
    setBusy(false)
  }
  return (
    <div className="center">
      <form className="login" onSubmit={submit}>
        <div className="brand">Sinemart <span>Casting</span></div>
        <p className="muted">Masuk dengan nama kamu untuk menyaring pelamar.</p>
        <input name="name" placeholder="Nama" autoComplete="name" autoFocus />
        <button className="primary" disabled={busy}>{busy ? 'Masuk…' : 'Masuk'}</button>
        {error && <p className="err">{error}</p>}
      </form>
    </div>
  )
}

function Admin({ token, onLogout }: { token: string; onLogout: () => void }) {
  const search = Route.useSearch()
  const navigate = useNavigate({ from: '/' })
  const set = useCallback((s: Partial<Search>) => navigate({ search: (p) => ({ ...p, ...s }), replace: true }), [navigate])
  const [data, setData] = useState<Casting | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [open, setOpen] = useState<Face | null>(null)
  const [cut, setCut] = useState(search.cut) // follows the slider at once; the URL catches up
  useEffect(() => setCut(search.cut), [search.cut])
  useEffect(() => { const t = setTimeout(() => cut !== search.cut && set({ cut }), 250); return () => clearTimeout(t) }, [cut, search.cut, set])

  useEffect(() => {
    let live = true
    setLoading(true); setError('')
    fetch('/api/casting' + (search.source ? `?source=${encodeURIComponent(search.source)}` : ''), { headers: { 'x-rater-token': token } })
      .then(async (r) => {
        if (r.status === 401) return onLogout()
        const d = (await r.json()) as Casting
        if (!live) return
        // No source picked yet: open on the newest batch.
        if (!search.source && d.sources.length) return set({ source: d.sources[0].source })
        setData(d)
      })
      .catch(() => live && setError('Gagal memuat. Coba muat ulang halaman.'))
      .finally(() => live && setLoading(false))
    return () => { live = false }
  }, [search.source, token]) // eslint-disable-line react-hooks/exhaustive-deps

  const img = (id: number, n?: number) => `/img/${id}${n == null ? '' : `/${n}`}?t=${encodeURIComponent(token)}`
  const faces = useMemo(() => (data?.faces ?? []).filter((f) => search.g === 'all' || f.gender === search.g), [data, search.g])
  const scored = useMemo(() => faces.filter((f) => f.skor != null), [faces])
  const kept = useMemo(() => scored.filter((f) => f.skor! >= cut), [scored, cut])
  const dropped = useMemo(() => scored.filter((f) => f.skor! < cut), [scored, cut])
  const unread = useMemo(() => (data?.unreadable ?? []).filter((u) => search.g === 'all' || u.gender === search.g), [data, search.g])
  const shown = search.tab === 'lolos' ? kept : search.tab === 'tersaring' ? dropped : faces

  // Cutoff that filters the bottom p% of this view.
  const cutForBottom = (p: number) => {
    if (!scored.length) return 0
    const asc = [...scored].sort((a, b) => a.skor! - b.skor!)
    const i = Math.min(asc.length - 1, Math.floor((asc.length * p) / 100))
    return Math.ceil(asc[i].skor!)
  }

  const g = search.g === 'all' ? '' : `-${GENDER[search.g].toLowerCase()}`
  function save(csv: string, name: string) {
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    a.download = name
    a.click()
  }
  const download = () => save(toCsv(kept), `lolos-${search.source ?? 'semua'}${g}-batas${cut}.csv`)

  const n = Math.min(10, Math.floor(scored.length / 2))
  return (
    <div className="app">
      <header className="top">
        <div className="brand">Sinemart <span>Casting</span></div>
        <button className="ghost" onClick={onLogout}>Keluar</button>
      </header>

      <div className="layout">
        <aside className="side">
          <section className="panel">
            <label className="lab">Sumber pelamar</label>
            <select value={search.source ?? ''} onChange={(e) => set({ source: e.target.value || undefined, cut: 0 })}>
              {data?.sources.map((s) => <option key={s.source} value={s.source}>{sourceName(s.source)} · {s.n}</option>)}
            </select>
            <label className="lab">Gender</label>
            <div className="seg">
              {(['m', 'f', 'all'] as const).map((g) => (
                <button key={g} aria-pressed={search.g === g} onClick={() => set({ g })}>{g === 'all' ? 'Semua' : GENDER[g]}</button>
              ))}
            </div>
          </section>

          <section className="panel">
            <div className="cutrow">
              <div><label className="lab">Batas skor</label><div className="big">{cut}</div></div>
              <div className="pct">{scored.length ? Math.round((100 * dropped.length) / scored.length) : 0}%<span>tersaring</span></div>
            </div>
            <Histogram faces={scored} cut={cut} onCut={setCut} />
            <input type="range" min={0} max={100} value={cut} onChange={(e) => setCut(Number(e.target.value))} aria-label="Batas skor" />
            <div className="presets">
              <span className="muted">Saring terbawah</span>
              {[10, 25, 50].map((p) => <button key={p} className="chip" onClick={() => setCut(cutForBottom(p))}>{p}%</button>)}
              <button className="chip" onClick={() => setCut(0)}>Reset</button>
            </div>
            <div className="stats">
              <div><b className="good">{kept.length}</b><span>Lolos</span></div>
              <div><b className="bad">{dropped.length}</b><span>Tersaring</span></div>
              <div><b>{faces.length}</b><span>Total</span></div>
            </div>
            {faces.length > scored.length && <p className="muted small">{faces.length - scored.length} belum ada skor.</p>}
            <button className="primary wide" disabled={!kept.length} onClick={download}>Unduh {kept.length} yang lolos (CSV)</button>
          </section>
          <p className="muted small">
            Skor 0–100 = seberapa besar peluang orang ini dipilih tim dibanding pelamar rata-rata. Geser batas untuk menyaring yang jelas kurang cocok.
          </p>
        </aside>

        <main className="content">
          {error && <p className="err">{error}</p>}
          {loading && !data ? <div className="center muted">Memuat pelamar…</div> : (
            <>
              {n > 0 && (
                <section className="contrast">
                  <Strip title="Paling atas" tone="good" faces={scored.slice(0, n)} img={img} onOpen={setOpen} />
                  <Strip title="Paling bawah" tone="bad" faces={scored.slice(-n).reverse()} img={img} onOpen={setOpen} />
                </section>
              )}
              <nav className="tabs">
                {([['lolos', `Lolos · ${kept.length}`], ['tersaring', `Tersaring · ${dropped.length}`], ['semua', `Semua · ${faces.length}`], ...(unread.length ? [['tidak-terbaca', `Foto tidak terbaca · ${unread.length}`] as const] : [])] as const).map(([t, label]) => (
                  <button key={t} aria-pressed={search.tab === t} onClick={() => set({ tab: t })}>{label}</button>
                ))}
              </nav>
              {search.tab === 'tidak-terbaca' ? (
                <Unread list={unread} onDownload={() => save(unreadableCsv(unread), `foto-tidak-terbaca-${search.source ?? 'semua'}${g}.csv`)} />
              ) : (
              <div className={`grid${loading ? ' dim' : ''}`}>
                {shown.map((f) => (
                  <Tile key={f.id} f={f} img={img} out={f.skor == null || f.skor < cut} onOpen={setOpen} showOut={search.tab === 'semua'} />
                ))}
                {!shown.length && <p className="muted">Tidak ada pelamar di sini.</p>}
              </div>
              )}
            </>
          )}
        </main>
      </div>

      {open && <Viewer list={shown.includes(open) ? shown : scored} face={open} cut={cut} img={img} onClose={() => setOpen(null)} onMove={setOpen} />}
    </div>
  )
}

function Unread({ list, onDownload }: { list: Unreadable[]; onDownload: () => void }) {
  const counts = useMemo(() => {
    const m = new Map<string, number>()
    for (const u of list) m.set(u.reason, (m.get(u.reason) ?? 0) + 1)
    return [...m].sort((a, b) => b[1] - a[1])
  }, [list])
  return (
    <section>
      <div className="unread-head">
        <p className="muted">
          Foto mereka tidak bisa dinilai, jadi belum ada skor. Cek manual di form lewat nomor barisnya.
        </p>
        <button className="chip" onClick={onDownload}>Unduh daftar (CSV)</button>
      </div>
      <div className="reasons">{counts.map(([r, n]) => <span key={r} className="chip static">{reasonName(r)} · {n}</span>)}</div>
      <table className="list">
        <thead><tr><th>Baris form</th><th>No. pelamar</th><th>Gender</th><th>Alasan</th></tr></thead>
        <tbody>
          {list.map((u) => (
            <tr key={u.id}><td><b>{u.row ?? '–'}</b></td><td className="muted">#{u.id}</td><td>{u.gender ? GENDER[u.gender] : '–'}</td><td>{reasonName(u.reason)}</td></tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}

function Histogram({ faces, cut, onCut }: { faces: Face[]; cut: number; onCut: (c: number) => void }) {
  const bins = useMemo(() => {
    const b = Array.from({ length: 20 }, () => 0)
    for (const f of faces) b[Math.min(19, Math.floor(f.skor! / 5))]++
    return b
  }, [faces])
  const max = Math.max(1, ...bins)
  return (
    <div className="hist" aria-hidden>
      {bins.map((c, i) => (
        <button key={i} className={i * 5 + 5 <= cut ? 'out' : i * 5 < cut ? 'part' : ''} style={{ height: `${8 + (92 * c) / max}%` }}
          title={`Skor ${i * 5}–${i * 5 + 5}: ${c} orang`} onClick={() => onCut(i * 5)} />
      ))}
      <i style={{ left: `${cut}%` }} />
    </div>
  )
}

type ImgFn = (id: number, n?: number) => string

function Strip({ title, tone, faces, img, onOpen }: { title: string; tone: 'good' | 'bad'; faces: Face[]; img: ImgFn; onOpen: (f: Face) => void }) {
  return (
    <div className={`strip ${tone}`}>
      <h3>{title}</h3>
      <div className="strip-grid">
        {faces.map((f) => (
          <button key={f.id} className="mini" onClick={() => onOpen(f)}>
            <img src={img(f.id)} alt="" loading="lazy" />
            <span>{Math.round(f.skor!)}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function Tile({ f, img, out, showOut, onOpen }: { f: Face; img: ImgFn; out: boolean; showOut: boolean; onOpen: (f: Face) => void }) {
  return (
    <button className={`tile${showOut && out ? ' out' : ''}`} onClick={() => onOpen(f)}>
      <img src={img(f.id)} alt="" loading="lazy" />
      <div className="meta">
        <b className={out ? 'bad' : 'good'}>{f.skor == null ? '–' : Math.round(f.skor)}</b>
        <span>#{f.id}</span>
      </div>
    </button>
  )
}

function Viewer({ list, face, cut, img, onClose, onMove }: { list: Face[]; face: Face; cut: number; img: ImgFn; onClose: () => void; onMove: (f: Face) => void }) {
  const [n, setN] = useState(-1) // -1 = face crop, 0.. = whole photos
  useEffect(() => setN(-1), [face])
  const i = list.indexOf(face)
  const go = useCallback((d: number) => { const j = i + d; if (j >= 0 && j < list.length) onMove(list[j]) }, [i, list, onMove])
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); if (e.key === 'ArrowLeft') go(-1); if (e.key === 'ArrowRight') go(1) }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [go, onClose])
  const row = formRow(face)
  const out = face.skor == null || face.skor < cut
  return (
    <div className="viewer" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="sheet">
        <button className="x ghost" onClick={onClose} aria-label="Tutup">✕</button>
        <img className="photo" src={img(face.id, n < 0 ? undefined : n)} alt="" />
        {face.n_photos > 0 && (
          <div className="thumbs">
            {[-1, ...Array.from({ length: face.n_photos }, (_, k) => k)].map((k) => (
              <button key={k} aria-pressed={k === n} onClick={() => setN(k)}><img src={img(face.id, k < 0 ? undefined : k)} alt="" /></button>
            ))}
          </div>
        )}
        <div className="info">
          <div className={`score ${out ? 'bad' : 'good'}`}>{face.skor == null ? '–' : Math.round(face.skor)}</div>
          <div>
            <div className="tag">{out ? 'Tersaring' : 'Lolos'}</div>
            <div className="muted small">#{face.id}{row ? ` · baris ${row} di form` : ''} · {face.gender ? GENDER[face.gender] : '–'} · {sourceName(face.source)}</div>
          </div>
        </div>
        <div className="nav">
          <button className="chip" disabled={i <= 0} onClick={() => go(-1)}>‹ Sebelumnya</button>
          <span className="muted small">{i + 1} / {list.length}</span>
          <button className="chip" disabled={i < 0 || i >= list.length - 1} onClick={() => go(1)}>Berikutnya ›</button>
        </div>
      </div>
    </div>
  )
}
