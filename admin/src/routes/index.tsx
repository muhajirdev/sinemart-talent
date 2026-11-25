import { useCallback, useEffect, useMemo, useState } from 'react'
import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { EXPORT_HEAD, GENDER, PRESET_TAGS, exportCells, exportCsv, formRow, reasonName, sourceName, toCsv, unreadableCsv, type Casting, type ExportRow, type Face, type Unreadable } from '#/lib/data'

// 2026-11-24 (Muajir): "allow the filtering or threshold ... so the admin can
// kinda choose to filter out at what threshold", "contrast effect, at a glance
// we can see top and bad", "default to cowok". The view lives in the URL, so a
// link opens the same source, gender and cutoff.
// 2026-11-24 (Muajir: "so we can change and choose the right taste"): taste =
// score everyone by one rater's own picks instead of the whole team's.
type Search = { source?: string; taste?: string; tag?: string; g: 'm' | 'f' | 'all'; cut: number; tab: 'lolos' | 'tersaring' | 'semua' | 'tidak-terbaca' }

export const Route = createFileRoute('/')({
  validateSearch: (s: Record<string, unknown>): Search => ({
    source: s.source == null || s.source === '' ? undefined : String(s.source), // '2026' arrives as a number
    taste: s.taste == null || s.taste === '' ? undefined : String(s.taste),
    tag: s.tag == null || s.tag === '' ? undefined : String(s.tag),
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
  // 2026-11-24 (Muajir: "photos taste builder"): tap faces as Suka / Tidak, name it, save as a taste.
  const [build, setBuild] = useState(false)
  const [marks, setMarks] = useState<Map<number, 'suka' | 'tidak'>>(new Map())
  const [reload, setReload] = useState(0)
  const [tags, setTags] = useState<Map<number, string[]>>(new Map())
  const [cut, setCut] = useState(search.cut) // follows the slider at once; the URL catches up
  useEffect(() => setCut(search.cut), [search.cut])
  useEffect(() => { const t = setTimeout(() => cut !== search.cut && set({ cut }), 250); return () => clearTimeout(t) }, [cut, search.cut, set])

  useEffect(() => {
    let live = true
    setLoading(true); setError('')
    const q = new URLSearchParams()
    if (search.source) q.set('source', search.source)
    if (search.taste) q.set('taste', search.taste)
    fetch('/api/casting' + (q.size ? `?${q}` : ''), { headers: { 'x-rater-token': token } })
      .then(async (r) => {
        if (r.status === 401) return onLogout()
        const d = (await r.json()) as Casting
        if (!live) return
        // No source picked yet: open on the newest batch.
        if (!search.source && d.sources.length) return set({ source: d.sources[0].source })
        setData(d)
        setTags(new Map(Object.entries(d.tags ?? {}).map(([k, v]) => [Number(k), v])))
      })
      .catch(() => live && setError('Gagal memuat. Coba muat ulang halaman.'))
      .finally(() => live && setLoading(false))
    return () => { live = false }
  }, [search.source, search.taste, token, reload]) // eslint-disable-line react-hooks/exhaustive-deps

  const img = (id: number, n?: number) => `/img/${id}${n == null ? '' : `/${n}`}?t=${encodeURIComponent(token)}`
  const faces = useMemo(() => (data?.faces ?? []).filter((f) => (search.g === 'all' || f.gender === search.g)
    && (!search.tag || (search.tag === '-' ? !tags.get(f.id)?.length : tags.get(f.id)?.includes(search.tag)))), [data, search.g, search.tag, tags])
  // Every tag in use (with counts in this gender), presets first.
  const tagCounts = useMemo(() => {
    const c = new Map<string, number>(PRESET_TAGS.map((t) => [t, 0]))
    for (const f of data?.faces ?? []) if (search.g === 'all' || f.gender === search.g) for (const t of tags.get(f.id) ?? []) c.set(t, (c.get(t) ?? 0) + 1)
    return [...c]
  }, [data, search.g, tags])
  async function toggleTag(f: Face, tag: string) {
    const on = !(tags.get(f.id) ?? []).includes(tag)
    setTags((m) => { const n = new Map(m), cur = n.get(f.id) ?? []; n.set(f.id, on ? [...cur, tag] : cur.filter((t) => t !== tag)); return n })
    const r = await post('/api/casting/tag', { id: f.id, tag, on })
    if (!r.ok) setReload((x) => x + 1) // didn't save: show what the server has
  }
  const scored = useMemo(() => faces.filter((f) => f.skor != null), [faces])
  const kept = useMemo(() => scored.filter((f) => f.skor! >= cut), [scored, cut])
  const dropped = useMemo(() => scored.filter((f) => f.skor! < cut), [scored, cut])
  const unread = useMemo(() => (data?.unreadable ?? []).filter((u) => search.g === 'all' || u.gender === search.g), [data, search.g])
  const shown = build ? faces : search.tab === 'lolos' ? kept : search.tab === 'tersaring' ? dropped : faces
  const current = data?.tastes?.find((t) => t.key === data.taste)
  const mark = (f: Face) => setMarks((m) => {
    const n = new Map(m), v = m.get(f.id)
    if (!v) n.set(f.id, 'suka'); else if (v === 'suka') n.set(f.id, 'tidak'); else n.delete(f.id)
    return n
  })
  async function post(path: string, body: unknown) {
    const r = await fetch(path, { method: 'POST', headers: { 'content-type': 'application/json', 'x-rater-token': token }, body: JSON.stringify(body) })
    return (await r.json()) as { key?: string; error?: string; ok?: boolean }
  }
  async function saveTaste(name: string) {
    const ids = (k: 'suka' | 'tidak') => [...marks].filter(([, v]) => v === k).map(([id]) => id)
    const d = await post('/api/casting/taste', { name, suka: ids('suka'), tidak: ids('tidak') })
    if (d.error) return d.error
    setBuild(false); setMarks(new Map()); set({ taste: d.key, cut: 0 }); setReload((x) => x + 1)
    return ''
  }
  async function deleteTaste() {
    if (!current || current.kind !== 'foto') return
    await post('/api/casting/taste/delete', { key: current.key })
    set({ taste: undefined }); setReload((x) => x + 1)
  }

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
  const base = `${search.source ?? 'semua'}${g}${search.tag ? `-${search.tag.toLowerCase().replace(/[^a-z0-9]+/g, '-')}` : ''}`
  const rows = (): ExportRow[] => faces.map((f) => ({ f, status: f.skor == null ? 'Belum ada skor' : f.skor >= cut ? 'Lolos' : 'Tersaring', tags: tags.get(f.id) ?? [] }))
  async function downloadExcel() {
    const { default: writeXlsxFile } = await import('write-excel-file/browser')
    const bold = (v: string) => ({ value: v, fontWeight: 'bold' as const })
    const people = [EXPORT_HEAD.map(bold), ...rows().map(exportCells)]
    const sheets = [{ data: people, sheet: 'Pelamar', stickyRowsCount: 1, columns: [14, 11, 9, 26, 7, 14, 30].map((width) => ({ width })) }]
    if (unread.length) sheets.push({ data: [['No. pelamar', 'Baris form', 'Gender', 'Alasan'].map(bold), ...unread.map((u) => [u.id, u.row, u.gender ? GENDER[u.gender] : null, reasonName(u.reason)])], sheet: 'Foto tidak terbaca', stickyRowsCount: 1, columns: [14, 11, 9, 28].map((width) => ({ width })) })
    await writeXlsxFile(sheets as never).toFile(`pelamar-${base}-batas${cut}.xlsx`)
  }
  const downloadCsv = () => save(exportCsv(rows()), `pelamar-${base}-batas${cut}.csv`)
  const download = () => save(toCsv(kept), `lolos-${search.source ?? 'semua'}${g}${data?.taste ? `-taste-${data.taste.toLowerCase().replace(/[^a-z0-9]+/g, '-')}` : ''}-batas${cut}.csv`)

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
            <label className="lab">Taste</label>
            <select value={search.taste ?? ''} onChange={(e) => set({ taste: e.target.value || undefined })}>
              <option value="">Tim (semua penilai)</option>
              {!!data?.tastes?.some((t) => t.kind === 'foto') && (
                <optgroup label="Dari contoh foto">
                  {data.tastes.filter((t) => t.kind === 'foto').map((t) => <option key={t.key} value={t.key}>{t.name} · {t.n} foto</option>)}
                </optgroup>
              )}
              <optgroup label="Taste penilai">
                {data?.tastes?.filter((t) => t.kind === 'rater').map((t) => <option key={t.key} value={t.key}>{t.name} · {t.n} pilihan</option>)}
              </optgroup>
            </select>
            <div className="presets taste-actions">
              {!build && <button className="chip" onClick={() => { setBuild(true); setMarks(new Map()) }}>+ Buat taste dari foto</button>}
              {current?.kind === 'foto' && !build && <button className="chip" onClick={deleteTaste}>Hapus taste ini</button>}
            </div>
            {search.taste && data && data.taste !== search.taste && !loading && <p className="muted small">Taste ini belum cukup pilihan, jadi pakai taste tim.</p>}
            <label className="lab">Gender</label>
            <div className="seg">
              {(['m', 'f', 'all'] as const).map((g) => (
                <button key={g} aria-pressed={search.g === g} onClick={() => set({ g })}>{g === 'all' ? 'Semua' : GENDER[g]}</button>
              ))}
            </div>
            <label className="lab">Tag</label>
            <select value={search.tag ?? ''} onChange={(e) => set({ tag: e.target.value || undefined })}>
              <option value="">Semua orang</option>
              {tagCounts.map(([t, c]) => <option key={t} value={t}>{t} · {c}</option>)}
              <option value="-">Belum ditandai</option>
            </select>
          </section>

          {build && <Builder marks={marks} onSave={saveTaste} onCancel={() => { setBuild(false); setMarks(new Map()) }} />}
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
            <button className="primary wide" disabled={!faces.length} onClick={downloadExcel}>Unduh Excel ({faces.length} orang + status)</button>
            <div className="presets dl">
              <span className="muted">Atau</span>
              <button className="chip" disabled={!faces.length} onClick={downloadCsv}>CSV semua</button>
              <button className="chip" disabled={!kept.length} onClick={download}>CSV {kept.length} yang lolos</button>
            </div>
          </section>
          <p className="muted small">
            Skor 0–100 = seberapa besar peluang orang ini dipilih {data?.taste ? data.taste : 'tim'} dibanding pelamar rata-rata. Geser batas untuk menyaring yang jelas kurang cocok.
          </p>
        </aside>

        <main className="content">
          {error && <p className="err">{error}</p>}
          {loading && !data ? <div className="center muted">Memuat pelamar…</div> : (
            <>
              {build && <p className="build-hint">Ketuk wajah: sekali = <b className="good">Suka</b>, dua kali = <b className="bad">Tidak</b>, tiga kali = batal. Pilih yang jelas cocok dan jelas tidak cocok.</p>}
              {n > 0 && !build && (
                <section className="contrast">
                  <Strip title="Paling atas" tone="good" faces={scored.slice(0, n)} img={img} onOpen={setOpen} />
                  <Strip title="Paling bawah" tone="bad" faces={scored.slice(-n).reverse()} img={img} onOpen={setOpen} />
                </section>
              )}
              {!build && <nav className="tabs">
                {([['lolos', `Lolos · ${kept.length}`], ['tersaring', `Tersaring · ${dropped.length}`], ['semua', `Semua · ${faces.length}`], ...(unread.length ? [['tidak-terbaca', `Foto tidak terbaca · ${unread.length}`] as const] : [])] as const).map(([t, label]) => (
                  <button key={t} aria-pressed={search.tab === t} onClick={() => set({ tab: t })}>{label}</button>
                ))}
              </nav>}
              {search.tab === 'tidak-terbaca' && !build ? (
                <Unread list={unread} onDownload={() => save(unreadableCsv(unread), `foto-tidak-terbaca-${search.source ?? 'semua'}${g}.csv`)} />
              ) : (
              <div className={`grid${loading ? ' dim' : ''}`}>
                {shown.map((f) => (
                  <Tile key={f.id} f={f} tags={tags.get(f.id)} img={img} out={f.skor == null || f.skor < cut} onOpen={build ? mark : setOpen} showOut={search.tab === 'semua' && !build} mark={build ? marks.get(f.id) : undefined} />
                ))}
                {!shown.length && <p className="muted">Tidak ada pelamar di sini.</p>}
              </div>
              )}
            </>
          )}
        </main>
      </div>

      {open && <Viewer list={shown.includes(open) ? shown : scored} face={open} cut={cut} img={img} tags={tags.get(open.id) ?? []} allTags={tagCounts.map(([t]) => t)} onTag={(t) => toggleTag(open, t)} onClose={() => setOpen(null)} onMove={setOpen} />}
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

function Tile({ f, tags, img, out, showOut, onOpen, mark }: { f: Face; tags?: string[]; img: ImgFn; out: boolean; showOut: boolean; onOpen: (f: Face) => void; mark?: 'suka' | 'tidak' }) {
  return (
    <button className={`tile${showOut && out ? ' out' : ''}${mark ? ` mark-${mark}` : ''}`} onClick={() => onOpen(f)}>
      <img src={img(f.id)} alt="" loading="lazy" />
      {mark && <span className="badge">{mark === 'suka' ? 'Suka' : 'Tidak'}</span>}
      {!mark && !!tags?.length && <span className="tagbadge" title={tags.join(', ')}>{tags[0]}{tags.length > 1 ? ` +${tags.length - 1}` : ''}</span>}
      <div className="meta">
        <b className={out ? 'bad' : 'good'}>{f.skor == null ? '–' : Math.round(f.skor)}</b>
        <span>#{f.id}</span>
      </div>
    </button>
  )
}

function Viewer({ list, face, cut, img, tags, allTags, onTag, onClose, onMove }: { list: Face[]; face: Face; cut: number; img: ImgFn; tags: string[]; allTags: string[]; onTag: (t: string) => void; onClose: () => void; onMove: (f: Face) => void }) {
  const [typed, setTyped] = useState('')
  const [n, setN] = useState(-1) // -1 = face crop, 0.. = whole photos
  useEffect(() => setN(-1), [face])
  const i = list.indexOf(face)
  const go = useCallback((d: number) => { const j = i + d; if (j >= 0 && j < list.length) onMove(list[j]) }, [i, list, onMove])
  useEffect(() => {
    const k = (e: KeyboardEvent) => { if (e.target instanceof HTMLInputElement) return; if (e.key === 'Escape') onClose(); if (e.key === 'ArrowLeft') go(-1); if (e.key === 'ArrowRight') go(1) }
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
        <div className="tagbox">
          {[...new Set([...allTags, ...tags])].map((t) => (
            <button key={t} className="chip tagchip" aria-pressed={tags.includes(t)} onClick={() => onTag(t)}>{tags.includes(t) ? '✓ ' : '+ '}{t}</button>
          ))}
          <form onSubmit={(e) => { e.preventDefault(); const t = typed.trim(); if (t) { if (!tags.includes(t)) onTag(t); setTyped('') } }}>
            <input value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="Tag lain…" maxLength={30} />
          </form>
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

const PHOTO_MIN = 10 // same as the worker: fewer examples than this is close to a coin flip

function Builder({ marks, onSave, onCancel }: { marks: Map<number, 'suka' | 'tidak'>; onSave: (name: string) => Promise<string>; onCancel: () => void }) {
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const suka = [...marks.values()].filter((v) => v === 'suka').length
  const tidak = marks.size - suka
  const ready = suka >= PHOTO_MIN && tidak >= PHOTO_MIN && name.trim().length > 0
  async function save() {
    setBusy(true); setError('')
    setError(await onSave(name.trim()).catch(() => 'Gagal menyimpan. Coba lagi.'))
    setBusy(false)
  }
  return (
    <section className="panel builder">
      <label className="lab">Taste baru dari foto</label>
      <div className="stats">
        <div><b className="good">{suka}</b><span>Suka</span></div>
        <div><b className="bad">{tidak}</b><span>Tidak</span></div>
      </div>
      <p className="muted small">Minimal {PHOTO_MIN} Suka dan {PHOTO_MIN} Tidak. Makin banyak contoh, makin bisa dipercaya.</p>
      <input placeholder="Nama taste, mis. Pemeran utama" value={name} onChange={(e) => setName(e.target.value)} maxLength={60} />
      <button className="primary wide" disabled={!ready || busy} onClick={save}>{busy ? 'Menyimpan…' : 'Simpan taste'}</button>
      <button className="ghost wide" onClick={onCancel}>Batal</button>
      {error && <p className="err">{error}</p>}
    </section>
  )
}
