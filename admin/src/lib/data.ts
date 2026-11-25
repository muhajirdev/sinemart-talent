// What /api/casting (worker/src/index.ts) sends: everyone in one source with
// the team's score on the Lab's 0-100 scale (chance to beat an average applicant).
export type Face = { id: number; gender: 'f' | 'm' | null; source: string; n_photos: number; skor: number | null; name?: string | null; nick?: string | null }
// People whose photo gave no usable face: listed so the admin can check them in the form by hand.
export type Unreadable = { id: number; row: number | null; gender: 'f' | 'm' | null; reason: string; name?: string | null; nick?: string | null }
// Full name, with the nickname when it adds something.
export const displayName = (x: { name?: string | null; nick?: string | null }) =>
  x.name && x.name !== '-' ? x.name + (x.nick && x.nick.toLowerCase() !== x.name.toLowerCase() ? ` (${x.nick})` : '') : x.nick || null
// tastes: raters with enough comparisons to have their own taste; taste: the one scoring this list (null = the team).
export type Casting = { sources: { source: string; n: number }[]; faces: Face[]; unreadable?: Unreadable[]; tastes?: Taste[]; taste?: string | null; tags?: Record<number, string[]> }
// key = rater name, or 'foto:<slug>' for a taste built from example photos.
export type Taste = { key: string; name: string; n: number; kind: 'rater' | 'foto'; by?: string }

export const REASON: Record<string, string> = {
  no_photo: 'Tidak ada foto',
  no_face: 'Wajah tidak ketemu',
  low_face_score: 'Wajah kurang jelas',
  face_too_small: 'Wajah terlalu kecil',
  blurry: 'Foto buram',
  bad_exposure: 'Terlalu gelap / terang',
  not_frontal: 'Wajah tidak menghadap depan',
  multiple_faces: 'Ada beberapa wajah',
  unreadable: 'File tidak bisa dibuka',
  no_gender: 'Gender tidak diisi',
}
export const reasonName = (r: string) => REASON[r] ?? r

export function unreadableCsv(list: Unreadable[]) {
  const rows = list.map((u) => [u.id, (u.name ?? '').replace(/,/g, ' '), u.row ?? '', u.gender ? GENDER[u.gender].toLowerCase() : '', reasonName(u.reason)])
  return [['no_pelamar', 'nama', 'baris_form', 'gender', 'alasan'], ...rows].map((r) => r.join(',')).join('\n') + '\n'
}

export const SOURCE_NAME: Record<string, string> = {
  'sctv-mencari-bintang-2026': 'Form Mencari Bintang 2026',
  '2026': 'Pelamar 2026',
  '2021': 'Pelamar 2021',
  bc: 'Showcase',
}
export const sourceName = (s: string) => SOURCE_NAME[s] ?? s

// SCTV form: id = 910000 + response index, so its sheet row is id − 910000 + 2.
export const formRow = (f: Face) => (f.source === 'sctv-mencari-bintang-2026' ? f.id - 910000 + 2 : null)

export const GENDER = { m: 'Cowok', f: 'Cewek' } as const

export function toCsv(faces: Face[]) {
  const head = ['no_pelamar', 'nama', 'baris_form', 'gender', 'sumber', 'skor']
  const rows = faces.map((f) => [f.id, (f.name ?? '').replace(/,/g, ' '), formRow(f) ?? '', f.gender ? GENDER[f.gender].toLowerCase() : '', sourceName(f.source), f.skor ?? ''])
  return [head, ...rows].map((r) => r.join(',')).join('\n') + '\n'
}

// 2026-11-25 (Muajir: "a way to mark or tag"): ready-made tags; any other word can be typed.
export const PRESET_TAGS = ['Shortlist', 'Panggil casting', 'Ragu', 'Tidak cocok']

// 2026-11-25 (Muajir: "an option to export ... as excel or csv"): everyone in the
// view, best first, with Lolos / Tersaring at the current cutoff and their tags.
export type ExportRow = { f: Face; status: string; tags: string[] }
export const EXPORT_HEAD = ['No. pelamar', 'Nama', 'Panggilan', 'Baris form', 'Gender', 'Sumber', 'Skor', 'Status', 'Tag']
export function exportCells(r: ExportRow): (string | number | null)[] {
  return [r.f.id, r.f.name ?? null, r.f.nick ?? null, formRow(r.f), r.f.gender ? GENDER[r.f.gender] : null, sourceName(r.f.source), r.f.skor, r.status, r.tags.join(', ') || null]
}
export function exportCsv(rows: ExportRow[]) {
  const q = (v: string | number | null) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v))
  return [EXPORT_HEAD, ...rows.map(exportCells)].map((r) => r.map(q).join(',')).join('\n') + '\n'
}
