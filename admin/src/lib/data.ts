// What /api/casting (worker/src/index.ts) sends: everyone in one source with
// the team's score on the Lab's 0-100 scale (chance to beat an average applicant).
export type Face = { id: number; gender: 'f' | 'm' | null; source: string; n_photos: number; skor: number | null }
// People whose photo gave no usable face: listed so the admin can check them in the form by hand.
export type Unreadable = { id: number; row: number | null; gender: 'f' | 'm' | null; reason: string }
export type Casting = { sources: { source: string; n: number }[]; faces: Face[]; unreadable?: Unreadable[] }

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
  const rows = list.map((u) => [u.id, u.row ?? '', u.gender ? GENDER[u.gender].toLowerCase() : '', reasonName(u.reason)])
  return [['no_pelamar', 'baris_form', 'gender', 'alasan'], ...rows].map((r) => r.join(',')).join('\n') + '\n'
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
  const head = ['no_pelamar', 'baris_form', 'gender', 'sumber', 'skor']
  const rows = faces.map((f) => [f.id, formRow(f) ?? '', f.gender ? GENDER[f.gender].toLowerCase() : '', sourceName(f.source), f.skor ?? ''])
  return [head, ...rows].map((r) => r.join(',')).join('\n') + '\n'
}
