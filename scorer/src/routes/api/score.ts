import { createFileRoute } from '@tanstack/react-router'
import { raterName, scorePhotos } from '#/lib/score'

const MAX_PHOTOS = 6
const MAX_BYTES = 15 * 1024 * 1024

// POST multipart: photo (1-6 files of one person), gender auto|f|m.
// Header x-rater-token. Photos are scored in memory and dropped.
export const Route = createFileRoute('/api/score')({
  server: {
    handlers: {
      POST: async ({ request }) => {
        if (!(await raterName(request.headers.get('x-rater-token')))) {
          return Response.json({ error: 'Masuk dulu.' }, { status: 401 })
        }
        const form = await request.formData().catch(() => null)
        const files = (form?.getAll('photo') ?? []).filter((f): f is File => f instanceof File && f.size > 0)
        if (!files.length || files.length > MAX_PHOTOS) {
          return Response.json({ error: `Pilih 1-${MAX_PHOTOS} foto.` }, { status: 400 })
        }
        if (files.some((f) => f.size > MAX_BYTES)) return Response.json({ error: 'Foto terlalu besar (maks 15 MB).' }, { status: 413 })
        const g = String(form?.get('gender') ?? 'auto')
        const gender = g === 'f' || g === 'm' ? g : 'auto'
        try {
          return Response.json(await scorePhotos(await Promise.all(files.map((f) => f.arrayBuffer())), gender), {
            headers: { 'cache-control': 'no-store' },
          })
        } catch (e) {
          return Response.json({ error: e instanceof Error ? e.message : 'Gagal menilai foto.' }, { status: 502 })
        }
      },
    },
  },
})
