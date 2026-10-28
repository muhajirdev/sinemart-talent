import { createFileRoute } from '@tanstack/react-router'
import { knownRater } from '#/lib/score'

export const Route = createFileRoute('/api/login')({
  server: {
    handlers: {
      POST: async ({ request }) => {
        const { name } = (await request.json().catch(() => ({}))) as { name?: string }
        const r = await knownRater(String(name ?? ''))
        if (!r) return Response.json({ error: 'Nama belum terdaftar. Masuk dulu di halaman pilih.' }, { status: 401 })
        return Response.json(r)
      },
    },
  },
})
