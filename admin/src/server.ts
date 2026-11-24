import handler, { createServerEntry } from '@tanstack/react-start/server-entry'
import { env } from 'cloudflare:workers'

// Sign-in, scores and photos live in the main app: pass those straight through.
export default createServerEntry({
  async fetch(request, opts) {
    const path = new URL(request.url).pathname
    const res = path.startsWith('/api/') || path.startsWith('/img/')
      ? await env.TALENT.fetch(request)
      : await handler.fetch(request, opts)
    const out = new Response(res.body, res)
    out.headers.set('x-robots-tag', 'noindex, nofollow, noimageindex')
    return out
  },
})
