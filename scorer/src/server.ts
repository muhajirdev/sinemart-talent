import handler, { createServerEntry } from '@tanstack/react-start/server-entry'

export { ScorerContainer } from './lib/container'

// Like the main app: public, but kept out of search engines.
export default createServerEntry({
  async fetch(request, opts) {
    const res = await handler.fetch(request, opts)
    const out = new Response(res.body, res)
    out.headers.set('x-robots-tag', 'noindex, nofollow, noimageindex')
    return out
  },
})
