# Casting admin ("Sinemart Casting")

For Sinemart's casting admin: pick a source and gender (default cowok), slide a
score cutoff, see who passes and who is filtered (with a score histogram and
the top vs bottom faces side by side), open anyone's photos, and download the
passing list as CSV. Built 2026-11-24 at Muajir's request.

TanStack Start on Workers (`src/`). It holds no data: `/api/*` and `/img/*` go
to the main app (`worker/`, service binding `TALENT`), so sign-in, scores and
photos are the same as the picking page and Lab. Read-only; the view (source,
gender, cutoff, tab) lives in the URL.

```
npm install
npm run dev      # http://localhost:3001, talks to the live main app (remote binding)
npm run deploy   # needs Muajir's OK, like any deploy
```
