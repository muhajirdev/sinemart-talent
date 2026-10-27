import { Container } from '@cloudflare/containers'

// The face models (container/server.py). One shared instance: it sleeps
// after 30 quiet minutes, and waking it loads ~2 GB of weights, so the first
// photo after a pause is slow. No internet: the weights are in the image.
export class ScorerContainer extends Container {
  defaultPort = 8080
  sleepAfter = '30m'
  enableInternet = false
}
