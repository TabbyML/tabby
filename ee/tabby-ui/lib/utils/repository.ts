import { go } from 'fuzzysort'

export const findClosestRepositoryMatch = (
  target: string,
  gitUrls: string[]
) => {
  const results = go(target.replace(/\.git$/, ''), gitUrls)
  return results.length > 0 ? results[0].target : null
}
