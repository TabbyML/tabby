import gitUrlParse from 'git-url-parse'
import type { GitRepository } from 'tabby-chat-panel'

export function findClosestGitRepository(
  repositories: GitRepository[],
  gitUrl: string
): GitRepository | undefined {
  const targetSearch = gitUrlParse(gitUrl)
  if (!targetSearch) {
    return undefined
  }

  const repos = repositories.filter(repo => {
    const search = gitUrlParse(repo.url)
    const isSameResource =
      search.resource === targetSearch.resource || search.protocol === 'file'
    return isSameResource && search.name === targetSearch.name
  })

  // If there're multiple matches, we pick the one with highest alphabetical order
  return repos.sort((a, b) => {
    const canonicalUrlA = canonicalizeUrl(a.url)
    const canonicalUrlB = canonicalizeUrl(b.url)
    return canonicalUrlB.localeCompare(canonicalUrlA)
  })[0]
}

export function canonicalizeUrl(url: string): string {
  const strippedUrl = url.replace(/\.git$/, '')
  const parsedUrl = new URL(strippedUrl)
  parsedUrl.username = ''
  parsedUrl.password = ''
  return parsedUrl.toString()
}
