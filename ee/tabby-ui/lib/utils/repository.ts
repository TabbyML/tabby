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

  const filteredRepos = repositories.filter(repo => {
    const search = gitUrlParse(repo.url)
    const isSameResource =
      search.resource === targetSearch.resource || search.protocol === 'file'
    return isSameResource && search.name === targetSearch.name
  })

  if (filteredRepos.length === 0) {
    return undefined
  } else {
    // If there're multiple matches, we pick the one with highest alphabetical order
    return filteredRepos.reduce((min, current) => {
      const minUrl = canonicalizeUrl(min.url)
      const currentUrl = canonicalizeUrl(current.url)
      return minUrl > currentUrl ? min : current
    })
  }
}

export function canonicalizeUrl(url: string): string {
  const strippedUrl = url.replace(/\.git$/, '')
  const parsedUrl = new URL(strippedUrl)
  parsedUrl.username = ''
  parsedUrl.password = ''
  return parsedUrl.toString()
}
