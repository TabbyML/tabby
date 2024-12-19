import gitUrlParse from 'git-url-parse'
import type { GitRepository } from 'tabby-chat-panel'

export function findClosestGitRepository<T extends GitRepository>(
  repositories: T[],
  gitUrl: string
): T | undefined {
  const targetSearch = gitUrlParse(gitUrl)
  if (!targetSearch) {
    return undefined
  }

  const filteredRepos = repositories.filter(repo => {
    const search = gitUrlParse(repo.url)
    return search.name === targetSearch.name
  })

  if (filteredRepos.length === 0) {
    return undefined
  } else {
    // If there're multiple matches, we pick the one with highest alphabetical order
    return filteredRepos.sort((a, b) => a.url.localeCompare(b.url))[0]
  }
}
