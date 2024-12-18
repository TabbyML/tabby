import gitUrlParse from 'git-url-parse'
import type { GitRepository } from 'tabby-chat-panel'

export function findClosestGitRepository(
  repositories: GitRepository[],
  gitUrl: string
): GitRepository | undefined {
  const gitSearch = gitUrlParse(gitUrl)
  if (!gitSearch) {
    return undefined
  }

  const repos = repositories.filter(repo => {
    const search = gitUrlParse(repo.url)
    return search.name === gitSearch.name
  })
  // If there're multiple matches, we pick the one with highest alphabetical order
  return repos.sort((a, b) => b.url.localeCompare(a.url))[0]
}
