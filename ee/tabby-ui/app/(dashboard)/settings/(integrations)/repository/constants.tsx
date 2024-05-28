import { RepositoryKind } from '@/lib/gql/generates/graphql'

export const PROVIDER_KIND_METAS: Array<{
  name: string
  enum: RepositoryKind
  meta: {
    displayName: string
  }
}> = [
  {
    name: 'github',
    enum: RepositoryKind.Github,
    meta: {
      displayName: 'GitHub'
    }
  },
  {
    name: 'github-self-hosted',
    enum: RepositoryKind.GithubSelfHosted,
    meta: {
      displayName: 'GitHub Self-Hosted'
    }
  },
  {
    name: 'gitlab',
    enum: RepositoryKind.Gitlab,
    meta: {
      displayName: 'GitLab'
    }
  },
  {
    name: 'gitlab-self-hosted',
    enum: RepositoryKind.GitlabSelfHosted,
    meta: {
      displayName: 'GitLab Self-Hosted'
    }
  }
]
