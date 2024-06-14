import { IntegrationKind } from '@/lib/gql/generates/graphql'

export const PROVIDER_KIND_METAS: Array<{
  name: string
  enum: IntegrationKind
  meta: {
    displayName: string
  }
}> = [
  {
    name: 'github',
    enum: IntegrationKind.Github,
    meta: {
      displayName: 'GitHub'
    }
  },
  {
    name: 'github-self-hosted',
    enum: IntegrationKind.GithubSelfHosted,
    meta: {
      displayName: 'GitHub Self-Hosted'
    }
  },
  {
    name: 'gitlab',
    enum: IntegrationKind.Gitlab,
    meta: {
      displayName: 'GitLab'
    }
  },
  {
    name: 'gitlab-self-hosted',
    enum: IntegrationKind.GitlabSelfHosted,
    meta: {
      displayName: 'GitLab Self-Hosted'
    }
  }
]
