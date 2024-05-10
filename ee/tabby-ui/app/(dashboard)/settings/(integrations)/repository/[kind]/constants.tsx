import { RepositoryKind } from '@/lib/gql/generates/graphql'

export const REPOSITORY_KIND_METAS: Array<{
  name: string
  enum: RepositoryKind
  meta: {
    domain: string
    displayName: string
  }
}> = [
  {
    name: 'github',
    enum: RepositoryKind.Github,
    meta: {
      domain: 'github.com',
      displayName: 'GitHub'
    }
  },
  {
    name: 'gitlab',
    enum: RepositoryKind.Gitlab,
    meta: {
      domain: 'gitlab.com',
      displayName: 'GitLab'
    }
  }
]
