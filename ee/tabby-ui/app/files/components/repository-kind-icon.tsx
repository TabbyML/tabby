import { RepositoryKind } from '@/lib/gql/generates/graphql'
import {
  IconDirectorySolid,
  IconGitHub,
  IconGitLab
} from '@/components/ui/icons'

export function RepositoryKindIcon({
  kind,
  fallback
}: {
  kind: RepositoryKind | undefined
  fallback?: React.ReactNode
}) {
  switch (kind) {
    case RepositoryKind.Git:
    case RepositoryKind.GitConfig:
      return <IconDirectorySolid style={{ color: 'rgb(84, 174, 255)' }} />
    case RepositoryKind.Github:
    case RepositoryKind.GithubSelfHosted:
      return <IconGitHub />
    case RepositoryKind.Gitlab:
    case RepositoryKind.GitlabSelfHosted:
      return <IconGitLab />
    default:
      return fallback ?? null
  }
}
