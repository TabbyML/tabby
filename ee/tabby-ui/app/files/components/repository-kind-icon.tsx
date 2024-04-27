import { RepositoryKind } from '@/lib/gql/generates/graphql'
import { IconGit, IconGitHub, IconGitLab } from '@/components/ui/icons'

export function RepositoryKindIcon({
  kind,
  fallback
}: {
  kind: RepositoryKind | undefined
  fallback?: React.ReactNode
}) {
  switch (kind) {
    case RepositoryKind.Git:
      return <IconGit />
    case RepositoryKind.Github:
      return <IconGitHub />
    case RepositoryKind.Gitlab:
      return <IconGitLab />
    default:
      return fallback ?? null
  }
}
