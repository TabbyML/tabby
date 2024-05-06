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
      return <IconDirectorySolid style={{ color: 'rgb(84, 174, 255)' }} />
    case RepositoryKind.Github:
      return <IconGitHub />
    case RepositoryKind.Gitlab:
      return <IconGitLab />
    default:
      return fallback ?? null
  }
}
