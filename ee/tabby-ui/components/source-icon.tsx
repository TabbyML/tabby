import { ReactNode } from 'react'

import { ContextSourceKind } from '@/lib/gql/generates/graphql'
import {
  IconCode,
  IconEmojiBook,
  IconEmojiGlobe,
  IconGitHub,
  IconGitLab
} from '@/components/ui/icons'

interface SourceIconProps {
  kind: ContextSourceKind
  gitIcon?: ReactNode
  className?: string
}

export function SourceIcon({ kind, gitIcon, ...props }: SourceIconProps) {
  switch (kind) {
    case ContextSourceKind.Doc:
      return <IconEmojiBook {...props} />
    case ContextSourceKind.Web:
      return <IconEmojiGlobe {...props} />
    case ContextSourceKind.Github:
      return <IconGitHub {...props} />
    case ContextSourceKind.Gitlab:
      return <IconGitLab {...props} />
    case ContextSourceKind.Git:
      // for custom git icon
      return gitIcon || <IconCode {...props} />
    default:
      return null
  }
}
