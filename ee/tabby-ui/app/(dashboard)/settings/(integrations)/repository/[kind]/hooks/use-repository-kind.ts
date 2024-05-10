import { useParams } from 'next/navigation'
import { findIndex } from 'lodash-es'

import { REPOSITORY_KIND_METAS } from '../constants'

export function useRepositoryKind() {
  const params = useParams<{ kind?: string }>()
  const kindIndex = findIndex(
    REPOSITORY_KIND_METAS,
    item => item.enum.toLocaleLowerCase() === params.kind?.toLocaleLowerCase()
  )
  const kind =
    kindIndex > -1
      ? REPOSITORY_KIND_METAS[kindIndex].enum
      : REPOSITORY_KIND_METAS[0].enum
  return kind
}
