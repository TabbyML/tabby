import { useParams } from 'next/navigation'
import { findIndex } from 'lodash-es'

import { PROVIDER_KIND_METAS } from '../../constants'

export function useIntegrationKind() {
  const params = useParams<{ kind?: string }>()
  const kindIndex = findIndex(
    PROVIDER_KIND_METAS,
    item => item.name === params.kind?.toLowerCase()
  )
  const kind =
    kindIndex > -1
      ? PROVIDER_KIND_METAS[kindIndex].enum
      : PROVIDER_KIND_METAS[0].enum
  return kind
}
