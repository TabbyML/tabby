'use client'

import useSWRImmutable from 'swr/immutable'
import { SWRResponse } from 'swr'

export interface ReleaseInfo {
  name: string
}

export function useLatestRelease(): SWRResponse<ReleaseInfo> {
  const fetcher = (url: string) => fetch(url).then(x => x.json())
  return useSWRImmutable(
    'https://api.github.com/repos/TabbyML/tabby/releases/latest',
    fetcher
  )
}
