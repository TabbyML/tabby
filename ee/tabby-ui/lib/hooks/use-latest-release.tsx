'use client'

import { SWRResponse } from 'swr'
import useSWRImmutable from 'swr/immutable'

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
