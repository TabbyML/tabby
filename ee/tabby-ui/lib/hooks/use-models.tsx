'use client'

import useSWR, { SWRResponse } from 'swr'

import fetcher from '@/lib/tabby/fetcher'

export interface ModelInfo {
  completion: Array<string>
  chat: Array<string>
}

export function useModel(): SWRResponse<ModelInfo> {
  return useSWR(
    '/v1beta/models',
    (url: string) => {
      return fetcher(url, {
        errorHandler: () => {
          throw new Error('Fetch supported model failed.')
        }
      })
    },
    {
      shouldRetryOnError: false
    }
  )
}
