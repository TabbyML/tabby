'use client'

import useSWRImmutable from 'swr/immutable'
import { SWRResponse } from 'swr'
import fetcher from '@/lib/tabby-fetcher'

export interface HealthInfo {
  device: string
  model?: string
  chat_model?: string
  version: {
    build_date: string
    git_describe: string
  }
}

export function useHealth(): SWRResponse<HealthInfo> {
  return useSWRImmutable('/v1/health', fetcher)
}
