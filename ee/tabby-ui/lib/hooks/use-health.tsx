'use client'

import useSWRImmutable from 'swr/immutable'
import { SWRResponse } from 'swr'
import fetcher from '@/lib/tabby/fetcher'

export interface HealthInfo {
  device: 'metal' | 'cpu' | 'cuda'
  model?: string
  chat_model?: string
  cpu_info: string
  cpu_count: number
  cuda_devices: string[]
  version: {
    build_date: string
    git_describe: string
  }
}

export function useHealth(): SWRResponse<HealthInfo> {
  return useSWRImmutable('/v1/health', fetcher)
}
