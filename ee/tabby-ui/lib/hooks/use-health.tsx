'use client'

import useSWR, { SWRResponse } from 'swr'

import fetcher from '@/lib/tabby/fetcher'

import { useAuthenticatedApi } from '../tabby/auth'

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
  return useSWR(useAuthenticatedApi('/v1/health'), fetcher)
}
