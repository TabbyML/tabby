'use client'

import useSWR, { SWRResponse } from 'swr'

import fetcher from '@/lib/tabby/fetcher'

export interface HealthInfo {
  device: 'metal' | 'cpu' | 'cuda'
  model?: string
  chat_model?: string
  chat_device?: string
  cpu_info: string
  cpu_count: number
  cuda_devices: string[]
  version: {
    build_date: string
    git_describe: string
  }
}

export function useHealth(): SWRResponse<HealthInfo> {
  return useSWR(
    '/v1/health',
    (url: string) => {
      return fetcher(url, {
        errorHandler: response => {
          throw new Error(response?.statusText.toString() || 'Unhealth')
        }
      })
    },
    {
      shouldRetryOnError: false
    }
  )
}
