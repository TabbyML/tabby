'use client'

import useSWR, { SWRResponse } from 'swr'

import fetcher from '@/lib/tabby/fetcher'

import { useAuthenticatedApi } from '../tabby/auth'

export interface Accelerator {
  uuid?: string;
  chip_name?: string;
  display_name: string;
  device_type: 'Cuda' | 'Rocm';
}

export interface HealthInfo {
  device: 'metal' | 'cpu' | 'cuda' | 'rocm'
  model?: string
  chat_model?: string
  cpu_info: string
  cpu_count: number
  accelerators: Accelerator[]
  version: {
    build_date: string
    git_describe: string
  }
}

export function useHealth(): SWRResponse<HealthInfo> {
  return useSWR(useAuthenticatedApi('/v1/health'), fetcher)
}
