import React from 'react'
import { findIndex, groupBy, slice } from 'lodash-es'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import { Worker, WorkerKind } from '@/lib/gql/generates/graphql'

import { useHealth, type HealthInfo } from './use-health'

function transformHealthInfoToCompletionWorker(healthInfo: HealthInfo): Worker {
  return {
    kind: WorkerKind.Completion,
    device: healthInfo.device,
    addr: 'localhost',
    arch: '',
    cpuInfo: healthInfo.cpu_info,
    name: healthInfo.model!,
    cpuCount: healthInfo.cpu_count,
    cudaDevices: healthInfo.cuda_devices
  }
}

function transformHealthInfoToChatWorker(healthInfo: HealthInfo): Worker {
  return {
    kind: WorkerKind.Chat,
    device: healthInfo.chat_device!,
    addr: 'localhost',
    arch: '',
    cpuInfo: healthInfo.cpu_info,
    name: healthInfo.chat_model!,
    cpuCount: healthInfo.cpu_count,
    cudaDevices: healthInfo.cuda_devices
  }
}

export const getAllWorkersDocument = graphql(/* GraphQL */ `
  query GetWorkers {
    workers {
      kind
      name
      addr
      device
      arch
      cpuInfo
      cpuCount
      cudaDevices
    }
  }
`)

function useWorkers() {
  const { data: healthInfo } = useHealth()
  const [{ data, fetching }] = useQuery({ query: getAllWorkersDocument })
  let workers = data?.workers

  const groupedWorkers = React.useMemo(() => {
    const _workers = slice(workers)
    const haveRemoteCompletionWorkers =
      findIndex(_workers, { kind: WorkerKind.Completion }) > -1
    const haveRemoteChatWorkers =
      findIndex(_workers, { kind: WorkerKind.Chat }) > -1

    if (!haveRemoteCompletionWorkers && healthInfo?.model) {
      _workers.push(transformHealthInfoToCompletionWorker(healthInfo))
    }
    if (!haveRemoteChatWorkers && healthInfo?.chat_model) {
      _workers.push(transformHealthInfoToChatWorker(healthInfo))
    }
    return groupBy(_workers, 'kind')
  }, [healthInfo, workers])

  return { data: groupedWorkers, fetching }
}

export { useWorkers }
