import React from 'react'
import { findIndex, groupBy, slice } from 'lodash-es'

import { graphql } from '@/lib/gql/generates'
import { Worker, WorkerKind } from '@/lib/gql/generates/graphql'
import { useAuthenticatedGraphQLQuery } from '@/lib/tabby/gql'

import { useHealth, type HealthInfo } from './use-health'

const modelNameMap: Record<WorkerKind, 'chat_model' | 'model'> = {
  [WorkerKind.Chat]: 'chat_model',
  [WorkerKind.Completion]: 'model'
}

function transformHealthInfoToWorker(
  healthInfo: HealthInfo,
  kind: WorkerKind
): Worker {
  return {
    kind,
    device: healthInfo.device,
    addr: 'localhost',
    arch: '',
    cpuInfo: healthInfo.cpu_info,
    name: healthInfo?.[modelNameMap[kind]] ?? '',
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
  const { data } = useAuthenticatedGraphQLQuery(getAllWorkersDocument)
  let workers = data?.workers

  const groupedWorkers = React.useMemo(() => {
    const _workers = slice(workers)
    const haveRemoteCompletionWorkers =
      findIndex(_workers, { kind: WorkerKind.Completion }) > -1
    const haveRemoteChatWorkers =
      findIndex(_workers, { kind: WorkerKind.Chat }) > -1

    if (!haveRemoteCompletionWorkers && healthInfo?.model) {
      _workers.push(
        transformHealthInfoToWorker(healthInfo, WorkerKind.Completion)
      )
    }
    if (!haveRemoteChatWorkers && healthInfo?.chat_model) {
      _workers.push(transformHealthInfoToWorker(healthInfo, WorkerKind.Chat))
    }
    return groupBy(_workers, 'kind')
  }, [healthInfo, workers])

  return groupedWorkers
}

export { useWorkers }
