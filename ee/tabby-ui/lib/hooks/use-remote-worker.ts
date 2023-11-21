import { groupBy, findIndex, slice } from 'lodash-es'
import {
  GetWorkersDocument,
  Worker,
  WorkerKind
} from '@/lib/gql/generates/graphql'
import { useGraphQL } from './use-graphql'
import type { HealthInfo } from './use-health'
import React from 'react'

function useRemoteWorkers() {
  return useGraphQL(GetWorkersDocument)
}

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

function useMergedWorkers(healthInfo: HealthInfo | undefined) {
  const { data } = useRemoteWorkers()
  let workers = data?.workers

  const groupedWorkers = React.useMemo(() => {
    const _workers = slice(workers)
    const haveRemoteCompletionWorkers =
      findIndex(workers, { kind: WorkerKind.Completion }) > -1
    const haveRemoteChatWorkers =
      findIndex(workers, { kind: WorkerKind.Chat }) > -1

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

export { useRemoteWorkers, useMergedWorkers }
