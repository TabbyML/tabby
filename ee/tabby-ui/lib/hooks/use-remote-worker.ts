import { groupBy, findIndex } from 'lodash-es'
import {
  GetWorkersDocument,
  Worker,
  WorkerKind
} from '@/lib/gql/generates/graphql'
import { useGraphQL } from './use-graphql'
import type { HealthInfo } from './use-health'

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
  let workers = data?.workers || []

  const haveRemoteCompletionWorkers =
    findIndex(workers, { kind: WorkerKind.Completion }) > -1
  const haveRemoteChatWorkers =
    findIndex(workers, { kind: WorkerKind.Chat }) > -1

  if (!haveRemoteCompletionWorkers && healthInfo?.model) {
    workers.push(transformHealthInfoToWorker(healthInfo, WorkerKind.Completion))
  }
  if (!haveRemoteChatWorkers && healthInfo?.chat_model) {
    workers.push(transformHealthInfoToWorker(healthInfo, WorkerKind.Chat))
  }
  return groupBy(workers, worker => worker.kind)
}

export { useRemoteWorkers, useMergedWorkers }
