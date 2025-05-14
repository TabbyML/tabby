import React from 'react'
import { groupBy } from 'lodash-es'

import { ModelHealthBackend } from '../gql/generates/graphql'
import { useHealth, type HealthInfo } from './use-health'

export enum ModelSource {
  local = 'local',
  remote = 'remote'
}

function transformHealthInfoToCompletionWorker(healthInfo: HealthInfo) {
  const remoteModel = healthInfo.models.completion?.remote

  return {
    kind: ModelHealthBackend.Completion,
    device: healthInfo.device,
    addr: remoteModel ? remoteModel.api_endpoint || 'localhost' : 'localhost',
    arch: '',
    cpuInfo: healthInfo.cpu_info,
    name: healthInfo.model!,
    cpuCount: healthInfo.cpu_count,
    cudaDevices: healthInfo.cuda_devices,
    source: remoteModel ? ModelSource.remote : ModelSource.local
  }
}

function transformHealthInfoToChatWorker(healthInfo: HealthInfo) {
  const remoteModel = healthInfo.models.chat?.remote

  return {
    kind: ModelHealthBackend.Chat,
    device: healthInfo.device,
    addr: remoteModel ? remoteModel.api_endpoint || 'localhost' : 'localhost',
    arch: '',
    cpuInfo: healthInfo.cpu_info,
    name: healthInfo.chat_model!,
    cpuCount: healthInfo.cpu_count,
    cudaDevices: healthInfo.cuda_devices,
    source: remoteModel ? ModelSource.remote : ModelSource.local
  }
}

function transformHealthInfoToEmbeddingWorker(healthInfo: HealthInfo) {
  const remoteModel = healthInfo.models.embedding.remote
  const localModel = healthInfo.models.embedding.local

  return {
    kind: ModelHealthBackend.Embedding,
    device: healthInfo.device,
    addr: remoteModel ? remoteModel.api_endpoint || 'localhost' : 'localhost',
    arch: '',
    cpuInfo: healthInfo.cpu_info,
    name:
      (remoteModel ? remoteModel.model_name : localModel?.model_id) ||
      'Embedding',
    cpuCount: healthInfo.cpu_count,
    cudaDevices: healthInfo.cuda_devices,
    source: remoteModel ? ModelSource.remote : ModelSource.local
  }
}

function useWorkers() {
  const { data: healthInfo, isLoading, error } = useHealth()
  const groupedWorkers = React.useMemo(() => {
    const workers = []

    if (healthInfo?.models.completion) {
      workers.push(transformHealthInfoToCompletionWorker(healthInfo))
    }
    if (healthInfo?.models.chat) {
      workers.push(transformHealthInfoToChatWorker(healthInfo))
    }
    if (healthInfo) {
      workers.push(transformHealthInfoToEmbeddingWorker(healthInfo))
    }
    return groupBy(workers, 'kind')
  }, [healthInfo])

  return { data: groupedWorkers, isLoading, error }
}

export { useWorkers }
