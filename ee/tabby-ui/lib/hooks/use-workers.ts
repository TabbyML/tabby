import React from 'react'
import { groupBy } from 'lodash-es'

import { useHealth, type HealthInfo } from './use-health'

function transformHealthInfoToCompletionWorker(healthInfo: HealthInfo) {
  return {
    kind: 'COMPLETION',
    device: healthInfo.device,
    addr: 'localhost',
    arch: '',
    cpuInfo: healthInfo.cpu_info,
    name: healthInfo.model!,
    cpuCount: healthInfo.cpu_count,
    cudaDevices: healthInfo.cuda_devices
  }
}

function transformHealthInfoToChatWorker(healthInfo: HealthInfo) {
  return {
    kind: 'CHAT',
    device: healthInfo.chat_device!,
    addr: 'localhost',
    arch: '',
    cpuInfo: healthInfo.cpu_info,
    name: healthInfo.chat_model!,
    cpuCount: healthInfo.cpu_count,
    cudaDevices: healthInfo.cuda_devices
  }
}

function useWorkers() {
  const { data: healthInfo, isLoading: fetching } = useHealth()

  const groupedWorkers = React.useMemo(() => {
    const workers = []

    if (healthInfo?.model) {
      workers.push(transformHealthInfoToCompletionWorker(healthInfo))
    }
    if (healthInfo?.chat_model) {
      workers.push(transformHealthInfoToChatWorker(healthInfo))
    }
    return groupBy(workers, 'kind')
  }, [healthInfo])

  return { data: groupedWorkers, fetching }
}

export { useWorkers }
