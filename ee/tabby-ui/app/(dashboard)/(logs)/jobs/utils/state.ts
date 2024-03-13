import { isNil } from 'lodash-es'

export function findColorByExitCode(exitCode?: number | null) {
  return isNil(exitCode)
    ? 'orange-400'
    : exitCode === 0
    ? 'green-400'
    : 'red-400'
}

export function findLabelByExitCode(exitCode?: number | null) {
  return isNil(exitCode) ? 'Running' : exitCode === 0 ? 'Success' : 'Failed'
}
