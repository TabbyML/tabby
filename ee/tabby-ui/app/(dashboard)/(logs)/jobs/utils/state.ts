import { isNil } from 'lodash-es'

export function getLabelByExitCode(exitCode?: number | null) {
  return isNil(exitCode) ? 'Running' : exitCode === 0 ? 'Success' : 'Failed'
}
