import { isNil } from 'lodash-es'

export function getLabelByExitCode(exitCode?: number | null) {
  return isNil(exitCode) ? 'Pending' : exitCode === 0 ? 'Success' : 'Failed'
}
