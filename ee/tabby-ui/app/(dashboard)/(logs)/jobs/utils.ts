import { isNil } from 'lodash-es'

import { JobRun } from '@/lib/gql/generates/graphql'

const JOB_DISPLAY_NAME_MAPPINGS = {
  scheduler_git: 'Git',
  scheduler_github_gitlab: 'Github / Gitlab',
  web_crawler: 'Docs'
}

export function getJobDisplayName(name: string): string {
  if (name in JOB_DISPLAY_NAME_MAPPINGS) {
    return JOB_DISPLAY_NAME_MAPPINGS[
      name as keyof typeof JOB_DISPLAY_NAME_MAPPINGS
    ]
  } else {
    return name
  }
}

// status: pending, running, success, failed
export function getLabelByJobRun(
  info?: Pick<JobRun, 'exitCode' | 'createdAt' | 'startedAt'>
) {
  if (!info) return 'Pending'
  if (isNil(info.exitCode)) {
    return info.startedAt ? 'Running' : 'Pending'
  }
  return info.exitCode === 0 ? 'Success' : 'Failed'
}
