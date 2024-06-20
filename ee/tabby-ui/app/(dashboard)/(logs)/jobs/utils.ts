import { isNil } from 'lodash-es'

const JOB_DISPLAY_NAME_MAPPINGS = {
  "scheduler_git": "Git",
  "scheduler_github_gitlab": "Github / Gitlab",
  "web": "Web"
}

export function getJobDisplayName(name: string): string {
  if (name in JOB_DISPLAY_NAME_MAPPINGS) {
    return JOB_DISPLAY_NAME_MAPPINGS[name as keyof typeof JOB_DISPLAY_NAME_MAPPINGS]
  } else {
    return name;
  }
}

export function getLabelByExitCode(exitCode?: number | null) {
  return isNil(exitCode) ? 'Pending' : exitCode === 0 ? 'Success' : 'Failed'
}