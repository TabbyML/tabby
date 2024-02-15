import { SubHeader } from '@/components/sub-header'

import { JobRuns } from './job-list'

export default function Jobs() {
  return (
    <>
      <SubHeader>Job runs</SubHeader>
      <JobRuns />
    </>
  )
}
