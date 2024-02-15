import { Metadata } from 'next'

import JobRuns from './components/jobs'

export const metadata: Metadata = {
  title: 'Job runs'
}

export default function IndexPage() {
  return <JobRuns />
}
