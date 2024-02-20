import { Metadata } from 'next'

import JobRunsPage from './components/jobs'

export const metadata: Metadata = {
  title: 'Job runs'
}

export default function IndexPage() {
  return <JobRunsPage />
}
