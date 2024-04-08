import { Metadata } from 'next'

import { Report } from './components/report'

export const metadata: Metadata = {
  title: 'Reports'
}

export default function Page() {
  return <Report />
}
