import { Metadata } from 'next'

import ClusterInfo from './components/cluster'

export const metadata: Metadata = {
  title: 'Workers'
}

export default function IndexPage() {
  return <ClusterInfo />
}
