import { Metadata } from 'next'

import ClusterInfo from './components/cluster'

export const metadata: Metadata = {
  title: 'Cluster'
}

export default function IndexPage() {
  return <ClusterInfo />
}
