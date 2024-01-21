import { Metadata } from 'next'

import ClusterInfo from './components/cluster'

export const metadata: Metadata = {
  title: 'Cluster Information'
}

export default function IndexPage() {
  return (
    <div className="p-6">
      <ClusterInfo />
    </div>
  )
}
