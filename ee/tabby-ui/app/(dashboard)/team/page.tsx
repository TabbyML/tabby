import { Metadata } from 'next'

import Team from './components/team'

export const metadata: Metadata = {
  title: 'User Management'
}

export default function IndexPage() {
  return (
    <div className="p-4 lg:p-16">
      <Team />
    </div>
  )
}
