import { Metadata } from 'next'

import Team from './components/team'

export const metadata: Metadata = {
  title: 'Team Management'
}

export default function IndexPage() {
  return <Team />
}
