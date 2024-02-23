import { Metadata } from 'next'

import Team from './components/team'

export const metadata: Metadata = {
  title: 'Members'
}

export default function IndexPage() {
  return <Team />
}
