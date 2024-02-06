import { Metadata } from 'next'

import Repository from './components/repository'

export const metadata: Metadata = {
  title: 'Repository Management'
}

export default function IndexPage() {
  return <Repository />
}
