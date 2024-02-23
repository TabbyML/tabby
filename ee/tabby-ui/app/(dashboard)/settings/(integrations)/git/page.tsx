import { Metadata } from 'next'

import Repository from './components/repository'

export const metadata: Metadata = {
  title: 'Git Providers'
}

export default function IndexPage() {
  return <Repository />
}
