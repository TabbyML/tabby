import { Metadata } from 'next'

import Repository from './components/repository'

export const metadata: Metadata = {
  title: 'Git Repositories'
}

export default function IndexPage() {
  return <Repository />
}
