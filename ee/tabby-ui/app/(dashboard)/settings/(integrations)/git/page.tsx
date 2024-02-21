import { Metadata } from 'next'

import Repository from './components/repository'

export const metadata: Metadata = {
  title: 'Git Provider'
}

export default function IndexPage() {
  return <Repository />
}
