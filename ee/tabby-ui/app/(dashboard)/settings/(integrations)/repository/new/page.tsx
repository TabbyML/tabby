import { Metadata } from 'next'

import { NewRepository } from './components/new-page'

export const metadata: Metadata = {
  title: 'New Repository'
}

export default function IndexPage() {
  return <NewRepository />
}
