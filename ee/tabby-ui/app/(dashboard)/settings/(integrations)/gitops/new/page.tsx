import { Metadata } from 'next'

import { NewProvider } from './components/new-page'

export const metadata: Metadata = {
  title: 'New Provider'
}

export default function IndexPage() {
  return <NewProvider />
}
