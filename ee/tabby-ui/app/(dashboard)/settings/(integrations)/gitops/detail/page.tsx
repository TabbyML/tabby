import { Metadata } from 'next'

import ProviderDetail from './components/detail'

export const metadata: Metadata = {
  title: 'New Provider'
}

export default function IndexPage() {
  return <ProviderDetail />
}
