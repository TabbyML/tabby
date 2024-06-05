import { Metadata } from 'next'

import SearchGate from './components/search-gate'

export const metadata: Metadata = {
  title: 'Search'
}

export default function SearchPage() {
  return <SearchGate />
}
