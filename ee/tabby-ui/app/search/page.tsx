import { Metadata } from 'next'

import Search from './components/search'

export const metadata: Metadata = {
  title: 'Search'
}

export default function SearchPage() {
  return <Search />
}
