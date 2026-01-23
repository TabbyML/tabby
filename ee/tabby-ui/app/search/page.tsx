import { Metadata } from 'next'
import { redirect } from 'next/navigation'

import { ENABLE_CHAT } from '@/lib/constants'

import { Search } from './components/search'

export const metadata: Metadata = {
  title: 'Search'
}

export default function IndexPage() {
  if (ENABLE_CHAT) {
    return <Search />
  }

  redirect('/')
}
