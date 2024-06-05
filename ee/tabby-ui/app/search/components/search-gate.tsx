'use client'

import { useEnableSearch } from '@/lib/experiment-flags'

import { Search } from './search'

export default function SearchGate() {
  const [searchFlag] = useEnableSearch()
  if (!searchFlag.value) {
    return <></>
  }

  return <Search />
}
