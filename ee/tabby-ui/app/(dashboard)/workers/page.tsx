import { Metadata } from 'next'

import Workers from './components/workers'

export const metadata: Metadata = {
  title: 'Workers'
}

export default function IndexPage() {
  return <Workers />
}
