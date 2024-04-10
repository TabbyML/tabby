import { Metadata } from 'next'

import { Report } from './components/report'

export const metadata: Metadata = {
  title: 'Reports'
}

export default function Page({
  searchParams: { sample }
}: {
  searchParams: {
    sample?: string
  }
}) {
  return <Report sample={sample === 'true'} />
}
