import { Metadata } from 'next'

import { Analytic } from './components/analytic'

export const metadata: Metadata = {
  title: 'Analytics Dashboard'
}

export default function Page() {
  return <Analytic />
}
