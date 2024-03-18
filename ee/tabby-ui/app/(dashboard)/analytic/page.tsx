import { Metadata } from 'next'

import { Analytic } from './components/analytic'

export const metadata: Metadata = {
  title: 'Analytic dashboard'
}

export default function Page() {
  return <Analytic />
}
