import { Metadata } from 'next'

import { Analytic } from './components/analyticAccptance'

export const metadata: Metadata = {
  title: 'Analytic dashboard'
}

export default function Page() {
  return <Analytic />
}