import { Metadata } from 'next'

import Callback from './components/callback'

export const metadata: Metadata = {
  title: 'Integrations callback'
}

export default function Page() {
  return <Callback />
}
