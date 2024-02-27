import { Metadata } from 'next'

import General from './components/general'

export const metadata: Metadata = {
  title: 'General'
}

export default function GeneralSettings() {
  return <General />
}
