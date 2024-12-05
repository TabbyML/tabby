import { Metadata } from 'next'

import Activity from './components/activity'

export const metadata: Metadata = {
  title: 'Activities'
}
export default function Page() {
  return <Activity />
}
