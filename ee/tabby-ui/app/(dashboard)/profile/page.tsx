import { Metadata } from 'next'

import Profile from './components/profile'

export const metadata: Metadata = {
  title: 'Profile'
}

export default function Page() {
  return <Profile />
}
