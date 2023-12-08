import { Metadata } from 'next'

import Signin from './components/signin'

export const metadata: Metadata = {
  title: 'Sign In'
}

export default function Page() {
  return <Signin />
}
