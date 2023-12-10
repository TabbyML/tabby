import { Metadata } from 'next'

import Signup from './components/signup'

export const metadata: Metadata = {
  title: 'Sign Up'
}

export default function Page() {
  return <Signup />
}
