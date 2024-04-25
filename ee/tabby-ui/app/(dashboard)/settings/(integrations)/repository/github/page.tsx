import { Metadata } from 'next'

import GithubPage from './components/github'

export const metadata: Metadata = {
  title: 'Git Providers'
}

export default function Github() {
  return <GithubPage />
}
