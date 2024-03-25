import { Metadata } from 'next'

import GitProvidersPage from './components/git-privoders-page'

export const metadata: Metadata = {
  title: 'Git Providers'
}

export default function IndexPage() {
  return <GitProvidersPage />
}
