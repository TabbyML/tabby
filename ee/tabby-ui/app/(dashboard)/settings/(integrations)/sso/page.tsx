import { Metadata } from 'next'

import { CredentialList } from './components/credential-list'

export const metadata: Metadata = {
  title: 'SSO'
}

export default function CredentialPage() {
  return <CredentialList />
}
