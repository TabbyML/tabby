import { Metadata } from 'next'

import { OAuthCredentialList } from './components/oauth-credential-list'

export const metadata: Metadata = {
  title: 'SSO'
}

export default function IndexPage() {
  return <OAuthCredentialList />
}
