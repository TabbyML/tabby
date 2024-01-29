import { Metadata } from 'next'

import { OauthCredentialList } from './components/oauth-credential-list'

export const metadata: Metadata = {
  title: 'SSO Management'
}

export default function IndexPage() {
  return <OauthCredentialList />
}
