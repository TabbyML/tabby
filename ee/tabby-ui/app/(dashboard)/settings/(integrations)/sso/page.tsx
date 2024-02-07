import { Metadata } from 'next'

import { OauthCredentialList } from './components/oauth-credential-list'

export const metadata: Metadata = {
  title: 'SSO'
}

export default function IndexPage() {
  return <OauthCredentialList />
}
