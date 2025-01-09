import React from 'react'
import type { NextPage } from 'next'
import { find } from 'lodash-es'

import { PROVIDER_METAS } from '../../constant'
import { OAuthCredentialDetail } from './components/oauth-credential-detail'

type Params = {
  'oauth-provider': string
}

export function generateStaticParams() {
  return PROVIDER_METAS.map(item => ({ 'oauth-provider': item.name }))
}

const OAuthCredentialDetailPage: NextPage<{ params: Params }> = ({
  params
}) => {
  const provider = find(PROVIDER_METAS, {
    name: params['oauth-provider']
  })!.enum

  return <OAuthCredentialDetail provider={provider} />
}

export default OAuthCredentialDetailPage
