import React from 'react'
import type { NextPage } from 'next'
import { find } from 'lodash-es'

import { PROVIDER_METAS } from '../../components/constant'
import { OAuthCredentialDetail } from '../../components/oauth-credential-detail'
import { SSOHeader } from '../../components/sso-header'

type Params = {
  provider: string
}

export function generateStaticParams() {
  return PROVIDER_METAS.map(item => ({ provider: item.name }))
}

const OAuthCredentialDetailPage: NextPage<{ params: Params }> = ({
  params
}) => {
  const provider = find(PROVIDER_METAS, {
    name: params.provider
  })!.enum

  return (
    <div className="p-6">
      <SSOHeader />
      <OAuthCredentialDetail provider={provider} />
    </div>
  )
}

export default OAuthCredentialDetailPage
