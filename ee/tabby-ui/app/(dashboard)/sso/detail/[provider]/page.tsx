import React from 'react'
import type { NextPage } from 'next'
import { find } from 'lodash-es'

import { OAuthProvider } from '@/lib/gql/generates/graphql'

import { OAuthCredentialDetail } from '../../components/oauth-credential-detail'

type Params = {
  provider: string
}

export const PARAMS_TO_ENUM = [
  {
    name: 'github',
    enum: OAuthProvider.Github
  },
  {
    name: 'google',
    enum: OAuthProvider.Google
  }
]

export function generateStaticParams() {
  return PARAMS_TO_ENUM.map(item => ({ provider: item.name }))
}

const OAuthCredentialDetailPage: NextPage<{ params: Params }> = ({
  params
}) => {
  const provider = find(PARAMS_TO_ENUM, { name: params.provider })!.enum

  return (
    <div>
      <OAuthCredentialDetail provider={provider} />
    </div>
  )
}

export default OAuthCredentialDetailPage
