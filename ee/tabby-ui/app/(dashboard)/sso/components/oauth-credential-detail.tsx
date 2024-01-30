"use client"

import React from 'react'
import { OAuthProvider } from '@/lib/gql/generates/graphql'
import OAuthCredentialForm from './oauth-credential-form'
import { useQuery } from 'urql'
import { oauthCredential } from './oauth-credential-list'

interface OAuthCredentialDetailProps extends React.HTMLAttributes<HTMLDivElement> {
  provider: OAuthProvider
}


const OAuthCredentialDetail: React.FC<OAuthCredentialDetailProps> = ({ provider }) => {

  const [{ data, fetching }] = useQuery({ query: oauthCredential, variables: { provider } })

  const credential = data?.oauthCredential

  return (
    <div>
      detail
      {fetching ? <div>loading</div> : credential ? <OAuthCredentialForm provider={provider} /> : (
        <div>cta go to create or redirect</div>
      )}
    </div>
  )
}

export { OAuthCredentialDetail }
