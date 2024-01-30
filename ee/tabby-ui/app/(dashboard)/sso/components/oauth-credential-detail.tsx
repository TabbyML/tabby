'use client'

import React from 'react'
import { pick } from 'lodash-es'
import { useQuery } from 'urql'

import { OAuthProvider } from '@/lib/gql/generates/graphql'

import OAuthCredentialForm from './oauth-credential-form'
import { oauthCredential } from './oauth-credential-list'

interface OAuthCredentialDetailProps
  extends React.HTMLAttributes<HTMLDivElement> {
  provider: OAuthProvider
}

const OAuthCredentialDetail: React.FC<OAuthCredentialDetailProps> = ({
  provider
}) => {
  const [{ data, fetching }] = useQuery({
    query: oauthCredential,
    variables: { provider }
  })

  const credential = data?.oauthCredential

  return (
    <div>
      {fetching ? (
        <div>loading</div>
      ) : (
        <OAuthCredentialForm
          provider={provider}
          defaultValues={
            credential
              ? pick(credential, [
                  'clientId',
                  'provider',
                  'clientSecret',
                  'redirectUri'
                ])
              : undefined
          }
        />
      )}
    </div>
  )
}

export { OAuthCredentialDetail }
