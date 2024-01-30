'use client'

import React from 'react'
import Link from 'next/link'
import { compact } from 'lodash-es'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import {
  OAuthCredentialQuery,
  OAuthProvider
} from '@/lib/gql/generates/graphql'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export const oauthCredential = graphql(/* GraphQL */ `
  query OAuthCredential($provider: OAuthProvider!) {
    oauthCredential(provider: $provider) {
      provider
      clientId
      clientSecret
      redirectUri
      createdAt
      updatedAt
    }
  }
`)

const OauthCredentialList = () => {
  const [{ data: githubData, fetching: fetchingGithub }] = useQuery({
    query: oauthCredential,
    variables: { provider: OAuthProvider.Github }
  })
  const [{ data: googleData, fetching: fetchingGoogle }] = useQuery({
    query: oauthCredential,
    variables: { provider: OAuthProvider.Google }
  })

  const isLoading = fetchingGithub || fetchingGoogle
  const credentialList = React.useMemo(() => {
    return compact([githubData?.oauthCredential, googleData?.oauthCredential])
  }, [githubData, googleData])

  if (!credentialList?.length) {
    return (
      <div>
        <CardTitle className="flex justify-between items-center mb-6">
          OAuth Credentials
        </CardTitle>
        {isLoading ? (
          <div>loading...</div>
        ) : (
          <div className="border-4 border-dashed py-8 flex flex-col items-center gap-4 rounded-lg">
            <div>No Data</div>
            <div className="flex justify-center">
              <Link href="/sso/new">
                <Button>Create</Button>
              </Link>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div>
      <CardTitle className="flex justify-between items-center mb-6">
        <span>OAuth Credentials</span>
        {credentialList.length < 2 && <Button>Create</Button>}
      </CardTitle>
      <div className="flex flex-col gap-4">
        {credentialList.map(credential => {
          return (
            <OauthCredentialCard key={credential.provider} data={credential} />
          )
        })}
      </div>
    </div>
  )
}

const OauthCredentialCard = ({
  data
}: {
  data: OAuthCredentialQuery['oauthCredential']
}) => {
  return (
    <Card>
      <CardHeader className="border-b">
        <div className="flex justify-between items-center">
          <CardTitle className="text-xl">{data?.provider}</CardTitle>
          <Link href={`/sso/detail/${data?.provider.toLowerCase()}`}>
            <Button variant="secondary">View</Button>
          </Link>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex border-b py-4">
          <span className="w-[100px]">Type</span>
          <span>OAuth 2.0</span>
        </div>
        <div className="flex py-4">
          <span className="w-[100px]">Client ID</span>
          <span>{data?.clientId}</span>
        </div>
      </CardContent>
    </Card>
  )
}

export { OauthCredentialList }
