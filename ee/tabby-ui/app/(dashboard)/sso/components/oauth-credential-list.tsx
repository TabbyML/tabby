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
import { buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'

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
          <div className="grid grid-cols-2 gap-8">
            <Skeleton className="rounded-xl h-10" />
            <Skeleton className="rounded-xl h-10" />
          </div>
        ) : (
          <div className="border-4 border-dashed py-8 flex flex-col items-center gap-4 rounded-lg">
            <div>No Data</div>
            <div className="flex justify-center">
              <Link
                href="/sso/new"
                className={buttonVariants({ variant: 'default' })}
              >
                Create
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
        {credentialList.length < 2 && (
          <Link
            href="/sso/new"
            className={buttonVariants({ variant: 'default' })}
          >
            Create
          </Link>
        )}
      </CardTitle>
      <div className="grid grid-cols-2 gap-8">
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
      <CardHeader className="border-b p-4">
        <div className="flex justify-between items-center">
          <CardTitle className="text-xl">
            {data?.provider?.toLocaleLowerCase()}
          </CardTitle>
          <Link
            href={`/sso/detail/${data?.provider.toLowerCase()}`}
            className={buttonVariants({ variant: 'secondary' })}
          >
            View
          </Link>
        </div>
      </CardHeader>
      <CardContent className="text-sm p-4">
        <div className="flex border-b py-2">
          <span className="w-[100px]">Type</span>
          <span>OAuth 2.0</span>
        </div>
        <div className="flex py-3">
          <span className="w-[100px]">Client ID</span>
          <span>{data?.clientId}</span>
        </div>
      </CardContent>
    </Card>
  )
}

export { OauthCredentialList }
