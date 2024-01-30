'use client'

import React from 'react'
import { useQuery } from "urql"
import { graphql } from '@/lib/gql/generates'
import { OAuthProvider } from "@/lib/gql/generates/graphql"
import { compact } from 'lodash-es'
import { Button } from '@/components/ui/button'
import Link from 'next/link'


export const oauthCredential = graphql(/* GraphQL */ `
  query OAuthCredential($provider: OAuthProvider!) {
    oauthCredential(provider: $provider) {
      provider
      clientId
      redirectUri
      createdAt
      updatedAt
    }
  }
`)

const OauthCredentialList = () => {

  const [{ data: githubData, fetching: fetchingGithub }] = useQuery({ query: oauthCredential, variables: { provider: OAuthProvider.Github } })
  const [{ data: googleData, fetching: fetchingGoogle }] = useQuery({ query: oauthCredential, variables: { provider: OAuthProvider.Google } })

  const isLoading = fetchingGithub || fetchingGoogle
  const credentialList = React.useMemo(() => {
    return compact([githubData?.oauthCredential, googleData?.oauthCredential])
  }, [githubData, googleData])

  if (isLoading) return <div>loading...</div> 

  if (!credentialList?.length) {
    return (
      <div>
        No Data
        <div className='flex justify-center'>
          <Link href='/sso/new'>
            <Button>Create</Button>
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div>
      {}
    </div>
  )
}

export { OauthCredentialList }