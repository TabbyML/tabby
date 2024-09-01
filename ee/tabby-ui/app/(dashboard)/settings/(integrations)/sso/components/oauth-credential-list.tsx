'use client'

import React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { compact, find } from 'lodash-es'
import { useQuery } from 'urql'

import { graphql } from '@/lib/gql/generates'
import {
  LicenseType,
  OAuthCredentialQuery,
  OAuthProvider
} from '@/lib/gql/generates/graphql'
import { Button, buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { IconGitHub, IconGitLab, IconGoogle } from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import { LicenseGuard } from '@/components/license-guard'
import LoadingWrapper from '@/components/loading-wrapper'

import { PROVIDER_METAS } from './constant'
import { SSOHeader } from './sso-header'

export const oauthCredential = graphql(/* GraphQL */ `
  query OAuthCredential($provider: OAuthProvider!) {
    oauthCredential(provider: $provider) {
      provider
      clientId
      createdAt
      updatedAt
    }
  }
`)

const OAuthCredentialList = () => {
  const [{ data: githubData, fetching: fetchingGithub }] = useQuery({
    query: oauthCredential,
    variables: { provider: OAuthProvider.Github }
  })
  const [{ data: googleData, fetching: fetchingGoogle }] = useQuery({
    query: oauthCredential,
    variables: { provider: OAuthProvider.Google }
  })
  const [{ data: gitlabData, fetching: fetchingGitlab }] = useQuery({
    query: oauthCredential,
    variables: { provider: OAuthProvider.Gitlab }
  })

  const isLoading = fetchingGithub || fetchingGoogle || fetchingGitlab
  const credentialList = React.useMemo(() => {
    return compact([
      githubData?.oauthCredential,
      googleData?.oauthCredential,
      gitlabData?.oauthCredential
    ])
  }, [githubData, googleData, gitlabData])

  const router = useRouter()
  const createButton = (
    <LicenseGuard licenses={[LicenseType.Enterprise]}>
      {({ hasValidLicense }) => (
        <Button
          disabled={!hasValidLicense}
          onClick={() => router.push('/settings/sso/new')}
        >
          Create
        </Button>
      )}
    </LicenseGuard>
  )

  if (!credentialList?.length) {
    return (
      <div>
        <SSOHeader />
        <LoadingWrapper
          loading={isLoading}
          fallback={
            <div className="flex flex-col gap-8">
              <Skeleton className="h-[180px] w-full rounded-xl" />
              <Skeleton className="h-[180px] w-full rounded-xl" />
            </div>
          }
        >
          <div className="flex flex-col items-center gap-4 rounded-lg border-4 border-dashed py-8">
            <div>No Data</div>
            <div className="flex justify-center">
              <Link
                href="/settings/sso/new"
                className={buttonVariants({ variant: 'default' })}
              >
                Create
              </Link>
            </div>
          </div>
        </LoadingWrapper>
      </div>
    )
  }

  return (
    <div>
      <SSOHeader />
      <div className="flex flex-col gap-8">
        {credentialList.map(credential => {
          return (
            <OauthCredentialCard key={credential.provider} data={credential} />
          )
        })}
      </div>
      {credentialList.length < 3 && (
        <div className="mt-4 flex justify-end">{createButton}</div>
      )}
    </div>
  )
}

const OauthCredentialCard = ({
  data
}: {
  data: OAuthCredentialQuery['oauthCredential']
}) => {
  const meta = React.useMemo(() => {
    return find(PROVIDER_METAS, { enum: data?.provider })?.meta
  }, [data])

  if (!data) return null

  return (
    <Card>
      <CardHeader className="border-b p-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-xl">
            <OAuthProviderIcon provider={data.provider} />
            {meta?.displayName || data.provider}
          </CardTitle>
          <Link
            href={`/settings/sso/detail/${data.provider.toLowerCase()}`}
            className={buttonVariants({ variant: 'secondary' })}
          >
            View
          </Link>
        </div>
      </CardHeader>
      <CardContent className="p-0 text-sm">
        <div className="flex border-b px-8 py-4">
          <span className="w-[100px]">Type</span>
          <span>OAuth 2.0</span>
        </div>
        <div className="flex px-8 py-4">
          <span className="w-[100px] shrink-0">Domain</span>
          <span className="truncate">{meta?.domain}</span>
        </div>
      </CardContent>
    </Card>
  )
}

function OAuthProviderIcon({ provider }: { provider: OAuthProvider }) {
  switch (provider) {
    case OAuthProvider.Github:
      return <IconGitHub className="h-6 w-6" />
    case OAuthProvider.Google:
      return <IconGoogle className="h-6 w-6" />
    case OAuthProvider.Gitlab:
      return <IconGitLab className="h-6 w-6" />
    default:
      return null
  }
}

export { OAuthCredentialList }
