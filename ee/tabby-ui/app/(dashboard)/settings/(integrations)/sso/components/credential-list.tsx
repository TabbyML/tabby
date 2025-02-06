'use client'

import React, { useMemo } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { compact, find } from 'lodash-es'
import { useQuery } from 'urql'

import {
  AuthProviderKind,
  LdapCredentialQuery,
  LicenseType,
  OAuthCredentialQuery,
  OAuthProvider
} from '@/lib/gql/generates/graphql'
import { ldapCredentialQuery, oauthCredential } from '@/lib/tabby/query'
import { Button, buttonVariants } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  IconGitHub,
  IconGitLab,
  IconGoogle,
  IconUsers
} from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import { LicenseGuard } from '@/components/license-guard'
import LoadingWrapper from '@/components/loading-wrapper'

import { PROVIDER_METAS } from '../constant'

export const CredentialList = () => {
  const authProviderKindCount = useMemo(() => {
    return Object.keys(AuthProviderKind).length
  }, [])

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

  const [{ data: ldapData, fetching: fetchingLdap }] = useQuery({
    query: ldapCredentialQuery
  })

  const isLoading =
    fetchingGithub || fetchingGoogle || fetchingGitlab || fetchingLdap

  const credentialList = React.useMemo(() => {
    return compact([
      githubData?.oauthCredential,
      googleData?.oauthCredential,
      gitlabData?.oauthCredential,
      ldapData?.ldapCredential
    ])
  }, [githubData, googleData, gitlabData, ldapData])

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
      <div className="flex flex-col gap-8">
        {credentialList.map(credential => {
          if ('provider' in credential) {
            return (
              <OauthCredentialCard
                key={credential.provider}
                data={credential}
              />
            )
          } else {
            return <LDAPCredentialCard key="ldap" data={credential} />
          }
        })}
      </div>
      {credentialList.length < authProviderKindCount && (
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
          <span className="w-[100px] shrink-0">Host</span>
          <span className="truncate">{meta?.domain}</span>
        </div>
      </CardContent>
    </Card>
  )
}

const LDAPCredentialCard = ({
  data
}: {
  data: LdapCredentialQuery['ldapCredential']
}) => {
  if (!data) return null

  return (
    <Card>
      <CardHeader className="border-b p-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-xl">
            <IconUsers className="h-6 w-6" />
            LDAP
          </CardTitle>
          <Link
            href={`/settings/sso/detail/ldap`}
            className={buttonVariants({ variant: 'secondary' })}
          >
            View
          </Link>
        </div>
      </CardHeader>
      <CardContent className="p-0 text-sm">
        <div className="flex border-b px-8 py-4">
          <span className="w-[100px]">Type</span>
          <span>LDAP</span>
        </div>
        <div className="flex px-8 py-4">
          <span className="w-[100px] shrink-0">Host</span>
          <span className="truncate">{data?.host}</span>
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
